import math
import os
import pickle
import time
from datetime import timedelta
from pathlib import Path
from typing import Type, Protocol

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


class InfoCallback(Protocol):
    def __call__(self, msg: str, *args) -> None:
        ...


class DatasetManager:
    __slots__ = (
        "dataset_path",
        "target",
        "split_names",
        "cache_dir",
        "tokenizer",
        "info",
        "max_line_len",
        "max_file_len",
    )
    NUM_TOKENS_IN_4_LINES = 64
    DATASET_PAGE_SIZE = 10_000
    CACHE_FILE_EXT = ".pt"

    def __init__(
        self,
        tokenizer_class: Type[PreTrainedTokenizerFast],
        tokenizer_name: str,
        dataset_path: str,
        target: str,
        split_names: tuple[str, ...],
        cache_dir: str,
        info: InfoCallback,
        max_line_len,
        max_file_len,
    ):
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        self.dataset_path = dataset_path
        assert target in ["line", "file"]
        self.target = target
        self.split_names = split_names
        self.cache_dir = f"{cache_dir}-{max_file_len}-{max_line_len}"
        self.info = info
        self.max_line_len = max_line_len
        self.max_file_len = max_file_len

    def load_dataset(self) -> tuple[TensorDataset, ...]:
        """
        This function loads the dataset from the cache if it exists.
        Otherwise, it reads the dataset from the file and caches it.
        """
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        assert os.path.isdir(self.dataset_path)
        cache_filepath_for, filepath_for = self._get_file_and_cache_path()

        last_modified_time_filepath = os.path.join(
            self.cache_dir, "last_modified_time.txt"
        )
        last_modified_time = max(map(os.path.getmtime, filepath_for.values()))
        cached_dataset = self._try_reading_cache(
            cache_filepath_for,
            last_modified_time_filepath,
            last_modified_time,
        )
        if cached_dataset:
            return cached_dataset

        self.info("Reading dataset.")

        tensor_datasets = self._read_and_cache(
            filepath_for,
            cache_filepath_for,
            last_modified_time_filepath,
            last_modified_time,
        )

        return tensor_datasets

    def _get_file_and_cache_path(self):
        """
        This function returns a tuple of two dictionaries.
        The first dictionary maps the split name to the cache filepath of the file.
        The second dictionary maps the split name to the actual filepath of the file.
        """
        filepath_for = {}
        cache_filepath_for = {}
        for filename in os.listdir(self.dataset_path):
            split_name = filename.split(".")[0]
            if split_name in self.split_names:
                filepath_for[split_name] = os.path.join(self.dataset_path, filename)
                cache_filepath_for[split_name] = os.path.join(
                    self.cache_dir, f"{split_name}{self.CACHE_FILE_EXT}"
                )
        return cache_filepath_for, filepath_for

    def _try_reading_cache(
        self,
        cache_filepath_for: dict[str, str],
        last_modified_time_filepath: str,
        last_modified_time: float,
    ):
        """
        This function tries to read the dataset from cache_filepath_for.
        If the last modified time of the files in filepath_for is the same as the last modified time,
        it returns the dataset from cache. Otherwise, it returns None.
        """
        if os.path.exists(last_modified_time_filepath):
            with open(
                last_modified_time_filepath, "r", encoding="utf8"
            ) as last_modified_time_file:
                prev_last_modified_time = float(last_modified_time_file.read())
            if math.isclose(prev_last_modified_time, last_modified_time, abs_tol=1e-6):
                self.info("Loading dataset from cache")
                dataset_splits = tuple(
                    torch.load(cache_filepath_for[split_name])
                    for split_name in self.split_names
                )

                return dataset_splits

            self.info("Cache outdated.")

        self.info("Cache miss.")

        return None

    def _read_and_cache(
        self,
        filepath_for,
        cache_filepath_for,
        last_modified_time_filepath,
        last_modified_time,
    ) -> tuple[TensorDataset, ...]:
        """
        This function reads the dataset from the filepath_for and caches it in cache_filepath_for.
        It also writes the last_modified_time to last_modified_time_filepath.
        Parameters
        ----------
        filepath_for : Dict[split_name, filepath for split_name]
        cache_filepath_for : Dict[split_name, cache filepath for split_name]
        last_modified_time_filepath : filepath to write last modified time
        last_modified_time : float the last modified time of the files in filepath_for

        Returns a tuple of TensorDataset for each split_name
        -------

        """
        tick = time.perf_counter_ns()
        split_dataframes: dict[str, pd.DataFrame] = {
            split_name: pd.read_parquet(filepath_for[split_name])
            for split_name in self.split_names
        }

        split_tensor_datasets: dict[str, tuple[TensorDataset, list[list[str]]]]
        if "linedp" in self.dataset_path:
            split_tensor_datasets = {
                split_name: self._tokenize_linedp(
                    split_dataframe, split_name=split_name
                )
                for split_name, split_dataframe in split_dataframes.items()
            }
        else:
            split_tensor_datasets = {
                split_name: self._tokenize_defector(
                    split_dataframe, split_name=split_name
                )
                for split_name, split_dataframe in split_dataframes.items()
            }
        for split_name, (
            tensor_dataset,
            *file_content,
        ) in split_tensor_datasets.items():
            torch.save(tensor_dataset, cache_filepath_for[split_name])

            file_content_file_path = cache_filepath_for[split_name].replace(
                ".pt", ".pickle"
            )
            with open(file_content_file_path, "wb") as file_content_file:
                pickle.dump(file_content, file_content_file)

        with open(
            last_modified_time_filepath, "w", encoding="utf8"
        ) as last_modified_time_file:
            last_modified_time_file.write(str(last_modified_time))

        tock = time.perf_counter_ns()
        elapsed = timedelta(microseconds=(tock - tick) / 1000)
        self.info(f"Tokenized dataset in {elapsed}")

        return tuple(
            tensor_dataset for tensor_dataset, *_ in split_tensor_datasets.values()
        )

    def _tokenize_defector(self, dataframe: pd.DataFrame, split_name: str):
        """
        Tokenizes the defector dataset.
        The output is a TensorDataset with the following tensors:
        - source: The tokenized source code
        The shape is (dataset_size, max_file_len, max_line_len).
        - target: The tokenized target code.
        For line-level prediction, its shape is (dataset_size, max_file_len).
        For file-level prediction, its shape is (dataset_size,).
        - actual_len: Only for line-level prediction. The actual length of the file.
        It helps during the evaluation to ignore lines that are padded or truncated.
        """
        file_info = []
        file_content = []
        mask = []
        for i, row in dataframe.iterrows():
            content = row["content"]
            try:
                lines = (
                    content.decode("utf-8")
                    .replace(self.tokenizer.eos_token, "")
                    .split("\n")
                )

                file_content.append(lines)
                file_info.append((row["repo"], row["commit"], row["filepath"]))
                mask.append(True)
            except (UnicodeError, AttributeError, ValueError):
                mask.append(False)

        tokenization_progress = tqdm(
            [
                (lines, self.tokenizer, self.max_line_len, self.max_file_len)
                for lines in file_content
            ],
            desc=f"Tokenizing {split_name}",
            smoothing=0.001,
        )
        file_tensors = tuple(
            file_chunk
            # _tokenize_lines returns a list of tensors for each file
            for file_chunks in map(_tokenize_lines, tokenization_progress)
            for file_chunk in file_chunks
        )
        source_tensor = torch.stack(file_tensors)
        self.info(f"Tokenization complete: {len(dataframe)=}, {source_tensor.shape=}")
        assert source_tensor.shape[1:] == (
            self.max_file_len,
            self.max_line_len,
        )

        if self.target == "line":
            buggy_lines_of_file = [
                # use only the lines for which we could decode the file content
                # keep the lines as a set for faster lookup
                set(lines.tolist())
                for lines in dataframe["lines"].iloc[mask]
            ]
            buggy_line_matrix = []
            for i, buggy_lines in enumerate(buggy_lines_of_file):
                file_len = len(file_content[i])
                for start in range(1, file_len + 1, self.max_file_len - 64):
                    if start > 1 and start + 64 > file_len:
                        # all lines of this chunk are already in the previous chunk
                        continue
                    buggy_line_matrix.append(
                        [
                            i in buggy_lines
                            for i in range(start, start + self.max_file_len)
                        ]
                    )

            target_tensor = torch.tensor(buggy_line_matrix)

            assert target_tensor.shape[1:] == (self.max_file_len,), target_tensor.shape
            assert (
                source_tensor.shape[0] == target_tensor.shape[0]
            ), f"{source_tensor.shape=} {target_tensor.shape=}"

            tensor_dataset = TensorDataset(source_tensor, target_tensor)
        elif self.target == "file":
            # use only the lines for which we could decode the file content
            is_buggy_file = dataframe["lines"].iloc[mask].str.len() > 0
            target_tensor = torch.tensor(is_buggy_file.to_list())
            assert target_tensor.shape == (len(file_content),), target_tensor.shape
            tensor_dataset = TensorDataset(source_tensor, target_tensor)
        else:
            raise ValueError(f"Unknown target {self.target}")

        fixed_len_file_content = self._truncate_or_pad(file_content)
        return tensor_dataset, fixed_len_file_content, file_info

    def _tokenize_linedp(self, dataframe: pd.DataFrame, split_name: str):
        """
        Tokenizes the linedp dataset.
        The output is a TensorDataset with the following tensors:
        - source: The tokenized source code
        The shape is (dataset_size, max_file_len, max_line_len).
        - target: The tokenized target code.
        For line-level prediction, its shape is (dataset_size, max_file_len).
        For file-level prediction, its shape is (dataset_size,).
        - actual_len: Only for line-level prediction. The actual length of the file.
        It helps during the evaluation to ignore lines that are padded or truncated.
        """
        dataframe = dataframe.groupby(
            dataframe["repo"] + dataframe["filename"], sort=False
        )
        dataframe.apply(lambda grp: grp.sort_values("line_number", inplace=True))

        # file_content contains list[list[line_of_code]]
        file_content = (
            dataframe["code_line"]
            .apply(
                lambda grp: grp.str.replace(
                    self.tokenizer.eos_token, "", regex=False
                ).to_list()
            )
            .to_list()
        )
        file_info = dataframe[["repo", "filename"]].first().values.tolist()

        tokenization_progress = tqdm(
            [
                (lines, self.tokenizer, self.max_line_len, self.max_file_len)
                for lines in file_content
            ],
            desc=f"Tokenizing {split_name}",
            smoothing=0.001,
        )
        file_tensors = tuple(
            file_chunk
            # _tokenize_lines returns a list of tensors for each file
            for file_chunks in map(_tokenize_lines, tokenization_progress)
            for file_chunk in file_chunks
        )
        source_tensor = torch.stack(file_tensors)
        self.info(f"Tokenization complete: {source_tensor.shape=}")
        assert source_tensor.shape[1:] == (
            self.max_file_len,
            self.max_line_len,
        )

        buggy_lines_of_file = (
            # dataframe is grouped by repo+filename and sorted by line_number
            dataframe["line-label"]
            .apply(lambda grp: grp.to_list())
            .to_list()
        )

        if self.target == "line":
            buggy_line_matrix = []
            for buggy_lines in buggy_lines_of_file:
                padding_len = self.max_file_len - len(buggy_lines)
                if padding_len < 0:
                    # split the buggy lines into chunks of size max_file_len
                    # with a stride of 64
                    new_buggy_lines = []
                    for start in range(0, len(buggy_lines), self.max_file_len - 64):
                        new_buggy_lines.append(
                            buggy_lines[start : start + self.max_file_len]
                        )

                    if len(new_buggy_lines) > 1 and len(new_buggy_lines[-1]) <= 64:
                        new_buggy_lines.pop()
                    if len(new_buggy_lines[-1]) < self.max_file_len:
                        padding = [False] * (
                            self.max_file_len - len(new_buggy_lines[-1])
                        )
                        new_buggy_lines[-1].extend(padding)
                    buggy_line_matrix.extend(new_buggy_lines)

                else:
                    padding = [False] * padding_len
                    buggy_line_matrix.append([*buggy_lines, *padding])

            target_tensor = torch.tensor(buggy_line_matrix)

            assert target_tensor.shape[1:] == (self.max_file_len,), target_tensor.shape
            assert (
                source_tensor.shape[0] == target_tensor.shape[0]
            ), f"{source_tensor.shape=} {target_tensor.shape=}"

            tensor_dataset = TensorDataset(source_tensor, target_tensor)
        elif self.target == "file":
            is_buggy_file = dataframe["line-label"].apply(lambda grp: grp.any())
            target_tensor = torch.tensor(is_buggy_file.to_list())
            assert target_tensor.shape == (len(dataframe),), target_tensor.shape
            tensor_dataset = TensorDataset(source_tensor, target_tensor)
        else:
            raise ValueError(f"Unknown target: {self.target}")

        fixed_len_file_content = self._truncate_or_pad(file_content)
        return tensor_dataset, fixed_len_file_content, file_info

    def _truncate_or_pad(self, file_content):
        """
        Truncates or pads the file content to the maximum file length.
        In current revision, _truncate_or_pad is not coherent with the
        shape of the tensor dataset.
        """
        return [
            tuple(lines[: self.max_file_len])
            if len(lines) > self.max_file_len
            else (*lines, *[""] * (self.max_file_len - len(lines)))
            for lines in file_content
        ]


def _tokenize_lines(args) -> list[torch.Tensor]:
    lines, tokenizer, max_line_len, max_file_len = args
    line_token_ids = tokenizer(
        lines,
        max_length=max_line_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    # split line_token_ids into max_file_len chunks with 64 tokens overlap.
    # this is to avoid truncating lines in the middle of a function call.
    split_line_token_ids = []
    for i in range(0, len(line_token_ids), max_file_len - 64):
        split_line_token_ids.append(line_token_ids[i : i + max_file_len])
    if len(split_line_token_ids) > 1 and len(split_line_token_ids[-1]) <= 64:
        # if the last chunk is less than 64 tokens, remove the last chunk
        # since it is already included in the previous chunk
        split_line_token_ids.pop()

    # pad the last chunk with pad_token_id
    if len(split_line_token_ids[-1]) < max_file_len:
        num_padded_lines = max_file_len - len(split_line_token_ids[-1])
        line_padding = torch.tensor([[tokenizer.pad_token_id]]).repeat(
            num_padded_lines, max_line_len
        )
        split_line_token_ids[-1] = torch.cat(
            [
                split_line_token_ids[-1],
                line_padding,
            ],
            dim=0,
        )

    return split_line_token_ids
