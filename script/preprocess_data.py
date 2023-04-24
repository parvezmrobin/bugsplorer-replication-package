import os
import re

import numpy as np
import pandas as pd

data_root_dir = "../dataset/linedp/"
save_dir = "../dataset/linedp/preprocessed_data/"
split_dir = '../dataset/linedp/splits/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_lvl_dir = os.path.join(data_root_dir, "File-level")
line_lvl_dir = os.path.join(data_root_dir, "Line-level")

char_to_remove = [
    "+",
    "-",
    "*",
    "/",
    "=",
    "++",
    "--",
    "\\",
    "<str>",
    "<char>",
    "|",
    "&",
    "!",
]

all_releases = {
    "activemq": [
        "activemq-5.0.0",
        "activemq-5.1.0",
        "activemq-5.2.0",
        "activemq-5.3.0",
        "activemq-5.8.0",
    ],
    "camel": ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
    "derby": ["derby-10.2.1.6", "derby-10.3.1.4", "derby-10.5.1.1"],
    "groovy": ["groovy-1_5_7", "groovy-1_6_BETA_1", "groovy-1_6_BETA_2"],
    "hbase": ["hbase-0.94.0", "hbase-0.95.0", "hbase-0.95.2"],
    "hive": ["hive-0.9.0", "hive-0.10.0", "hive-0.12.0"],
    "jruby": ["jruby-1.1", "jruby-1.4.0", "jruby-1.5.0", "jruby-1.7.0.preview1"],
    "lucene": ["lucene-2.3.0", "lucene-2.9.0", "lucene-3.0.0", "lucene-3.1"],
    "wicket": ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"],
}


def preprocess_data(proj_name):
    cur_all_rel = all_releases[proj_name]

    for rel in cur_all_rel:
        file_level_data = pd.read_csv(
            os.path.join(file_lvl_dir, rel + "_ground-truth-files_dataset.csv"), encoding="latin"
        )
        line_level_data = pd.read_csv(
            os.path.join(line_lvl_dir, rel + "_defective_lines_dataset.csv"), encoding="latin"
        )

        file_level_data = file_level_data.fillna("")

        buggy_files = list(line_level_data["File"].unique())

        preprocessed_df_list = []

        for idx, row in file_level_data.iterrows():

            filename = row["File"]

            if ".java" not in filename:
                continue

            code = row["SRC"]
            label = row["Bug"]

            code_df = create_code_df(code, filename)
            code_df["file-label"] = [label] * len(code_df)
            code_df["line-label"] = [False] * len(code_df)

            if filename in buggy_files:
                buggy_lines = list(
                    line_level_data[line_level_data["File"] == filename]["Line_number"]
                )
                code_df["line-label"] = code_df["line_number"].isin(buggy_lines)

            if len(code_df) > 0:
                preprocessed_df_list.append(code_df)

        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(os.path.join(save_dir, rel + ".csv"), index=False)
        print("finish release {}".format(rel))


def make_splits():
    train_releases = [
        release
        for releases in all_releases.values()
        for release in releases[:-2]
    ]
    val_releases = [
        releases[-2]
        for releases in all_releases.values()
    ]
    test_releases = [
        releases[-1]
        for releases in all_releases.values()
    ]

    releases_of_split = {
        "train": train_releases,
        "val": val_releases,
        "test": test_releases
    }

    for split, releases in releases_of_split.items():
        all_df = pd.concat([
            pd.read_csv(os.path.join(save_dir, release + ".csv"))
            for release in releases
        ])
        all_df.to_parquet(os.path.join(save_dir, split + ".parquet.gzip"), index=False)
        print("finish split {}".format(split))


def create_code_df(code_str, filename):
    """
    input
        code_str (string): a source code
        filename (string): a file name of source code

    output
        code_df (DataFrame): a dataframe of source code that contains the following columns
        - code_line (str): source code in a line
        - line_number (str): line number of source code line
        - is_comment (bool): boolean which indicates if a line is comment
        - is_blank_line(bool): boolean which indicates if a line is blank
    """

    df = pd.DataFrame()

    code_lines = code_str.splitlines()

    preprocess_code_lines = []
    is_comments = []
    is_blank_line = []

    comments = re.findall(r"(/\*[\s\S]*?\*/)", code_str, re.DOTALL)
    comments_str = "\n".join(comments)
    comments_list = comments_str.split("\n")

    for code_line in code_lines:
        code_line = code_line.strip()
        is_comment = is_comment_line(code_line, comments_list)
        is_comments.append(is_comment)
        # preprocess code here then check empty line...

        if not is_comment:
            code_line = preprocess_code_line(code_line)

        is_blank_line.append(is_empty_line(code_line))
        preprocess_code_lines.append(code_line)

    if "test" in filename:
        is_test = True
    else:
        is_test = False

    df["filename"] = [filename] * len(code_lines)
    df["is_test_file"] = [is_test] * len(code_lines)
    df["code_line"] = preprocess_code_lines
    df["line_number"] = np.arange(1, len(code_lines) + 1)
    df["is_comment"] = is_comments
    df["is_blank"] = is_blank_line

    return df


def is_comment_line(code_line, comments_list):
    """
    input
        code_line (string): source code in a line
        comments_list (list): a list that contains every comments
    output
        boolean value
    """

    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith("//"):
        return True
    elif code_line in comments_list:
        return True

    return False


def preprocess_code_line(code_line):
    """
    input
        code_line (string)
    """
    if __name__ == "__main__":
        code_line = re.sub("''", "'", code_line)
        code_line = re.sub('".*?"', "<str>", code_line)
        code_line = re.sub("'.*?'", "<char>", code_line)
        code_line = re.sub(r"\b\d+\b", "", code_line)
        code_line = re.sub("\\[.*?]", "", code_line)
        code_line = re.sub("[.,:;{}()]", " ", code_line)

        for char in char_to_remove:
            code_line = code_line.replace(char, " ")

    code_line = code_line.strip()

    return code_line


def is_empty_line(code_line):
    """
    input
        code_line (string)
    output
        boolean value
    """

    if len(code_line.strip()) == 0:
        return True

    return False


if __name__ == '__main__':
    for proj in list(all_releases.keys()):
        preprocess_data(proj)

    make_splits()
