from __future__ import annotations

import re
from functools import partial
from os.path import join
from pathlib import Path

from git import Commit as GitCommit
from pydriller import Git, Commit

PYTHON_DATASET_ROOT = (
    Path(__file__).parent.parent.joinpath("dataset").joinpath("python").resolve()
)

prepend_dataset_root = partial(join, PYTHON_DATASET_ROOT)
not_found_re = re.compile(r"SHA b\'(\w+)\' could not be resolved")

PYTHON_EXTENSIONS = ".py", ".pyi"
rename_pattern = re.compile("(.*){.+? => (.+?)}(.*)")


def get_commit_if_available(git: Git, commit_sha) -> Commit | None:
    try:
        commit = git.get_commit(commit_sha)
        return commit
    except ValueError as error:
        if not_found_re.match(str(error)):
            return None
        raise


def get_commit_from_gitpython_if_available(git: Git, commit_sha) -> GitCommit | None:
    try:
        commit = git.repo.commit(commit_sha)
        return commit
    except ValueError as error:
        if not_found_re.match(str(error)):
            return None
        raise


def is_python_script(filepath):
    if filepath is None:
        return False
    assert type(filepath) is str, type(filepath)
    return (
        any(filepath.endswith(ext) for ext in PYTHON_EXTENSIONS)
        and "test" not in filepath.lower()
    )


def commit_has_python_script(commit):
    # noinspection PyProtectedMember
    for filepath in commit._c_object.stats.files.keys():
        rename_pat_match = rename_pattern.match(filepath)
        if rename_pat_match:
            filepath = rename_pat_match[1] + rename_pat_match[2] + rename_pat_match[3]
        if is_python_script(filepath):
            return True
    return False
