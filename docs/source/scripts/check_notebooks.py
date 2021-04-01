import argparse
import copy
import os
import re
import shutil
import string
import subprocess
import textwrap
from functools import partial
from pathlib import Path

import multiprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


NOTEBOOKS_PATH = "../notebooks"
INSTALL_RELEASE_VERSION_COMMAND = "! pip install podium-nlp"
INSTALL_SOURCE_VERSION_COMMAND = "# ! pip install git+https://github.com/TakeLab/podium.git"
INSTALL_SST_COMMAND = "python -c \"from podium.datasets import SST; SST.get_dataset_splits()\""
TRANS_TABLE = str.maketrans(dict.fromkeys(string.whitespace))

_re_pip_install = re.compile(r"!\s*(pip\s+install\s+[^\\\"]*)")
_re_python = re.compile(r"!\s*(python[^\\\"]*)")


def init(notebook_paths):
    all_commands = []
    for notebook_path in notebook_paths:
        with open(notebook_path, encoding="utf-8") as f:
            notebook_raw = f.read()
            commands = _re_pip_install.findall(notebook_raw) + _re_python.findall(notebook_raw)
        all_commands.extend(commands)

    delim = "&" if os.name == "nt" else ";"
    subprocess.call(
        delim.join([*all_commands, INSTALL_SOURCE_VERSION_COMMAND[4:], INSTALL_SST_COMMAND]),
        shell=True,
        cwd=Path(NOTEBOOKS_PATH).absolute(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def cleanup(snap_before_exec, snap_after_exec):
    created_paths = set(snap_after_exec) - set(snap_before_exec)
    for path in created_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def print_notebook_name_with_error(func):
    def wrapper(*args, **kwargs):
        if args:
            notebook_path = args[0]
        else:
            notebook_path = kwargs.pop("self")
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f"Error in notebook {Path(notebook_path).name}:\n{err}")
            raise
    return wrapper


def replace_install_release_with_source(nb):
    cell = nb["cells"][0]
    # sanity check
    assert cell["cell_type"] == "code"
    assert isinstance(cell["source"], str)

    assert INSTALL_RELEASE_VERSION_COMMAND in cell["source"]
    cell["source"] = cell["source"].replace(INSTALL_RELEASE_VERSION_COMMAND, "# " + INSTALL_RELEASE_VERSION_COMMAND)

    assert INSTALL_SOURCE_VERSION_COMMAND in cell["source"]
    cell["source"] = cell["source"].replace(INSTALL_SOURCE_VERSION_COMMAND, INSTALL_SOURCE_VERSION_COMMAND[2:])


@print_notebook_name_with_error
def check_notebook_output(notebook_path, env="python3", ignore_whitespace=False):
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    original_nb = nb
    new_nb = copy.deepcopy(nb)
    replace_install_release_with_source(new_nb)

    ep = ExecutePreprocessor(kernel_name=env)
    ep.preprocess(new_nb, {"metadata": {"path": str(Path(notebook_path).parent)}})

    assert len(original_nb["cells"]) == len(new_nb["cells"])

    report = []
    for i, (original_cell, new_cell) in enumerate(zip(original_nb["cells"], new_nb["cells"])):
        # consider only cells with code
        if original_cell["cell_type"] != "code" or original_cell["outputs"] == [] or original_cell["metadata"].get("elippsis"):
            continue

        # sanity check
        assert isinstance(original_cell["source"], str)
        # skip cells with commands
        for line in original_cell["source"].splitlines():
            if line.strip().startswith(("!", "%")):
                continue

        # sanity check
        assert len(original_cell["outputs"]) == 1
        original_cell_stdout = original_cell["outputs"][0]["data"]["text/plain"]
        assert isinstance(original_cell_stdout, str)

        new_cell_stdout = "".join([
            new_cell_output["text"]
            for new_cell_output in new_cell["outputs"] if new_cell_output["output_type"] == "stream" and new_cell_output["name"] == "stdout"
        ])

        original_cell_stdout_ = original_cell_stdout
        new_cell_stdout_ = new_cell_stdout

        if ignore_whitespace:
            original_cell_stdout = original_cell_stdout.translate(TRANS_TABLE)
            new_cell_stdout = new_cell_stdout.translate(TRANS_TABLE)
        else:
            if new_cell_stdout[-1] == "\n" and original_cell_stdout[-1] != "\n":
                original_cell_stdout += "\n"

        if original_cell_stdout != new_cell_stdout:
            report.append((i, original_cell_stdout_, new_cell_stdout_))

    return notebook_path.name, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="python3", help="kernel that executes the notebook")
    parser.add_argument("--num_proc", help="number of processes for parallel execution")
    parser.add_argument("--ignore_whitespace", action="store_true", help="ignore whitespace when comparing cell outputs")
    parser.add_argument("--keep_artifacts", action="store_true", help="save files/directories created during execution")
    args = parser.parse_args()

    if args.num_proc is None:
        num_proc = 1
    elif args.num_proc == "auto":
        num_proc = multiprocess.cpu_count()
    else:
        num_proc = int(args.num_proc)

    notebook_paths = [
        notebook_path
        for notebook_path in Path(NOTEBOOKS_PATH).rglob("*.ipynb")
        if not (notebook_path.name.endswith("-checkpoint.ipynb") or notebook_path.parts[-2] == "examples")
    ]

    snap_before_exec = list(Path(NOTEBOOKS_PATH).iterdir())

    num_proc = min(min(num_proc, multiprocess.cpu_count()), len(notebook_paths))
    if num_proc == 1:
        reports = []
        for notebook_path in notebook_paths:
            report = check_notebook_output(notebook_path, env=args.env, ignore_whitespace=args.ignore_whitespace)
            reports.append(report)
    else:
        # install packages and predownload datasets/vectorizers to prevent parallel download
        init(notebook_paths)
        with multiprocess.Pool(num_proc) as pool:
            reports = pool.map(partial(check_notebook_output, env=args.env, ignore_whitespace=args.ignore_whitespace), notebook_paths)

    if args.keep_artifacts is False:
        snap_after_exec = list(Path(NOTEBOOKS_PATH).iterdir())
        cleanup(snap_before_exec, snap_after_exec)

    if any(report for _, report in reports):
        reports_str = "\n\n".join([
            f"In notebook {notebook}:\n" + textwrap.indent(
                "\n".join(
                    f"Cell {i}\n" + "=" * len(f"Cell {i}") + "\n" +
                    f"Original output:\n{original_output}\nAfter execution:\n{new_output}"
                    for i, original_output, new_output in report),
                " " * 4,
            )
            for notebook, report in reports if len(report) > 0
        ])
        raise Exception(
            "❌❌ Mismatches found in the outputs of the notebooks:\n\n" + reports_str
        )
