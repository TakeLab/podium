import argparse
import multiprocessing
import textwrap
from functools import partial
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


NOTEBOOKS_PATH = "../notebooks"
INSTALL_RELEASE_VERSION_COMMAND = "! pip install podium-nlp\n"
INSTALL_SOURCE_VERSION_COMMAND = "# ! pip install git+https://github.com/TakeLab/podium.git\n"


def replace_install_release_with_source(nb):
    cell = nb.cells[0]
    # sanity check
    assert cell["cell_type"] == "code"
    assert isinstance(cell.source, list)

    irv_idx = cell["source"].index(INSTALL_RELEASE_VERSION_COMMAND)
    cell["source"][irv_idx] = "# " + cell["source"][irv_idx]

    isv_idx = cell["source"].index(INSTALL_SOURCE_VERSION_COMMAND)
    cell["source"][isv_idx] = cell["source"][isv_idx][cell["source"][isv_idx].index("!"):]


def check_notebook_output(notebook_path, env="python3"):
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    original_nb = nb.copy()
    ep = ExecutePreprocessor(kernel_name=env)
    new_nb = nb
    replace_install_release_with_source(new_nb)
    try:
        ep.preprocess(new_nb, {"metadata": {"path": str(Path(notebook_path).parent)}})
    except CellExecutionError:
        print(f"Error happened while executing the notebook {notebook_path.name}")
        raise

    report = []
    assert len(original_nb["cells"]) == len(new_nb["cells"])
    for i, (original_cell, new_cell) in enumerate(zip(original_nb["cells"], new_nb["cell"])):
        # consider only cells with code
        if original_cell["cell_type"] != "code":
            continue

        # sanity check
        assert isinstance(original_cell["source"], list)
        # skip cells with commands
        for line in original_cell["source"]:
            if line.strip().startswith(("!", "%")):
                continue

        # sanity check
        assert isinstance(original_cell["outputs"]["data"]["text/plain"], list)
        original_cell_stdout = "".join(original_cell["outputs"]["data"]["text/plain"])

        new_cell_stdout = "".join([
            new_cell_output["text"]
            for new_cell_output in new_cell["outputs"] if new_cell_output["name"] == "stdout"
        ])

        if original_cell_stdout != new_cell_stdout:
            report.append(i, original_cell_stdout, new_cell_stdout)

    return notebook_path.name, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="kernel that executes the notebook")
    parser.add_argument("--num_proc", help="number of processes for parallel execution")
    args = parser.parse_args()

    if args.num_proc is None:
        num_proc = 1
    elif args.num_proc == "auto":
        num_proc = multiprocessing.cpu_count()
    else:
        num_proc = int(args.num_proc)

    notebook_paths = [notebook_path for notebook_path in Path(NOTEBOOKS_PATH).rglob("*.ipynb")]
    num_proc = min(min(num_proc, multiprocessing.cpu_count()), len(notebook_paths))
    if num_proc == 1:
        reports = []
        for notebook_path in notebook_paths:
            report = check_notebook_output(notebook_path, env=args.env)
            reports.append(report)
    else:
        with multiprocessing.Pool(num_proc) as pool:
            reports = pool.map(partial(check_notebook_output, env=args.env), notebook_paths)

    if any(report for _, report in reports):
        reports_str = "\n\n".join([
            f"In notebook {notebook}:\n" + textwrap.indent(
                "\n".join(
                    f"Original output:\n{original_output}\nAfter execution:\n{new_output}"
                    for original_output, new_output in report), " " * 4)
                for notebook, report in reports
        ])
        raise Exception(
            "❌❌ Found mismatches in the outputs of the notebooks:\n\n" + reports_str
        )
