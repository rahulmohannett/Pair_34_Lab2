"""Build a notebook from a "# %%" source file.

Usage: python3 src/build_notebook.py src/p2_train_source.py notebooks/p2_train.ipynb
Cells are separated by lines starting with "# %%"; "# %% [markdown]" cells hold markdown in comment lines.
"""
import json
import sys


def _keep_executed(nb_path):
    """Copy a notebook that holds run outputs to notebooks/versions/ before it is overwritten."""
    import os, shutil, time
    if not os.path.exists(nb_path):
        return
    old = json.load(open(nb_path, encoding="utf-8"))
    if not any(c.get("outputs") for c in old.get("cells", [])):
        return
    stamp = time.strftime("%Y%m%d_%H%M", time.localtime(os.path.getmtime(nb_path)))
    vdir = os.path.join(os.path.dirname(nb_path), "versions"); os.makedirs(vdir, exist_ok=True)
    dest = os.path.join(vdir, f"{os.path.basename(nb_path)[:-6]}_executed_{stamp}.ipynb")
    if not os.path.exists(dest):
        shutil.copy2(nb_path, dest); print("Saved the executed notebook to", dest)


def build(src_path, nb_path):
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        body = lines[:]
        while body and not body[-1].strip():
            body.pop()
        if kind == "markdown":
            body = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in body]
        src = [l + "\n" for l in body]
        if src:
            src[-1] = src[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in open(src_path, encoding="utf-8").read().split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    flush()
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU",
                       "colab": {"gpuType": "T4", "provenance": []}}}
    _keep_executed(nb_path)
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{nb_path}: {len(cells)} cells")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
