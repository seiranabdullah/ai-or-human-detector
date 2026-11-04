"""Turn nb_source.py into an executed .ipynb (outputs included), without Jupyter."""
import ast, base64, contextlib, io, json, os, re, sys, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.join(os.path.dirname(__file__), "nb_source.py")
OUT = os.path.join(os.path.dirname(__file__), "AI_or_Human_Analysis.ipynb")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

text = open(SRC).read()
chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
cells = []
for i in range(1, len(chunks), 2):
    kind = "markdown" if "markdown" in chunks[i] else "code"
    body = chunks[i + 1].strip("\n")
    if kind == "markdown":
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
    cells.append((kind, body.strip("\n")))

os.chdir(ROOT)
ns = {"__name__": "__main__"}
current_outputs = []


def fake_show(*a, **k):
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close("all")
    current_outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": ["<Figure>"]}})


plt.show = fake_show


def lines(s):
    s = s.splitlines(True)
    return s


nb_cells = []
count = 0
for kind, body in cells:
    if kind == "markdown":
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(body)})
        continue
    count += 1
    current_outputs = []
    tree = ast.parse(body)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    out = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(out):
        exec(compile(tree, f"<cell {count}>", "exec"), ns)
        value = eval(compile(last, f"<cell {count}>", "eval"), ns) if last else None
    printed = out.getvalue()
    outputs = []
    if printed:
        outputs.append({"output_type": "stream", "name": "stdout", "text": lines(printed)})
    outputs += current_outputs
    if value is not None:
        data = {"text/plain": lines(repr(value))}
        if hasattr(value, "to_html"):
            data["text/html"] = lines(value.to_html())
        outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
    sys.stderr.write(f"cell {count} done in {time.time() - t0:.1f}s\n")
    nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs,
                     "source": lines(body)})

nb = {"cells": nb_cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT)
