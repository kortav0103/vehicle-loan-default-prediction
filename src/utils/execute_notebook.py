"""In-process Jupyter notebook executor using nbformat.

Executes every code cell sequentially from top to bottom without launching
external ZMQ kernel sockets, capturing stdout/stderr, matplotlib PNG plots,
and rich pandas HTML/text expression results directly into the notebook file.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import nbformat


def _make_show_hook(captured_images: list[str]):
    import matplotlib.pyplot as plt

    def _capture_show(*args: Any, **kwargs: Any) -> None:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
            buf.seek(0)
            captured_images.append(base64.b64encode(buf.read()).decode("ascii"))
            plt.close(fig)

    return _capture_show


def execute_notebook(nb_path: Path) -> None:
    nb_path = nb_path.resolve()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib_cache"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = nbformat.read(nb_path, as_version=4)

    ns: dict[str, Any] = {"__name__": "__main__"}
    exec_count = 0
    orig_cwd = Path.cwd()
    os.chdir(nb_path.parent)

    try:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            exec_count += 1
            cell.execution_count = exec_count
            cell.outputs = []

            src = cell.source.strip()
            if not src:
                continue

            captured_images: list[str] = []
            orig_show = plt.show
            plt.show = _make_show_hook(captured_images)

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            result_obj: Any = None
            has_expr_result = False

            try:
                tree = ast.parse(src, filename=f"cell_{idx}", mode="exec")
                last_expr = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.Expression(tree.body.pop().value)
                    ast.fix_missing_locations(last_expr)

                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    if tree.body:
                        exec(compile(tree, filename=f"cell_{idx}", mode="exec"), ns, ns)
                    if last_expr is not None:
                        result_obj = eval(compile(last_expr, filename=f"cell_{idx}_expr", mode="eval"), ns, ns)
                        has_expr_result = result_obj is not None
            except Exception:
                plt.show = orig_show
                tb_text = traceback.format_exc()
                print(f"[ERROR] Cell {idx} (exec #{exec_count}) in {nb_path.name} failed:\n{tb_text}", file=sys.stderr)
                raise
            finally:
                plt.show = orig_show

            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()
            if out_text:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=out_text))
            if err_text:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=err_text))

            for img_b64 in captured_images:
                cell.outputs.append(
                    nbformat.v4.new_output("display_data", data={"image/png": img_b64, "text/plain": "<Figure>"})
                )

            if has_expr_result:
                data: dict[str, Any] = {"text/plain": repr(result_obj)}
                if hasattr(result_obj, "_repr_html_"):
                    try:
                        html_val = result_obj._repr_html_()
                        if isinstance(html_val, str):
                            data["text/html"] = html_val
                    except Exception:
                        pass
                cell.outputs.append(
                    nbformat.v4.new_output("execute_result", execution_count=exec_count, data=data)
                )
    finally:
        os.chdir(orig_cwd)

    nbformat.validate(nb)
    nbformat.write(nb, nb_path)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.execute_notebook <notebook.ipynb> [...]", file=sys.stderr)
        return 2
    for arg in sys.argv[1:]:
        p = Path(arg)
        t0 = time.perf_counter()
        print(f"[INFO] Executing {p} in-process...")
        execute_notebook(p)
        print(f"[OK] Finished {p} in {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
