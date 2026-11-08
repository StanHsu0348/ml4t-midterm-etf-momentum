"""Render report/proposal.md -> report/proposal.html -> report/proposal.pdf (headless Chrome).

Placeholders inside the markdown are filled from results/tables so no number is
transcribed by hand:
  {{table:FILE.csv}}                       whole table
  {{table:FILE.csv:col1,col2,...}}         selected columns
  {{json:FILE.json:key:fmt}}               one value, e.g. {{json:05_inference.json:net_sharpe:.2f}}
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
SRC = ROOT / "report" / sys.argv[1] if len(sys.argv) > 1 else ROOT / "report" / "proposal.md"


def fmt_cell(v):
    if isinstance(v, str):
        return v.replace("|", "\\|")
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:.3f}".rstrip("0").rstrip(".") if v == v else ""
    return "" if v is None or (isinstance(v, float) and v != v) else str(v)


def table(spec: str) -> str:
    parts = spec.split(":")
    df = pd.read_csv(TAB / parts[0])
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: ""})
    if len(parts) > 1 and parts[1]:
        df = df[parts[1].split(",")]
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = "".join("| " + " | ".join(fmt_cell(v) for v in r) + " |\n" for r in df.itertuples(index=False))
    return "\n" + head + rows + "\n"


def jval(spec: str) -> str:
    f, key, fmt = spec.split(":", 2)
    v = json.loads((TAB / f).read_text())
    for k in key.split("."):
        v = v[k]
    return format(float(v), fmt) if fmt else str(v)


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "PingFang TC", "Heiti TC", "Noto Sans CJK TC", sans-serif; font-size: 10pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 4pt; } h2 { font-size: 13pt; border-bottom: 1.5px solid #333; padding-bottom: 2pt; margin-top: 14pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; } p { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.4pt; line-height: 1.25; margin: 4pt 0; } tr { page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1pt 3pt; text-align: left; vertical-align: top; } th { background: #eef2f6; }
img { max-width: 100%; display: block; margin: 4pt auto; page-break-inside: avoid; }
.fig { text-align: center; font-size: 8.5pt; color: #444; margin-bottom: 6pt; page-break-inside: avoid; }
.break { page-break-before: always; } .wcb { background: #fff6e6; border-left: 3px solid #e6a23c; padding: 4pt 8pt; margin: 6pt 0; font-size: 9pt; }
code { font-size: 8.5pt; background: #f4f4f4; padding: 0 2pt; } .meta { color: #555; font-size: 9pt; }
blockquote { margin: 4pt 0; padding: 2pt 8pt; border-left: 3px solid #4C72B0; background: #f3f6fb; }
"""

if __name__ == "__main__":
    md = SRC.read_text()
    md = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: table(m.group(1)), md)
    md = re.sub(r"\{\{json:([^}]+)\}\}", lambda m: jval(m.group(1)), md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "md_in_html", "attr_list"])
    html = f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>{SRC.stem}</title><style>{CSS}</style></head><body>{body}</body></html>"
    out = SRC.with_suffix(".html")
    out.write_text(html)
    import os
    chrome = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if Path(chrome).exists():
        pdf = SRC.with_suffix(".pdf")
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                       check=True, capture_output=True)
        print("wrote", pdf)
    else:
        print("wrote", out, "(Chrome not found: open the HTML and print to PDF)")
