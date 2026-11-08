"""Render report/proposal.md -> report/proposal.docx (plain academic Word version).

Uses the same placeholder filling as build_report.py, so the Word file and the
PDF carry identical numbers.
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import jval, table as _unused  # noqa: E402,F401
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
SRC = ROOT / "report" / "proposal.md"
OUT = ROOT / "report" / "proposal.docx"
LATIN, EA = "Times New Roman", "PMingLiU"
TEXT_W = 21.0 - 1.8 * 2   # A4 width minus margins (cm)


def set_font(run, size=10.5, bold=False, italic=False, mono=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = "Courier New" if mono else LATIN
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    rf.set(qn("w:eastAsia"), EA)


def inline(par, text, size=10.5, bold=False):
    """Add text with **bold**, `code` and <b>/<code> inline markup."""
    text = (text.replace("<b>", "**").replace("</b>", "**").replace("<code>", "`").replace("</code>", "`")
            .replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    text = re.sub(r"<[^>]+>", "", text)
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            set_font(par.add_run(part[2:-2]), size, True)
        elif part.startswith("`") and part.endswith("`"):
            set_font(par.add_run(part[1:-1]), size - 1, bold, mono=True)
        else:
            set_font(par.add_run(part.replace("\\|", "|")), size, bold)


def border(cell, edges):
    tcPr = cell._element.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge, sz in edges.items():
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000")
        b.append(e)
    tcPr.append(b)


def add_table(doc, rows):
    """Three-line academic table from parsed markdown rows (first row = header)."""
    ncol = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    tblPr = t._element.tblPr
    tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), "5000"); tw.set(qn("w:type"), "pct")
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblPr.append(tw)
    size = 7.5 if ncol > 6 else 8.5
    for i, r in enumerate(rows):
        for j in range(ncol):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            inline(p, r[j] if j < len(r) else "", size, bold=(i == 0))
            edges = {}
            if i == 0:
                edges = {"top": 12, "bottom": 6}
            if i == len(rows) - 1:
                edges["bottom"] = 12
            if edges:
                border(c, edges)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def md_table_rows(block):
    rows = []
    for line in block:
        if re.match(r"^\|\s*-{2,}", line) or re.match(r"^\|(-+\|)+$", line.replace(" ", "")):
            continue
        cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        rows.append([c.strip() for c in cells])
    return rows


def csv_rows(spec):
    parts = spec.split(":")
    df = pd.read_csv(TAB / parts[0])
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: ""})
    if len(parts) > 1 and parts[1]:
        df = df[parts[1].split(",")]

    def f(v):
        if isinstance(v, float):
            if v != v:
                return ""
            return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3f}".rstrip("0").rstrip(".")
        return str(v)
    return [list(map(str, df.columns))] + [[f(v) for v in r] for r in df.itertuples(index=False)]


def main():
    md = SRC.read_text()
    md = re.sub(r"\{\{json:([^}]+)\}\}", lambda m: jval(m.group(1)), md)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(1.8))
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    st = doc.styles["Normal"]
    st.font.name = LATIN; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), EA)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.line_spacing = 1.25

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1; continue
        m = re.match(r"\{\{table:([^}]+)\}\}", line.strip())
        if m:
            add_table(doc, csv_rows(m.group(1))); i += 1; continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, md_table_rows(block)); continue
        if line.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(0)
                set_font(p.add_run(lines[i]), 8, mono=True); i += 1
            i += 1; continue
        if line.startswith("# "):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(line[2:]), 15, True)
        elif line.startswith("## "):
            p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(12); p.paragraph_format.keep_with_next = True
            set_font(p.add_run(line[3:]), 12, True)
        elif line.startswith("### "):
            p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.keep_with_next = True
            set_font(p.add_run(line[4:]), 10.5, True)
        elif line.startswith('<p class="meta">'):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            inline(p, line, 9.5)
        elif line.startswith("<img"):
            src = re.search(r'src="([^"]+)"', line).group(1)
            w = re.search(r"width:(\d+)%", line)
            frac = int(w.group(1)) / 100 if w else 0.8
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str((SRC.parent / src).resolve()), width=Cm(TEXT_W * frac))
        elif line.startswith('<div class="fig">'):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            inline(p, line, 9)
        elif line.startswith('<div class="break">'):
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif line.startswith('<div class="wcb">'):
            p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(3)
            inline(p, line, 10)
        elif line.startswith(">"):
            p = doc.add_paragraph(); inline(p, line.lstrip("> "), 10)
        else:
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            inline(p, line)
        i += 1
    doc.core_properties.title = "跨資產 ETF 動能 × 波動狀態：一次完整的 ML4T 研究循環"
    doc.core_properties.author = ""
    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
