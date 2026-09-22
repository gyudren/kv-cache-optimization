"""Markdown + Korean-capable PDF export; no bundled/user-shared fonts."""
from __future__ import annotations
from pathlib import Path
from html import escape
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import mm


def korean_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for name in candidates:
        path = Path(name)
        if path.is_file():
            try:
                pdfmetrics.registerFont(TTFont("KoreanReport", str(path)))
                return "KoreanReport"
            except Exception:
                continue
    raise RuntimeError("Korean TrueType font not installed: cannot export valid Hangul PDF")


def _safe(text: str) -> str:
    text = escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.replace("\n", "<br/>")


def summary_fits_half_page(summary: str) -> bool:
    """Same typography and page metrics as export_report, for report-gate retry."""
    font = korean_font()
    normal = ParagraphStyle("KSummaryCheck", fontName=font, fontSize=9.4, leading=14.6,
                            spaceAfter=6, wordWrap="CJK")
    paragraph = Paragraph(_safe(summary), normal)
    _, height = paragraph.wrap(A4[0] - 48*mm, A4[1] / 2)
    return height + 30 <= A4[1] / 2 - 28*mm


def export_report(report_md: str, output_dir: str, stem: str) -> dict:
    if not report_md.strip() or "## SUMMARY\n" not in report_md or "## REFERENCE\n" not in report_md:
        raise ValueError("Cannot export blank or structurally incomplete report")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    md_path, pdf_path = output / f"{stem}.md", output / f"{stem}.pdf"
    md_path.write_text(report_md, encoding="utf-8")
    font = korean_font()
    normal = ParagraphStyle("KNormal", fontName=font, fontSize=9.4, leading=14.6,
                            spaceAfter=6, textColor=colors.HexColor("#202632"), wordWrap="CJK", alignment=TA_LEFT)
    heading = ParagraphStyle("KHeading", parent=normal, fontSize=13, leading=19,
                             spaceBefore=13, spaceAfter=9, keepWithNext=True)
    summary = report_md.split("## SUMMARY\n", 1)[1].split("\n## 1. 분석 배경", 1)[0].strip()
    # Enforce physical half-page limit, not mere text count.
    if not summary_fits_half_page(summary):
        raise ValueError("SUMMARY exceeds half A4 page after rendering; shorten report before PDF export")
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=24*mm, rightMargin=24*mm,
                            topMargin=24*mm, bottomMargin=24*mm, title="KV Cache 다관점 평가")
    story = []
    for block in re.split(r"(?=^## )", report_md, flags=re.MULTILINE):
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        if lines[0].startswith("## "):
            story.append(Paragraph(_safe(lines[0][3:]), heading))
            lines = lines[1:]
        paragraph_lines = []
        table_lines = []
        def flush() -> None:
            if paragraph_lines:
                story.append(Paragraph(_safe("\n".join(paragraph_lines)), normal))
                paragraph_lines.clear()
        def flush_table() -> None:
            if not table_lines:
                return
            rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
                    for line in table_lines]
            rows = [row for row in rows if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)]
            columns = max(map(len, rows))
            cell_style = ParagraphStyle("KCell", parent=normal, fontSize=8, leading=11.5, spaceAfter=0)
            data = [[Paragraph(_safe(cell), cell_style) for cell in row + [""] * (columns-len(row))]
                    for row in rows]
            table = Table(data, colWidths=[(A4[0]-48*mm)/columns]*columns, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bac6d2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 8)])
            table_lines.clear()
        for line in lines:
            if line.strip().startswith("|") and line.strip().endswith("|"):
                flush()
                table_lines.append(line)
                continue
            flush_table()
            if line.startswith("### "):
                flush()
                story.append(Paragraph(_safe(line[4:]), heading))
            elif line.strip():
                paragraph_lines.append(line)
            else:
                flush()
        flush()
        flush_table()
        story.append(Spacer(1, 5))
    doc.build(story)
    return {"md": str(md_path), "pdf": str(pdf_path)}
