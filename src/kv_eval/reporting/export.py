"""Markdown + Korean-capable PDF export; no bundled/user-shared fonts.

보고서 본문은 Markdown으로 생성되므로 PDF에서도 표·소제목·목록·굵게를 해석해 렌더링한다
(설계 E의 대조표·신호표·상충표·증거 균형표가 파이프 문자열로 찍히지 않도록).
"""
from __future__ import annotations
from datetime import date
from glob import glob
from pathlib import Path
from html import escape
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import mm

REPORT_TITLE = "KV cache 최적화 기술 다관점 평가 보고서"
REPORT_SUBTITLE = "DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙"
REPORT_AUTHORS = "판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건"

PAGE_WIDTH = A4[0] - 48*mm
INK = colors.HexColor("#202632")
MUTED = colors.HexColor("#5b6472")
RULE = colors.HexColor("#c9cfd8")
HEADER_FILL = colors.HexColor("#eef1f6")

# (regular, regular subfont, bold, bold subfont). Bold가 없으면 regular로 대체한다.
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 0,
     "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", 0),
    *[(path, 0, path, 1) for path in sorted(glob(
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/NanumGothic.ttc"))],
    ("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf", 0,
     "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf", 0),
    ("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf", 0,
     "/usr/share/fonts/truetype/unfonts-core/UnDotumBold.ttf", 0),
    ("/Library/Fonts/Arial Unicode.ttf", 0, None, 0),
]
_FONTS: tuple[str, str] | None = None


def korean_fonts() -> tuple[str, str]:
    """(regular, bold) 폰트 이름. 한글 TrueType 폰트가 없으면 PDF를 만들지 않는다."""
    global _FONTS
    if _FONTS is not None:
        return _FONTS
    from reportlab.lib.fonts import addMapping
    for regular, r_index, bold, b_index in _FONT_CANDIDATES:
        if not Path(regular).is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont("KoreanReport", regular, subfontIndex=r_index))
        except Exception:
            continue
        bold_name = "KoreanReport"
        if bold and Path(bold).is_file():
            try:
                pdfmetrics.registerFont(TTFont("KoreanReportBold", bold, subfontIndex=b_index))
                bold_name = "KoreanReportBold"
            except Exception:
                pass
        # <b> 태그가 굵은 한글 폰트로 매핑되도록 패밀리를 등록한다.
        addMapping("KoreanReport", 0, 0, "KoreanReport")
        addMapping("KoreanReport", 1, 0, bold_name)
        addMapping("KoreanReport", 0, 1, "KoreanReport")
        addMapping("KoreanReport", 1, 1, bold_name)
        _FONTS = ("KoreanReport", bold_name)
        return _FONTS
    raise RuntimeError("Korean TrueType font not installed: cannot export valid Hangul PDF")


def korean_font() -> str:
    return korean_fonts()[0]


def _styles() -> dict[str, ParagraphStyle]:
    font, bold = korean_fonts()
    normal = ParagraphStyle("KNormal", fontName=font, fontSize=9.4, leading=14.6, spaceAfter=6,
                            textColor=INK, wordWrap="CJK", alignment=TA_LEFT)
    return {
        "normal": normal,
        "bullet": ParagraphStyle("KBullet", parent=normal, leftIndent=11, bulletIndent=2, spaceAfter=3),
        "h2": ParagraphStyle("KH2", parent=normal, fontName=bold, fontSize=13.5, leading=19,
                             spaceBefore=14, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("KH3", parent=normal, fontName=bold, fontSize=11, leading=16,
                             spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "h4": ParagraphStyle("KH4", parent=normal, fontName=bold, fontSize=9.8, leading=15,
                             spaceBefore=6, spaceAfter=3, keepWithNext=True),
        "cell": ParagraphStyle("KCell", parent=normal, fontSize=8.2, leading=11.6, spaceAfter=0),
        "head": ParagraphStyle("KHead", parent=normal, fontName=bold, fontSize=8.2, leading=11.6, spaceAfter=0),
        "title": ParagraphStyle("KTitle", parent=normal, fontName=bold, fontSize=17, leading=23,
                                alignment=TA_CENTER, spaceAfter=4),
        "subtitle": ParagraphStyle("KSub", parent=normal, fontSize=10, leading=14, alignment=TA_CENTER,
                                   textColor=MUTED, spaceAfter=2),
    }


def _inline(text: str) -> str:
    """Markdown 인라인(**굵게**, `코드`)을 reportlab 마크업으로 바꾼다. 나머지는 이스케이프."""
    out = escape(text.strip(), quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", out)
    out = re.sub(r"`([^`]+)`", r"\1", out)
    return out


_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_BULLET = re.compile(r"^(\s*)([-*•]|\d+[.)])\s+(.*)$")


def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _table(rows: list[list[str]], styles: dict) -> Table:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # 글자 수에 비례해 열 폭을 나누되 한 열이 너무 좁아지지 않게 한다.
    lengths = [max(min(len(r[i]), 60) for r in rows) + 4 for i in range(width)]
    total = sum(lengths)
    widths = [max(PAGE_WIDTH * n / total, 16*mm) for n in lengths]
    scale = PAGE_WIDTH / sum(widths)
    widths = [w * scale for w in widths]
    data = [[Paragraph(_inline(c), styles["head" if i == 0 else "cell"]) for c in r]
            for i, r in enumerate(rows)]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def markdown_flowables(text: str, styles: dict | None = None) -> list:
    """보고서 Markdown 일부를 flowable 목록으로 바꾼다(표·소제목·목록·문단)."""
    styles = styles or _styles()
    story: list = []
    paragraph: list[str] = []
    lines = text.splitlines()

    def flush() -> None:
        if paragraph:
            story.append(Paragraph("<br/>".join(_inline(x) for x in paragraph), styles["normal"]))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
        elif _TABLE_ROW.match(line) and i + 1 < len(lines) and _TABLE_RULE.match(lines[i + 1]):
            flush()
            rows = [_cells(line)]
            i += 2
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                rows.append(_cells(lines[i]))
                i += 1
            story.append(Spacer(1, 2))
            story.append(_table(rows, styles))
            story.append(Spacer(1, 7))
            continue
        elif stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            style = styles["h2" if level <= 2 else "h3" if level == 3 else "h4"]
            story.append(Paragraph(_inline(stripped.lstrip("#")), style))
        elif (match := _BULLET.match(line)):
            flush()
            marker = "•" if match.group(2) in "-*•" else match.group(2)
            indent = min(len(match.group(1)) // 2, 3) * 10
            style = ParagraphStyle(f"KBullet{indent}", parent=styles["bullet"],
                                   leftIndent=styles["bullet"].leftIndent + indent,
                                   bulletIndent=styles["bullet"].bulletIndent + indent)
            story.append(Paragraph(_inline(match.group(3)), style, bulletText=marker))
        elif stripped in ("---", "***"):
            flush()
            story.append(Spacer(1, 6))
        else:
            paragraph.append(stripped)
        i += 1
    flush()
    return story


def summary_fits_half_page(summary: str) -> bool:
    """Same typography and page metrics as export_report, for report-gate retry."""
    height = 0.0
    for flowable in markdown_flowables(summary):
        _, h = flowable.wrap(PAGE_WIDTH, A4[1])
        height += h + getattr(flowable.style, "spaceAfter", 0) if hasattr(flowable, "style") else h
    return height + 30 <= A4[1] / 2 - 28*mm


def _page_footer(canvas, doc) -> None:
    font, _ = korean_fonts()
    canvas.saveState()
    canvas.setFont(font, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(24*mm, 12*mm, REPORT_TITLE)
    canvas.drawRightString(A4[0] - 24*mm, 12*mm, f"{doc.page}")
    canvas.restoreState()


def export_report(report_md: str, output_dir: str, stem: str) -> dict:
    if not report_md.strip() or "## SUMMARY\n" not in report_md or "## REFERENCE\n" not in report_md:
        raise ValueError("Cannot export blank or structurally incomplete report")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    md_path, pdf_path = output / f"{stem}.md", output / f"{stem}.pdf"
    header = f"# {REPORT_TITLE}\n\n{REPORT_SUBTITLE}\n\n{REPORT_AUTHORS} · {date.today().isoformat()}\n\n"
    md_path.write_text(header + report_md, encoding="utf-8")
    styles = _styles()
    summary = report_md.split("## SUMMARY\n", 1)[1].split("\n## 1. 분석 배경", 1)[0].strip()
    # 물리적 1/2페이지 제한은 보고서 게이트(validate_report)가 판정해 재작성을 요구한다.
    # 한도 소진 후에도 초과하면 validation.json에 이슈로 남기고 PDF는 그대로 만든다.
    fits = summary_fits_half_page(summary)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=24*mm, rightMargin=24*mm,
                            topMargin=22*mm, bottomMargin=22*mm, title=REPORT_TITLE,
                            author=REPORT_AUTHORS)
    story: list = [
        Paragraph(escape(REPORT_TITLE), styles["title"]),
        Paragraph(escape(REPORT_SUBTITLE), styles["subtitle"]),
        Paragraph(escape(f"{REPORT_AUTHORS} · {date.today().isoformat()}"), styles["subtitle"]),
        Spacer(1, 6),
        Table([[""]], colWidths=[PAGE_WIDTH], rowHeights=[0.1],
              style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.8, RULE)])),
    ]
    story.extend(markdown_flowables(report_md, styles))
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return {"md": str(md_path), "pdf": str(pdf_path), "summary_fits_half_page": fits}
