"""실제 PDF 없이 전처리 로직을 검증하는 자체 점검 스크립트.

pypdf/pdfplumber가 반환하는 자료구조(Word, TableBlock 등)를 손으로 흉내 낸 합성 데이터로
각 단계(2단 레이아웃 재정렬, header/footer 제거, 표 정규화, 참고문헌 제외, 청킹,
페이지 경계 문장 이어붙이기)를 단위 검증한다.

reportlab이 설치되어 있으면 실제 2단 레이아웃 + header/footer + 표가 있는 합성 PDF를
만들어 pipeline 전체(로더 포함)를 한 번 더 통합 검증한다(선택적, 없으면 건너뜀).

실행:
    python -m preprocessing.selfcheck
"""

from preprocessing.chunker import chunk_document, split_into_paragraphs, stitch_cross_page_paragraphs
from preprocessing.columns import is_two_column, reconstruct_reading_order_text
from preprocessing.headers_footers import collect_band_lines, detect_boilerplate_lines, strip_boilerplate_lines
from preprocessing.loader import Word
from preprocessing.noise_filter import drop_reference_pages, find_reference_start_page
from preprocessing.page_text import Page
from preprocessing.tables import NormalizedTable, extract_tables_for_page


def check_reference_exclusion() -> None:
    pages = [
        Page(doc_id="doc", page_number=1, text="Introduction\n\nThis paper studies KV cache."),
        Page(doc_id="doc", page_number=2, text="Method\n\nWe propose a new attention scheme."),
        Page(doc_id="doc", page_number=3, text="References\n\n[1] Someone. Some Paper. 2024."),
        Page(doc_id="doc", page_number=4, text="[2] Another. Another Paper. 2023."),
    ]

    start = find_reference_start_page(pages)
    assert start == 3, f"참고문헌 시작 페이지 탐지 실패: {start}"

    body, excluded = drop_reference_pages(pages, start)
    assert [p.page_number for p in body] == [1, 2]
    assert [p.page_number for p in excluded] == [3, 4]
    print("[OK] 참고문헌 구간 자동 탐지 및 제외")


def check_reference_heading_at_bottom_of_page() -> None:
    # 2단 레이아웃 논문에서는 본문이 끝나자마자 같은 페이지 하단에 "References" 제목만 나오는
    # 경우가 있다. 이때 그 페이지의 본문(제목 이전)은 계속 색인 대상이어야 한다.
    pages = [
        Page(doc_id="doc", page_number=1, text="Section 4.\n\nWe conclude the discussion here.\n\nReferences"),
        Page(doc_id="doc", page_number=2, text="[1] Someone. Some Paper. 2024."),
    ]

    start = find_reference_start_page(pages)
    assert start == 1, f"제목이 페이지 하단에 있어도 그 페이지에서 탐지되어야 함: {start}"

    body, excluded = drop_reference_pages(pages, start)
    assert [p.page_number for p in body] == [1], "제목 이전 본문이 유지되지 않음"
    assert "We conclude the discussion here." in body[0].text
    assert "References" not in body[0].text
    assert [p.page_number for p in excluded] == [1, 2]
    print("[OK] 페이지 하단에 있는 References 제목도 탐지하고 그 이전 본문은 보존")


def check_manual_reference_override() -> None:
    pages = [Page(doc_id="doc", page_number=i, text=f"body {i}") for i in range(1, 4)]
    manual_reference_start = 2
    body, excluded = drop_reference_pages(pages, manual_reference_start)
    assert [p.page_number for p in body] == [1]
    assert [p.page_number for p in excluded] == [2, 3]
    print("[OK] manifest의 reference_start_page 수동 지정 반영")


def check_chunking_preserves_paragraphs() -> None:
    long_paragraph = "가나다라마바사아자차카타파하. " * 100  # 단일 문단이 chunk_size(1200)보다 큼
    pages = [
        Page(doc_id="doc", page_number=1, text="짧은 문단 A.\n\n" + long_paragraph),
        Page(doc_id="doc", page_number=2, text="짧은 문단 B.\n\n짧은 문단 C."),
    ]

    paragraphs = split_into_paragraphs(pages)
    assert len(paragraphs) == 4

    chunks, review_flags = chunk_document("doc", pages, chunk_size=1200, overlap=200)
    assert len(chunks) >= 2, "청크가 최소 2개는 나와야 함"
    assert any(long_paragraph.strip() in chunk.text for chunk in chunks), "긴 문단이 잘렸음"

    for chunk in chunks:
        assert chunk.start_page <= chunk.end_page
        assert 1 <= chunk.start_page <= 2
        assert 1 <= chunk.end_page <= 2

    print(f"[OK] 청킹: 문단 경계 보존, {len(chunks)}개 청크 생성 (review_flags={len(review_flags)})")


def check_overlap_between_consecutive_chunks() -> None:
    paragraphs_text = "\n\n".join(f"문단 {i} " + ("내용 " * 30) for i in range(10))
    pages = [Page(doc_id="doc", page_number=1, text=paragraphs_text)]

    chunks, _ = chunk_document("doc", pages, chunk_size=300, overlap=100)
    assert len(chunks) >= 2

    for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
        overlap_candidate = prev_chunk.text[-100:]
        assert overlap_candidate[-20:] in next_chunk.text, "연속 청크 간 겹침이 유지되지 않음"

    print(f"[OK] 청크 간 겹침(overlap) 유지, {len(chunks)}개 청크 생성")


def check_cross_page_sentence_stitching() -> None:
    # 페이지 1의 마지막 문단이 마침표 없이 끊기고, 페이지 2 첫 문단에서 이어짐
    pages = [
        Page(doc_id="doc", page_number=1, text="Section A.\n\nThe model reduces memory by compressing"),
        Page(doc_id="doc", page_number=2, text="key and value vectors into a latent space.\n\nSection B."),
    ]
    paragraphs = split_into_paragraphs(pages)
    stitched, review_flags = stitch_cross_page_paragraphs(paragraphs)

    merged = [text for text, start, end in stitched if start == 1 and end == 2]
    assert len(merged) == 1, "페이지 경계에서 끊긴 문장이 이어붙여지지 않음"
    assert "compressing key and value vectors" in merged[0]
    assert len(review_flags) == 1
    assert review_flags[0]["reason"] == "sentence_incomplete"
    print("[OK] 페이지 경계 문장 이어붙이기 + review_flags 기록")


def check_two_column_detection_and_reorder() -> None:
    # 페이지 폭 600pt, 좌 컬럼(왼쪽) 3줄 + 우 컬럼(오른쪽) 3줄인 2단 레이아웃을 흉내낸다.
    page_width = 600.0
    words = []
    for i, token in enumerate(["Left1", "Left2", "Left3"]):
        words.append(Word(text=token, x0=50, x1=90, top=100 + i * 20, bottom=110 + i * 20))
    for i, token in enumerate(["Right1", "Right2", "Right3"]):
        words.append(Word(text=token, x0=350, x1=400, top=100 + i * 20, bottom=110 + i * 20))

    assert is_two_column(words, page_width), "2단 레이아웃이 감지되지 않음"

    text = reconstruct_reading_order_text(words, page_width)
    left_part, right_part = text.split("\n\n")
    assert left_part.splitlines() == ["Left1", "Left2", "Left3"]
    assert right_part.splitlines() == ["Right1", "Right2", "Right3"]
    print("[OK] 2단 레이아웃 감지 및 좌→우 읽기 순서 재구성")


def check_single_column_not_misdetected() -> None:
    page_width = 600.0
    words = [
        Word(text=f"word{i}", x0=50 + (i % 5) * 40, x1=80 + (i % 5) * 40, top=100 + i * 20, bottom=110 + i * 20)
        for i in range(6)
    ]
    assert not is_two_column(words, page_width), "단일 컬럼 문서가 2단으로 오탐지됨"
    print("[OK] 단일 컬럼 문서 오탐지 방지")


def check_table_exclusion_from_body_text() -> None:
    page_width = 600.0
    body_words = [Word(text="Body", x0=50, x1=90, top=100, bottom=110)]
    table_words = [Word(text="1.23", x0=50, x1=80, top=300, bottom=310)]
    table_bbox = (40.0, 290.0, 200.0, 320.0)

    text_without_table = reconstruct_reading_order_text(
        body_words + table_words, page_width, exclude_bboxes=[table_bbox]
    )
    assert "1.23" not in text_without_table
    assert "Body" in text_without_table
    print("[OK] 표 영역 단어가 본문 텍스트에서 제외됨")


def check_header_footer_detection() -> None:
    page_height = 800.0
    band_lines = []
    for page_number in range(1, 6):
        lines = [(20.0, "Conference on AI 2026"), (400.0, f"unique body line {page_number}"), (780.0, f"Page {page_number}")]
        band_lines.append(collect_band_lines(page_number, page_height, lines))

    boilerplate = detect_boilerplate_lines(band_lines)
    assert boilerplate, "반복되는 header/footer가 탐지되지 않음"

    sample_text = "Conference on AI 2026\nunique body line 1\nPage 1"
    cleaned, removed = strip_boilerplate_lines(sample_text, boilerplate)
    assert "unique body line 1" in cleaned
    assert "Conference on AI 2026" not in cleaned
    assert "Page 1" not in cleaned
    assert len(removed) == 2
    print("[OK] 반복 header/footer 탐지 및 제거")


def check_table_normalization() -> None:
    page_width = 600.0
    words = [
        Word(text="Table", x0=50, x1=80, top=90, bottom=100),
        Word(text="1.", x0=85, x1=95, top=90, bottom=100),
        Word(text="Results", x0=100, x1=140, top=90, bottom=100),
        Word(text="*p<0.05", x0=50, x1=100, top=250, bottom=260),
    ]
    from preprocessing.loader import TableBlock

    table = TableBlock(
        bbox=(40.0, 105.0, 300.0, 240.0),
        rows=[
            ["Model", "Metric", "Value"],
            ["MLA", None, "0.93"],  # 병합 셀(None) -> forward-fill 필요
            ["ITME", "Latency", "12ms"],
        ],
    )

    normalized = extract_tables_for_page("doc", 1, words, [table])
    assert len(normalized) == 1
    result: NormalizedTable = normalized[0]
    assert result.caption == "Table 1. Results"
    assert result.footnote == "*p<0.05"
    assert "| MLA | Metric | 0.93 |" in result.markdown, "병합 셀이 forward-fill 되지 않음"
    print("[OK] 표 캡션·각주 분리 및 병합 셀 forward-fill")


def check_synthetic_pdf_pipeline() -> None:
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        print("[SKIP] reportlab 미설치 - 합성 PDF 통합 검증은 건너뜀 (개발 의존성, 필수 아님)")
        return

    import tempfile
    from pathlib import Path

    from preprocessing.columns import reconstruct_reading_order_text
    from preprocessing.loader import load_pdf_raw_pages

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "synthetic.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=(600, 800))

        # 2단 레이아웃 + header/footer 흉내
        c.drawString(250, 770, "Synthetic Paper Title")
        for i in range(6):
            c.drawString(60, 700 - i * 20, f"Left column line {i}")
        for i in range(6):
            c.drawString(350, 700 - i * 20, f"Right column line {i}")
        c.drawString(280, 20, "Page 1")
        c.showPage()

        c.drawString(250, 770, "Synthetic Paper Title")
        for i in range(6):
            c.drawString(60, 700 - i * 20, f"Left column line {6 + i}")
        for i in range(6):
            c.drawString(350, 700 - i * 20, f"Right column line {6 + i}")
        c.drawString(280, 20, "Page 2")
        c.showPage()
        c.save()

        raw_pages = load_pdf_raw_pages("synthetic", pdf_path)
        assert len(raw_pages) == 2

        text = reconstruct_reading_order_text(raw_pages[0].words, raw_pages[0].width)
        assert "Left column line 0" in text
        assert "Right column line 0" in text
        left_pos = text.index("Left column line 5")
        right_pos = text.index("Right column line 0")
        assert left_pos < right_pos, "좌측 컬럼이 우측 컬럼보다 먼저 읽히지 않음"

    print("[OK] 합성 PDF(2단 레이아웃) 로더 통합 검증")


def main() -> None:
    check_reference_exclusion()
    check_reference_heading_at_bottom_of_page()
    check_manual_reference_override()
    check_chunking_preserves_paragraphs()
    check_overlap_between_consecutive_chunks()
    check_cross_page_sentence_stitching()
    check_two_column_detection_and_reorder()
    check_single_column_not_misdetected()
    check_table_exclusion_from_body_text()
    check_header_footer_detection()
    check_table_normalization()
    check_synthetic_pdf_pipeline()
    print("\n모든 자체 점검 통과")


if __name__ == "__main__":
    main()
