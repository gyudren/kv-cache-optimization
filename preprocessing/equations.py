"""수식(라인 단위) 제거.

LaTeX로 조판된 논문은 수식의 이탤릭 변수(𝑄, 𝑊, 𝐷, h𝑡 등)를 일반 알파벳이 아니라
유니코드 "Mathematical Alphanumeric Symbols" 블록(U+1D400–U+1D7FF) 문자로 넣는다.
pdfplumber로 그대로 추출하면 문장이 아니라 깨진 기호 나열만 남아 검색 품질을 해치므로,
이런 줄은 자연어 본문에서 제거한다. 수식 자체의 의미(등호 좌변/우변 등)를 보존하려는
시도는 하지 않고, "여기 수식이 있었다"는 사실만 버린다.
"""

import re

_MATH_ITALIC_RANGE = (0x1D400, 0x1D7FF)  # LaTeX 이탤릭 변수(𝑄, 𝑊, 𝐷, 𝑡 등)
_SUPERSCRIPT_SUBSCRIPT_RANGE = (0x2070, 0x209F)  # 위/아래 첨자
_EQUATION_NUMBER_RE = re.compile(r"\(\s*\d+[a-z]?\s*\)\s*$")
MIN_MATH_CHAR_RATIO = 0.12  # 공백 제외 문자 중 수식 문자 비율이 이 값 이상이면 수식 줄로 판정


def _is_math_char(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        _MATH_ITALIC_RANGE[0] <= codepoint <= _MATH_ITALIC_RANGE[1]
        or _SUPERSCRIPT_SUBSCRIPT_RANGE[0] <= codepoint <= _SUPERSCRIPT_SUBSCRIPT_RANGE[1]
    )


def looks_like_equation_line(line: str) -> bool:
    stripped_chars = [ch for ch in line if not ch.isspace()]
    if not stripped_chars:
        return False

    math_chars = sum(1 for ch in stripped_chars if _is_math_char(ch))
    if math_chars == 0:
        return False

    ratio = math_chars / len(stripped_chars)
    if ratio >= MIN_MATH_CHAR_RATIO:
        return True

    # 수식 번호로 끝나는 줄("... (1)")은 수식 문자가 조금만 섞여 있어도 수식으로 간주
    return bool(_EQUATION_NUMBER_RE.search(line.strip()))


def strip_equation_lines(text: str) -> tuple[str, list[str]]:
    """수식으로 보이는 줄을 제거한다. (제거된 줄은 감사용으로 함께 반환)"""
    kept_lines = []
    removed_lines = []
    for line in text.splitlines():
        if looks_like_equation_line(line):
            removed_lines.append(line)
        else:
            kept_lines.append(line)
    return "\n".join(kept_lines), removed_lines
