#!/usr/bin/env python3
"""컴파일된 LangGraph 그래프를 그림으로 내보낸다(README Architecture용).

    python scripts/export_graph.py   → outputs/architecture.mmd (+ 가능하면 outputs/architecture.png)

API 키·색인 없이 노드/엣지 구조만 컴파일한다. PNG 렌더링(draw_mermaid_png)은 mermaid.ink
원격 API를 쓰므로 네트워크가 막힌 환경에서는 실패할 수 있다. 그때는 컴파일된 그래프의 실제 노드·엣지를
Pillow(reportlab 의존성으로 함께 설치됨)로 직접 그린다. 둘 다 안 되면 Mermaid 원문(.mmd)만 남긴다.
점선 = Supervisor의 conditional edge(State 기반 라우팅), 실선 = 고정 엣지(모든 작업 노드 → Supervisor 복귀).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kv_eval.graph import build_graph  # noqa: E402


def _local_png(drawable, path: Path) -> None:
    """네트워크 없이 컴파일 그래프를 계층 배치로 그린다(노드·엣지는 drawable에서 그대로 읽는다)."""
    from PIL import Image, ImageDraw, ImageFont
    agents = ["tech", "market", "stakeholder", "domain", "synthesis", "report"]
    width, height = 1320, 600
    pos = {"__start__": (560, 60), "supervisor": (560, 230), "__end__": (900, 70),
           "quality_evaluator": (1155, 230)}
    pos.update({name: (130 + i * 205, 430) for i, name in enumerate(agents)})
    missing = set(drawable.nodes) - set(pos)
    if missing:
        raise ValueError(f"배치되지 않은 노드: {missing}")
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=17)
        small = ImageFont.load_default(size=14)
    except TypeError:  # Pillow < 10.1
        font = small = ImageFont.load_default()
    box = {name: (88 if name == "quality_evaluator" else 80, 26) for name in pos}

    def anchor(name: str, toward: tuple[int, int], shift: int) -> tuple[int, int]:
        x, y = pos[name]
        w, h = box[name]
        dx, dy = toward[0] - x, toward[1] - y
        if abs(dy) * w > abs(dx) * h:  # 위·아래 면
            return x + shift, y + (h if dy > 0 else -h)
        return x + (w if dx > 0 else -w), y + shift

    def arrow(a: tuple[int, int], b: tuple[int, int], dashed: bool, color: str) -> None:
        import math
        if dashed:
            length = math.dist(a, b)
            steps = max(int(length // 12), 1)
            for i in range(0, steps, 2):
                p = (a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)
                q = (a[0] + (b[0] - a[0]) * min(i + 1, steps) / steps, a[1] + (b[1] - a[1]) * min(i + 1, steps) / steps)
                draw.line([p, q], fill=color, width=2)
        else:
            draw.line([a, b], fill=color, width=2)
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        head = [b, (b[0] - 11 * math.cos(angle - 0.4), b[1] - 11 * math.sin(angle - 0.4)),
                (b[0] - 11 * math.cos(angle + 0.4), b[1] - 11 * math.sin(angle + 0.4))]
        draw.polygon(head, fill=color)

    for edge in drawable.edges:
        # 같은 두 노드를 잇는 왕복 엣지는 좌우로 벌려 겹치지 않게 그린다.
        shift = -14 if edge.conditional else 14
        a = anchor(edge.source, pos[edge.target], shift)
        b = anchor(edge.target, pos[edge.source], shift)
        arrow(a, b, edge.conditional, "#6842a6" if edge.conditional else "#20789d")
    for name, (x, y) in pos.items():
        w, h = box[name]
        fill = "#e8ddff" if name == "supervisor" else "#fff2cc" if name == "quality_evaluator" else \
            "#eeeeee" if name.startswith("__") else "#dff3ff"
        draw.rounded_rectangle([x - w, y - h, x + w, y + h], radius=14, fill=fill, outline="#333333", width=2)
        label = name.strip("_").upper() if name.startswith("__") else name
        draw.text((x, y), label, fill="#111111", font=font, anchor="mm")
    draw.text((20, height - 50), "dashed = supervisor add_conditional_edges (State-based routing, Send fan-out)",
              fill="#6842a6", font=small)
    draw.text((20, height - 28), "solid = fixed edge (every worker incl. report and quality_evaluator returns only to supervisor)",
              fill="#20789d", font=small)
    image.save(path)


def main(output_dir: Path = ROOT / "outputs") -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    drawable = build_graph(rag=None, web=None, llm=None).get_graph()
    mermaid = drawable.draw_mermaid()
    (output_dir / "architecture.mmd").write_text(mermaid, encoding="utf-8")
    print(f"OK: {output_dir / 'architecture.mmd'}")
    target = output_dir / "architecture.png"
    try:
        target.write_bytes(drawable.draw_mermaid_png())
        print(f"OK: {target} (mermaid.ink)")
        return 0
    except Exception as exc:  # noqa: BLE001 - 네트워크 차단 시 로컬 렌더링으로 대체
        print(f"WARN: mermaid.ink 렌더링 실패({type(exc).__name__}) → 로컬 렌더링", file=sys.stderr)
    try:
        _local_png(drawable, target)
        print(f"OK: {target} (local Pillow renderer)")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: PNG 생성 실패({type(exc).__name__}: {str(exc)[:160]}) → .mmd만 생성", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
