# Backend development status — scaffold / local checks

Baseline: `Python_Backend_Designer_Final_Implementation_Spec.md` v1.0 and final RAG-Design PDF.
The original `README.md` is preserved verbatim (source: `README(1).md`).

## Implemented in the scaffold

- `app.py` source-checkout entry point and fixed configuration for MLA vs ITME and GPT-5.6 Sol.
- LangGraph State/Node/Edge code: Master, technical research, market (Tavily only), stakeholder, domain, synthesis, reporting; selective retry limits.
- Four-paper manifest, physical PDF-page checks, page-preserving parsing, 1200/200 chunking, Qwen3 local embedding, FAISS/BM25/RRF with paper filters and RAG rewriting.
- Evidence, citation, report-section checks; Markdown and ReportLab PDF export with Korean-font preflight.
- Pure/local unit tests; no placeholder or fabricated evaluation results.

## Not yet validated; do not claim completion

- Actual four research PDF files have **not** been supplied or indexed. Design-stated page/chunk counts are not actual measurements.
- No OpenAI/Tavily credentials or network-backed run has been supplied; GPT model access, real search, and generated final report are unverified.
- The local environment has no `langgraph`, so graph compilation/integration test is skipped until dependencies are installed.
- The six exact research questions and one baseline question remain implementation drafts pending PM approval under the final specification.
- The final report PDF has not been rendered or visually checked. The presence of an exporter is not proof of a completed deliverable.

## Local reproducibility

```bash
python -m pip install -e '.[dev]'
cp .env.example .env
# Set OPENAI_API_KEY, TAVILY_API_KEY; add the real research PDFs under data/papers/.
pytest -q
python app.py
```

Run from the project root. No alternative model fallback is configured. Never commit `.env` or real secrets.
