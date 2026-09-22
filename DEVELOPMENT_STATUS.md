# 개발 현황

기준 문서: `RAG-Design_판교_9반_....pdf`(설계 산출물)과 `README.md`.
설계서와 구현이 다른 지점은 숨기지 않고 아래와 표(README의 "설계 산출물 기재값과 실측값의 차이")에 남긴다.

## 구현 완료

- **파이프라인 실행 진입점** : `python app.py` — 전처리 산출물 적재 → FAISS/BM25 색인 → LangGraph 실행 → 보고서·검증·로그를 `outputs/`에 저장.
- **설계 D-2 그래프** : Master 게이트 5종(기술/결과/종합/보고서 + 질의 재작성·재할당)과 관점별 병렬 평가(Send 팬아웃)를 mermaid와 1:1로 구현. 재시도 한도는 설계 D-1 그대로(tech 2 / 관점 각 2 / synthesis 1 / report 2).
- **설계 D-1 State** : 14개 키 전부 구현, `evidence`·`logs`는 reducer로 누적.
- **RAG** : 전처리 청크(`data/processed/chunks.jsonl`) → Qwen3-Embedding-0.6B → FAISS(Dense) + BM25(Sparse) → RRF 융합, 기술별 문서 필터로 다른 기술 수치 혼입 차단. 질의는 Qwen3의 query instruction을 적용해 임베딩한다.
- **Agentic RAG** : 질의 계획(한국어→영어) → 검색 → 관련성 판정 → 관련 청크 2개 미만이면 재작성 후 재검색(최대 2회) → 인용 포함 답변, 근거 없으면 "근거 부족".
- **웹 검색(Tavily)** : 시장·이해관계자 평가(설계 B-2에 따라 RAG 미사용)와 TRL 7~9 판정 근거(설계 C-1의 상용화·통합 발표) 수집.
- **보고서** : 필수 목차(SUMMARY~7, REFERENCE) 순서·중복 검증, 본문 인용과 근거 카탈로그 대조, SUMMARY 1/2페이지 제한 검사, Markdown + 한글 PDF 내보내기.
- **테스트** : 54개 통과. 네트워크·API 키 없이 실행된다(`pytest -q`).
- **검색 품질 평가** : `python -m eval.evaluate_retrieval` — 운영 검색기를 그대로 평가하며 합격 기준 충족 여부를 판정한다.

## 확인이 필요한 항목

- **설계서 수치와의 차이** : 참고문헌 제외 14p(설계 11p), 색인 84p(설계 85p), 청크 359개(설계 333개). 근거는 README에 표로 정리했고, 실행 시 `outputs/corpus_stats.json`의 `warnings`로도 출력된다.
- **전처리 수동 확인 항목** : `data/processed/summary.json`의 `pages_with_charts`, `pages_with_vector_diagrams`, `page_boundary_review_flags`는 자동 판단이 위험해 표시만 해둔 항목이라 사람이 확인해야 한다.
- **색인 단계 수식 정제** : `CLEAN_FORMULA_NOISE` 환경변수로 켜고 끌 수 있으며, 어느 쪽이 나은지는 동일 평가 세트로 측정해 결정한다.

## 재현 절차

```bash
pip install -e ".[dev]"
cp .env.example .env               # OPENAI_API_KEY, TAVILY_API_KEY 입력
# data/manifest.json의 filename대로 원문 PDF 4편을 data/raw/에 배치
python -m preprocessing.pipeline
pytest -q
python app.py
```

프로젝트 루트에서 실행한다. 모델 대체 경로는 두지 않는다(설정된 모델과 다르면 실행을 거부).
`.env`와 실제 키는 절대 커밋하지 않는다.
