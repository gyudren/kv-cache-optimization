# Subject

본 프로젝트는 KV cache 최적화 기술을 소프트웨어, 하드웨어 두 진영에서 선정하여,
시장·이해관계자·도메인 관점에서 평가하는 Agentic RAG를 개발하는 프로젝트 임.

## Overview

- Objective : 하나의 기술을 복수 관점에서 비교 평가
- Method : Multi-Agent(Distributed) + Agentic RAG
- Tools : LangGraph, FAISS + BM25(RRF 융합), Tavily Web API, pdfplumber (2단 레이아웃·표 좌표 처리를 위해 pypdf 대신 채택)

## Selected Technologies

- SW : **DeepSeek-V2 MLA** (Multi-head Latent Attention, arXiv 2405.04434) — Key·Value를 저차원 잠재 벡터로 압축하도록 어텐션 구조 자체를 재설계한 기술로, 원문 보고 기준 KV cache 93.3% 감소. DeepSeek-AI가 최초 실증한 이후 후속 모델과 AWS Bedrock, NVIDIA NIM 등 상용 추론 생태계에서 지속 채택되어 공개 정보 기준 TRL 9에 근접한 실제 운용 기술로 판단해 선정
- HW : **ITME** (CXL-Hybrid 계층 메모리 확장, arXiv 2606.12556, SK hynix) — CXL 기반 DRAM-NVMe Hybrid Memory로 TB급 원격 메모리 계층을 구성하고, 모델 가중치와 prefix KV cache의 예측 가능한 접근 패턴을 활용해 데이터를 미리 이동시키는 기술로, 데이터센터·클라우드 LLM 서빙의 확장성과 처리 효율 향상이 기대되어 선정

## Features

- PDF 자료 기반 정보 추출 : RAG 적재 문서 4편(DeepSeek-V2/MLA, ITME, InfiniGen, CXL-PNM 원문 논문, 총 96p ≤ 200p 예산) 페이지 단위 파싱 및 인용 `[n, p.X]` 지원
- Dense(FAISS) + BM25 하이브리드 검색을 RRF로 융합하고, 기술별 문서 필터로 다른 기술 수치 혼입 방지
- Agentic RAG 파이프라인 : 질의 계획(한국어→영어 기술어) → 검색 → 관련성 판정 → 관련 청크 2개 미만 시 질의 재작성 후 재검색(최대 2회) → 근거 기반 답변, 근거 없으면 "근거 부족" 명시
- 확증 편향 방지 전략 : HW 베이스라인 문서(InfiniGen, CXL-PNM)로 선정 기술(ITME)의 한계를 제3의 시각에서 교차 확인, SW·HW 두 기술을 동일 형식(쟁점/관점 A 평가/관점 B 평가/이유)으로 병기, 평가 종합 에이전트는 새로운 자료를 검색하지 않고 앞 단계에서 검증된 Evidence만 사용

## Tech Stack

- Framework : LangGraph
- LLM/Generator : {GPT version}
- LLM/Judge : {GPT version}
- Retrieval : FAISS(Dense) + BM25(Sparse), RRF 순위 융합 - {Hit Rate@K}, {MRR}
- Embedding : Qwen3-Embedding-0.6B (다국어·교차언어 검색, 최대 32K 토큰, Apache 2.0 라이선스)

## Agents

- Master Agent : 전체 작업 분배 및 실행 제어. State를 확인해 다음 Agent를 선택하고 병렬 실행·결과 수집·근거 부족 재실행·종료 여부를 결정
- 기술 조사 에이전트 (RAG) : 원문에서 MLA와 ITME의 기술 개요, 적용 범위, 성능, 한계 및 TRL 판단 근거 추출
- 시장 평가 에이전트 (웹 검색) : 시장 규모, 상용화·채택 사례, 성장 전망 및 생태계 지원 현황 검색
- 이해관계자 평가 에이전트 (웹 검색) : 경쟁사 반응, 개발자 평가, 도입 기업 의견, 투자·업계 시각 조사
- 도메인 평가 에이전트 (RAG) : 데이터센터·클라우드 장문맥 서빙 환경에서 D1~D7 공통 기준으로 MLA와 ITME의 적합성 평가
- 평가 종합 에이전트 : 기술 성숙도·시장·이해관계자·도메인 평가의 일치점과 상충 지점을 근거 중심으로 종합 (신규 검색 없이 기존 Evidence만 사용)
- 보고서 생성 에이전트 : 검증된 단계별 결과와 Evidence를 연결해 SUMMARY부터 REFERENCE까지 최종 평가 보고서 생성

## Architecture

```mermaid
flowchart TD
    START([START]) --> SUP_INIT["Master Agent<br/>요청 분석 및 State 초기화"]

    SUP_INIT --> TECH["기술 조사 에이전트<br/>MLA·ITME 논문 RAG"]
    TECH --> SUP_TECH{"Master<br/>기술·TRL 근거가 충분한가?"}

    SUP_TECH -- "부족<br/>최대 2회" --> QUERY_REWRITE["Master<br/>부족 항목 지정 및 질의 재작성"]
    QUERY_REWRITE --> TECH

    SUP_TECH -- "충분" --> SUP_FANOUT["Master<br/>관점별 평가 병렬 할당"]

    subgraph PARALLEL["관점별 병렬 평가"]
        direction LR
        MARKET["시장 평가 에이전트<br/>RAG"]
        STAKEHOLDER["이해관계자 평가 에이전트<br/>외부 검색 도구"]
        DOMAIN["도메인 평가 에이전트<br/>논문 RAG"]
    end

    SUP_FANOUT --> MARKET
    SUP_FANOUT --> STAKEHOLDER
    SUP_FANOUT --> DOMAIN

    MARKET --> SUP_JOIN["Master<br/>평가 결과 수집"]
    STAKEHOLDER --> SUP_JOIN
    DOMAIN --> SUP_JOIN

    SUP_JOIN --> RESULT_GATE{"Master<br/>모든 관점 결과가 완료되었는가?"}

    RESULT_GATE -- "미완료·근거 부족" --> RETRY["Master<br/>부족한 Agent만 재할당"]
    RETRY -. "시장 근거 부족" .-> MARKET
    RETRY -. "이해관계자 근거 부족" .-> STAKEHOLDER
    RETRY -. "도메인 근거 부족" .-> DOMAIN

    RESULT_GATE -- "완료" --> SYNTHESIS["평가 종합 에이전트<br/>TRL·시장·이해관계자·도메인 종합"]

    SYNTHESIS --> SUP_SYNTHESIS{"Master<br/>일치·상충·근거 공백이 정리되었는가?"}

    SUP_SYNTHESIS -- "부족<br/>최대 1회" --> SYNTHESIS
    SUP_SYNTHESIS -- "충분" --> REPORT["보고서 생성 에이전트<br/>최종 평가 보고서 작성"]

    REPORT --> SUP_FINAL{"Master<br/>필수 목차·인용·REFERENCE 확인"}

    SUP_FINAL -- "누락 있음<br/>최대 2회" --> REPORT
    SUP_FINAL -- "완료" --> END([END])

    classDef Master fill:#e8ddff,stroke:#6842a6,stroke-width:2px,color:#1f1235;
    classDef rag fill:#dff3ff,stroke:#20789d,stroke-width:1.5px,color:#102b38;
    classDef evaluation fill:#e6f4e8,stroke:#388e3c,stroke-width:1.5px,color:#18351a;
    classDef report fill:#fff2cc,stroke:#b8860b,stroke-width:1.5px,color:#3d2f00;

    class SUP_INIT,SUP_TECH,QUERY_REWRITE,SUP_FANOUT,SUP_JOIN,RESULT_GATE,RETRY,SUP_SYNTHESIS,SUP_FINAL Master;
    class TECH,MARKET,DOMAIN rag;
    class STAKEHOLDER,SYNTHESIS evaluation;
    class REPORT report;
```

※ 설계 산출물 PDF(D-2. Graph 흐름 설계) 원본 flowchart 이미지를 Mermaid로 옮긴 것으로, 노드 문구는 원본 이미지를 기준으로 최대한 그대로 옮겼으며 세부 배치·색상은 Mermaid 렌더링 방식에 따라 원본과 다를 수 있음

## Data Preprocessing

RAG 적재 문서 4편(DeepSeek-V2/MLA, ITME, InfiniGen, CXL-PNM)을 다음 순서로 전처리하여, 임베딩/색인 단계에서 바로 쓸 수 있는 청크 목록을 만든다.

파싱(pdfplumber 좌표 기반, 2단 레이아웃 읽기 순서 재정렬) → header/footer 제거 → 표 분리(캡션·각주 묶음, 병합 셀 정규화, 다이어그램 오탐지 필터링) → 수식 줄 제거 → 참고문헌 구간(부록 포함) 제외 → 청킹(1,200자/겹침 200자, 문단 경계 보존, 페이지 경계에서 끊긴 문장 이어붙이기) → 페이지 예산(≤200p) 검증

1. `data/manifest.json`에 정의된 파일명대로 원문 PDF 4편을 `data/raw/`에 배치
2. `pip install -r requirements.txt`
3. `python -m preprocessing.pipeline` 실행 → `data/processed/chunks.jsonl`(텍스트/표 청크), `data/processed/summary.json`(문서별 통계 + 수동 확인 필요 항목) 생성
4. (선택) `python -m preprocessing.selfcheck` 로 PDF 없이 파싱·청킹 로직만 별도 검증 가능

각 청크에는 `doc_id`, `camp`(SW/HW), `role`(primary/baseline), `content_type`(text/table), `start_page`/`end_page`, `citation`(`[n, p.X]` 형식) 메타데이터가 포함되어 있어 기술별 문서 필터링과 보고서 인용에 사용할 수 있다.

**처리 세부 사항**

- **References 이후 구간(부록 포함) 제외** : References 제목이 페이지 맨 앞이 아니라 중간/끝에 나와도 탐지하고, 그 줄 이전 본문은 계속 색인 대상으로 남긴다. DeepSeek-V2(MLA)처럼 References 뒤에 부록(Appendix B/C 등 기술 부연 설명)이 이어지는 경우도 있는데, 참고문헌 구간과 함께 통째로 제외하기로 결정함 — Appendix만 따로 살리고 싶다면 `data/manifest.json`의 `reference_start_page`를 직접 지정해 경계를 조정할 수 있다.
- **수식 줄 제거** : LaTeX로 조판된 논문은 수식의 이탤릭 변수(𝑄, 𝑊, 𝐷 등)가 유니코드 Mathematical Alphanumeric Symbols 문자로 추출되어 깨진 기호 나열만 남는다. 이런 줄은 자연어가 아니므로 청크에서 제거한다(`removed_equation_line_count`로 집계). 단, 표 안에 섞여 있는 수식 기호(예: KV cache 계산식이 들어간 비교표)는 표 자체가 문장이 아니라 걸러지지 않고 표/본문 텍스트에 그대로 남을 수 있음.
- **표 오탐지 필터링** : pdfplumber의 표 탐지가 선(line)만 보고 판단하다 보니 아키텍처 다이어그램·차트도 표로 오인하는 경우가 많아, 실제 데이터 표처럼 보이는지(행/열 개수, 빈 셀 비율, 셀당 줄바꿈 수) 최소 조건으로 한 번 걸러낸다. 그래도 테두리 없는(borderless) 표는 탐지되지 않아 본문 텍스트에 섞여 들어갈 수 있음(예: MLA 논문 Table 1이 본문에 섞인 사례 확인).

자동으로 판단하기 위험한 항목은 넘겨짚지 않고 `summary.json`에 표시만 하므로, 색인 전에 아래 항목을 사람이 한 번 확인해야 한다.

- `pages_with_charts` : 페이지 면적의 15% 이상을 차지하는 이미지(차트 등)가 있는 페이지 — 텍스트 레이어와 그래프 내용이 실제로 일치하는지, 그래프 정보가 꼭 필요한지 확인 필요
- `page_boundary_review_flags` : 페이지 경계에서 문장이 끝나지 않았거나 표처럼 보이는 텍스트가 있는 지점 — 표 헤더가 다음 페이지로 이어지는데 반복되지 않은 경우 등을 확인 필요
- `table_count` / `removed_boilerplate_line_count` / `removed_equation_line_count` : 문서별 표 추출 개수, 제거된 header/footer 줄 수, 제거된 수식 줄 수 — 과다 추출·과다 제거 여부 확인 필요

## Directory Structure

├── data/                    # 문서 풀
│   ├── manifest.json        # RAG 적재 문서 4편 메타데이터(진영/역할/참고문헌 시작 페이지 등)
│   ├── raw/                 # 원문 PDF (git 미포함, 로컬에 직접 배치)
│   └── processed/           # 전처리 결과 (chunks.jsonl, summary.json)
├── preprocessing/           # 데이터 전처리 (파싱·참고문헌 제외·청킹·페이지 예산 검증)
├── agents/                  # Agent 모듈
├── prompts/                 # 프롬프트 템플릿
├── outputs/                 # 평가 결과 저장
├── app.py                   # 실행 스크립트
└── README.md

## Usage

```bash
python {app.py}
```

## Contributors

2조 : 김동욱(P280), 김민정(P282), 김태동(P287), 박규리(P289), 이재겸(P298), 임동건(P301)
