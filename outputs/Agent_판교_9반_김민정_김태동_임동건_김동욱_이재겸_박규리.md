# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-10-07

## SUMMARY
- **TRL 추정(공개 정보 기반)**: MLA와 ITME 모두 개별 기술은 **TRL 4-6**으로 판정된다. MLA는 DeepSeek-V2 통합 및 논문 벤치마크, ITME는 FPGA 프로토타입과 통합 평가가 확인된 수준이다 [1, p.1] [1, p.32] [2, p.10] [2, p.11].
- MLA는 KV 표현 자체를 압축한다. 저자 보고 기준 DeepSeek-V2는 DeepSeek 67B 대비 KV cache를 93.3% 줄이고 최대 생성 처리량을 5.76배 높였다 [1, p.1]. 다만 장문맥 서비스의 TTFT·동시성·운영 안정성은 직접 검증되지 않았다.
- ITME는 원격 CXL-hybrid memory와 프리페치로 저장 계층을 확장한다. 저자 보고 기준 NVMe-oF 기준선 대비 처리량 1.80배 향상이 제시됐으나 [2, p.2], 성능은 프리페치·버퍼·원격 접근 조건에 의존한다.
- MLA는 생태계 지원 신호가 긍정적이나 실제 MLA 고객 채택은 근거 부족이다 [W4] [W5] [W22]. ITME는 CXL 계열 생태계 신호와 개별 ITME 상용화 근거가 분리된다 [W2] [W12].
- 두 접근은 경쟁 순위의 대상이 아니라, 각각 모델 내부 압축과 시스템 메모리 확장을 담당하는 보완 가능 계층이다. 결합 성능·호환성·TCO는 근거 부족이다.

## 1. 분석 배경
KV cache는 생성 추론에서 이전 토큰의 key·value를 보존하여 attention 계산을 돕지만, 레이어 수·head 수·문맥 길이·동시 요청 및 저장 정밀도에 따라 증가한다. 설계 기준상 HBM 점유 증가는 동시 요청·batch·문맥 길이를 제약하고, 이동·로딩 지연과 재계산 부담을 유발한다 [D].

본 평가는 대규모 동시 요청, 비용 민감성, 장문맥 요구가 공존하는 데이터센터·클라우드 LLM 서빙을 대상으로 한다 [D]. MLA와 ITME는 동일 병목을 각각 모델 구조와 계층 메모리에서 다루므로, 단일 성능 수치로 우열을 정하기보다 TRL·시장성·이해관계자·도메인 적용성에서 장점과 제약이 어떻게 엇갈리는지 비교한다.

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB(≈74.5GiB) 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 약 3.4% |
| 128K | 40 GiB | 320 GiB | 약 53.7% |
| 1M | 320 GiB | 2,560 GiB | 약 429.5% |

산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. 요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). HBM 대비 비율은 80GB = 80×10⁹B ≈ 74.5GiB를 분모로 계산했다(설계서 A-2의 3%·50%·400%는 GiB/GB 단위를 혼용한 근사치라 여기서 바로잡음) [D].

## 2. 기술 선정
선정 방식은 **2안: 조 토의 후 직접 선정**이다 [D]. 평가 기준은 TRL, 시장성, 이해관계자, D1-D7 도메인 적용성이며, 결론은 기술별 신호와 도입 전 확인사항을 제시하는 데 둔다 [D].

MLA는 key·value의 저랭크 공동 압축으로 KV cache를 줄이는 attention 재설계라는 점에서 선정했다 [D]. DeepSeek-V2 논문은 MLA가 추론 시 KV-cache 병목을 줄이도록 설계됐다고 설명한다 [1, p.6] [1, p.7].

ITME는 CXL-hybrid memory, RDMA 및 프리페치로 가중치와 KV cache의 원격 확장 계층을 구성한다는 점에서 선정했다 [D] [2, p.2]. KIVI·TurboQuant 및 기타 HW 후보의 비선정 사유는 설계 검토 의견 외 별도 검증 자료가 없어 추가 비교하지 않는다.

#### 표 2. 비선정 후보와 사유 (설계 단계 팀 판단 — 설계 산출물 A-4) [D]
| 후보 | 진영 | 비선정 사유 | RAG 문서 활용 |
|---|---|---|---|
| InfiniGen | HW | 호스트 메모리 오프로딩으로 신규 메모리 인프라 없이 SW 관리 성격이 강함 | O (ITME 비교용 베이스라인) |
| CXL-PNM | HW | 연산까지 메모리 측으로 옮기는 확장형 접근으로, '공간 확장' 자체의 대표성은 ITME가 더 직접적 | O (ITME 비교용 베이스라인) |
| KIVI | SW | 사후 양자화의 대표 베이스라인이나 공개 채택 근거가 연구·라이브러리 수준 | X (선정 검토만) |
| TurboQuant | SW | 재학습 없이 적용 가능한 최신 양자화이나 공식 상용화 근거가 제한적 | X (선정 검토만) |

## 3. 기술 개요
| 항목 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 진영 | 모델/소프트웨어 아키텍처 | 시스템/하드웨어 메모리 아키텍처 |
| 작동 계층 | Transformer attention 내부 | GPU HBM·호스트 메모리·CXL-hybrid memory·SSD 간 계층 |
| 핵심 접근 | key·value를 저랭크 잠재 벡터로 공동 압축 [1, p.7] | 원격 바이트 주소형 메모리와 HW/SW 프리페치 [2, p.2] |
| 저자 보고 효과 | KV cache 93.3% 감소, 최대 생성 처리량 5.76배 [1, p.1] | NVMe-oF 대비 처리량 1.80배 [2, p.2] |
| 전제조건 | MLA 구조를 포함한 모델·서빙 실행 경로 | CXL-hybrid memory, RDMA, 프리페치 제어 |
| 한계 | 서비스 지연·동시성의 직접 측정 부족 | 원격 접근·프리페치 적중 및 버퍼 관리 의존 |

#### MLA
MLA는 압축 잠재 벡터를 캐시하고, 추론에서 key·value 복원을 행렬에 흡수할 수 있도록 설계된다 [1, p.7]. 저자 보고 비교에서 MHA 대비 토큰당 KV-cache 원소 수가 감소하고, 제시된 hard benchmark 점수는 MLA 구성이 높았다 [1, p.32]. 다만 해당 비교쌍의 활성 파라미터 수가 완전히 같지 않아 관측 차이를 MLA 단독 효과로 단정할 수 없다 [1, p.32].

#### ITME
ITME는 SSD 데이터를 내부 DRAM cache로 프리페치하고, CXL-hybrid memory에서 GPU로 필요한 가중치와 prefix KV를 선전송한다 [2, p.2] [2, p.8]. 저자 보고 성능은 특정 평가 구성의 기준선 비교다. FPGA 프로토타입은 하드웨어 실현 가능성을 검증했지만, cache miss 조건에서는 성능 저하 가능성이 보고된다 [2, p.10] [2, p.11].

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | 개별 기술 TRL: 4-6 (판정). 기준은 통합 구현 및 유사 운용 환경 시연이다. MLA는 DeepSeek-V2에 통합되어 MHA 비교 및 KV-cache·벤치마크 평가가 문서화되었으므로, 논문 근거만으로는 구현·검증 단계(TRL 4-6)가 뒷받침된다 [deepseek_v2, p.1, p.32]. 다만 제공된 허용 PDF 발췌에는 MLA 자체를 사용한 제품 출시, 상용 API/서비스 운영, 또는 외부 고객 채택의 직접 근거가 없다. 따라서 TRL 7-9는 이 자료만으로 부여하지 않는다. 계열 TRL: 4-6 (판정). attention/KV-cache 압축 아키텍처 계열에 대해 이 문서가 직접 제공하는 근거는 DeepSeek-V2 내 MLA의 구현과 내부/논문 벤치마크뿐이다 [deepseek_v2, p.6-8, p.32]. 계열 전반의 상용 운용 성숙도를 입증하는 허용 문서 근거는 없다. | MLA 자체: 논문에 DeepSeek-V2 통합, KV-cache 절감 및 MHA 대비 벤치마크 검증이 확인되어 ‘통합 구현·내부/논문 평가 확인’ 수준이다 [deepseek_v2, p.1, p.32]. 상용 서비스 운영 확인에 필요한 MLA 기반 모델의 운영 서비스 또는 MLA backend/kernel의 실제 운영 배포 근거는 허용 PDF 발췌에 없다. MLA 계열: 저랭크 KV 압축 attention의 구조와 DeepSeek-V2 적용은 확인되지만 [deepseek_v2, p.6-8], 계열 전체의 독립 재현, 외부 고객 채택 및 상용 운영은 확인되지 않는다. |
| ITME | 개별 기술 TRL: 4-6 (판정). ITME 자체는 FPGA 기반 프로토타입으로 하드웨어 실현 가능성을 검증했고, CMM 기반 구성 및 동일 조건의 Llama-3.1 8B/70B 평가를 제시한다 [itme, p.10-11]. 이는 프로토타입·통합 검증 근거이나, 제공 문서에는 ITME 개별 기술의 제품 출시·실제 고객 서비스 채택·상용 운용 근거가 없으므로 TRL 7-9는 부여하지 않는다. 계열 TRL: 4-6 (판정). 문서는 CXL 기반 메모리 확장·풀링을 현대 데이터센터의 핵심 기술로 서술하고, SK hynix CMM 및 PCIe Gen5 SSD 기반 평가를 제시한다 [itme, p.11]. 그러나 허용 발췌만으로는 CXL 메모리 제품군의 제품 출시 또는 상용 고객 운용을 검증할 수 없다. 따라서 CXL 계열의 TRL 7-9 판정은 보류한다. | ITME 자체: FPGA 프로토타입, CMM 기반 성능 특성화, Llama-3.1 8B/70B 비교 평가가 확인되어 ‘프로토타입·통합 평가 확인’ 수준이다 [itme, p.2, p.10-11]. 독립 재현, 외부 고객 채택, 제품 출시 및 실제 서비스 운용은 확인되지 않는다. CXL 계열: 논문은 CXL 기반 메모리 확장 연구 및 CMM 활용을 언급하지만 [itme, p.11], 제공 발췌에는 CXL 메모리 제품의 상용 출시 또는 고객 운용을 직접 입증하는 출처가 없다. 또한 CXL 제품군의 성숙도는 ITME 개별 FPGA 프로토타입의 성숙도를 자동으로 높이는 근거가 아니다. |

TRL은 1-9 단계의 기술 성숙도 척도이며, 본 판정은 공개 자료에만 근거한 추정이다 [W17].

#### DeepSeek-V2 MLA
개별 MLA의 **TRL 4-6** 판정 근거는 DeepSeek-V2 통합, KV-cache 및 MHA 대비 벤치마크가 문서화된 점이다. 즉, 증거 수준은 **통합 구현·내부/논문 평가 확인**이다 [1, p.1] [1, p.32]. MLA 계열도 저랭크 KV 압축의 구조와 모델 적용은 확인되지만 [1, p.6] [1, p.8], 이 자료만으로 외부 독립 재현·고객 채택·운영 규모는 판정할 수 없다.

#### ITME
개별 ITME의 **TRL 4-6** 판정은 FPGA 프로토타입, CMM 기반 성능 특성화 및 Llama-3.1 평가에 근거한다. 증거 수준은 **프로토타입·통합 평가 확인**이다 [2, p.10] [2, p.11]. CXL 메모리 계열은 ITME와 별도로 보아야 한다. 논문은 CXL 기반 메모리 확장 연구와 CMM 활용을 언급하지만 [2, p.11], CXL 제품군의 성숙도가 ITME FPGA 프로토타입의 성숙도를 자동으로 높이지는 않는다.

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 혼재 | 근거 부족 |
| 상용화·채택 현황 | 근거 부족 | 혼재 |
| 생태계 지지 | 긍정 | 긍정 |
| 종합 | 근거 부족 | 근거 부족 |
| 근거 | [W25] [W3] [W4] [W5] [W22] | [W19] [W12] [W24] [W2] |

#### DeepSeek-V2 MLA
시장 규모·성장성은 **혼재**다. 전체 LLM 시장의 2025년 83.1억 달러, 2031년 249.2억 달러 전망은 MLA 전용 시장이 아니라 간접 수요 신호다 [W25]. 상용화·채택 현황은 **근거 부족**이다. DeepSeek 모델 관측성 연동은 확인되지만 MLA 자체의 고객 도입 규모를 뜻하지 않는다 [W3]. 반면 생태계 지지는 **긍정**이다. vLLM은 MLA prefill/decode backend를, SGLang은 MLA용 최적화를 문서화했다 [W4] [W5] [W22]. 따라서 종합은 **근거 부족**이다.

#### ITME
시장 규모·성장성은 **근거 부족**이다. AI inference·RAG·KV cache 워크로드 성장 전망은 CXL memory controller IC 시장의 서술이지 ITME 시장 규모가 아니다 [W24]. 상용화·채택은 **혼재**다. SK hynix의 CMM-DDR5 제품 전시와 CXL 실제 배포 확대 신호는 확인되지만 [W12] [W2], ITME 개별 제품·고객 배포·운영 근거는 아니다. 생태계 지지는 CXL 계열에 한해 **긍정**이나, ITME 구성의 직접 호환성은 확인되지 않았다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 혼재 | 근거 부족 |
| 도입 기업·개발자 | 혼재 | 근거 부족 |
| 투자 업계 | 근거 부족 | 혼재 |
| 종합 | 혼재 | 혼재 |
| 근거 | [W14] [W13] | [W23] [W20] [W27] |

#### DeepSeek-V2 MLA
경쟁 진영과 개발자·도입 기업 신호는 **혼재**다. AMD는 MLA decoding이 병목이 될 수 있다고 설명하면서 AITER 커널의 vLLM·SGLang 통합을 제시한다 [W14]. 이는 실행 최적화 필요성과 지원 확대를 함께 시사한다. 개발자 커뮤니티에서는 특정 환경에서 SGLang의 DeepSeek 지원이 vLLM보다 빠르다는 개인 경험이 제기됐지만, 이는 독립 검증된 도입 사례가 아니다 [W13]. 투자 업계의 MLA 직접 평가는 **근거 부족**이므로 종합은 **혼재**다.

#### ITME
경쟁 진영과 도입 기업·개발자에 대한 ITME 직접 반응은 **근거 부족**이다. Marvell의 CXL 관련 주장은 CXL 제품 제공자의 계열 관점이며 ITME 독립 평가는 아니다 [W23]. 투자 업계는 일반 CXL 계열에 한해 **혼재**다. Gartner와 Greyhound Research 인용은 메모리 풀링이 적응·최적화에는 도움이 될 수 있으나 구조적 제약을 단기간에 해소하지는 못할 수 있다는 견해를 제시한다 [W27]. 이에 따라 종합은 **혼재**다.

### 4.4 도메인 적용성(D1-D7)
| 평가항목 | MLA 판정 | MLA 근거 | ITME 판정 | ITME 근거 |
|---|---|---|---|---|
| D1 워크로드 수용 능력 | 조건부 | [1, p.4] [1, p.6] | 조건부 | [2, p.9] [2, p.2] |
| D2 서비스 성능 | 근거 부족 | [1, p.6] | 조건부 | [2, p.2] [2, p.3] |
| D3 메모리 자원 효율 | 조건부 | [1, p.8] [1, p.6] | 조건부 | [2, p.2] |
| D4 품질·정보 보존 | 근거 부족 | [1, p.4] [1, p.13] | 근거 부족 | [2, p.3] |
| D5 운영 안정성 | 근거 부족 | [1, p.6] | 조건부 | [2, p.9] [2, p.2] |
| D6 도입·확장 용이성 | 조건부 | [1, p.4] [1, p.6] | 조건부 | [2, p.2] [2, p.3] |
| D7 비용 효율성 | 조건부 | [1, p.6] [1, p.8] | 근거 부족 | [2, p.2] |

#### DeepSeek-V2 MLA
D1·D3·D6·D7은 **조건부**다. MLA는 저랭크 압축으로 KV cache를 줄이고, DeepSeek-V2는 128K 문맥 지원 및 NIAH 평가 결과를 제시한다 [1, p.7] [1, p.8] [1, p.13]. 그러나 GPU 메모리 제약하의 최대 동시성, 실제 bytes/token, 기존 모델 전환 절차 및 비용 측정은 제시되지 않았다. D2·D4·D5는 **근거 부족**이다. 처리량·TTFT·지연시간, MLA 압축 전후 장문맥 품질, 다중 요청 안정성의 직접 측정이 불충분하다 [1, p.1] [1, p.6].

#### ITME
D1·D2·D3·D5·D6은 **조건부**다. ITME는 장문 prefix KV와 가중치를 원격 계층에 배치하고, 다중 턴·동시 대화 조건에서 평가했다 [2, p.2] [2, p.3] [2, p.9]. 저자 보고 기준선 대비 처리량 개선도 제시된다 [2, p.2]. 다만 D4와 D7은 **근거 부족**이다. 생성 품질·정보 보존 및 장비·전력·운영비를 포함한 TCO가 제공되지 않았다. 원격 지연과 프리페치 실패·경합의 영향도 도입 시험에서 확인해야 한다 [2, p.9] [2, p.10].

## 5. 종합 의견
| 관점 | MLA | ITME |
|---|---|---|
| TRL | 4-6: 통합 구현·논문 평가 [1, p.32] | 4-6: FPGA·통합 평가 [2, p.10] |
| 시장 | 종합 근거 부족; 생태계 긍정 [W4] [W22] | 종합 근거 부족; CXL 계열 신호 긍정 [W2] |
| 이해관계자 | 혼재 | 혼재 |
| 도메인 | 조건부; D2·D4·D5 근거 부족 | 조건부; D4·D7 근거 부족 |

관점 간 공통점은 두 기술 모두 구현·평가 근거는 있으나 공개 자료만으로 외부 고객 운영과 비용 효율을 확정할 수 없다는 점이다. MLA는 구조적 KV 축소와 서빙 프레임워크 지원이, ITME는 용량 확장과 계층 프리페치가 각각 도메인 가치의 직접 근거다 [1, p.7] [W5] [2, p.2].

| 기술 | 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|---|
| MLA | 생태계와 운영 성숙도 | vLLM·SGLang 지원은 긍정 [W4] [W22] | TRL은 4-6 | 기능 지원은 고객 운영 증거와 다름 |
| MLA | 성능 | 저자 보고 KV·처리량 개선 [1, p.1] | D2 근거 부족 | TTFT·부하 조건이 부족 |
| ITME | CXL 계열과 개별 기술 | 계열 생태계는 긍정 [W2] [W12] | ITME TRL은 4-6 | CXL 제품군 성숙도는 ITME 제품화 증거가 아님 |
| ITME | 처리량과 운영 조건 | 저자 보고 1.80배 개선 [2, p.2] | D5 조건부 | 프리페치·원격 접근·경합 조건 의존 |

MLA는 캐시해야 할 표현량을 줄이고, ITME는 남은 상태와 가중치를 둘 저장 계층을 넓힌다. 따라서 보완 가능성이 있으나 결합 실험 근거가 없어 직접 대체재 또는 결합 효과로 단정하지 않는다.

## 6. 시사점
평가는 관점별로 달라진다. 모델·서빙 개발자는 MLA의 구조적 압축과 프레임워크 지원을 볼 수 있지만, 운영자는 장문맥 부하의 지연·동시성·품질을 별도로 검증해야 한다. 인프라 관점에서는 ITME의 용량 확장 가능성과 함께 CXL·RDMA·프리페치 운영 복잡성 및 TCO 미측정을 고려해야 한다.

| 이해관계자 | MLA 도입 전 확인 | ITME 도입 전 확인 |
|---|---|---|
| 클라우드 사업자 | 목표 문맥·동시성에서 TTFT, tail latency, HBM 점유 | CPU·BIOS·펌웨어·스위치 상호운용성, 실제 워크로드 성능 [W20] |
| 모델 개발사 | MLA 구조 적용 가능성, 재학습·변환 범위, 장문맥 품질 | prefix KV·가중치 접근 패턴의 예측 가능성 |
| 메모리·HW 벤더 | MLA decode 커널·가속기 호환성 [W14] | CXL-hybrid memory·RDMA·프리페치 API 및 cache miss 동작 |
| 투자자 | MLA 기능과 DeepSeek 모델 채택을 구분 | CXL 계열 확대와 ITME 개별 제품화·고객 채택을 구분 |

저자 보고 성능값은 도입 근거의 출발점일 뿐이다. MLA는 동일 모델·동일 부하에서, ITME는 동일 메모리 예산·네트워크 조건에서 서비스 수준과 비용을 분리 측정할 필요가 있다.

## 7. 한계점
**자료 기준 시점**: 웹 자료 검색·검증일 2026-10-07 (Tavily 검색 결과 기준). 이후 공개된 발표·제품 정보는 반영되지 않았다.

모든 TRL 판정은 논문, 특허성 자료, 상용 발표를 포함한 **공개 정보만으로 한 추정**이다. TRL 4-6은 통합 구현 또는 프로토타입 평가를 뜻하며, 실제 제품화·고객 운영·운영 규모와는 시차가 있을 수 있다 [W17]. 특히 TRL 4-6 구간에서는 내부 검증, 장애율, 고객 환경의 호환성, 운영 비용이 비공개일 수 있다.

확증편향을 줄이기 위해 ITME는 HW 베이스라인 논문(CXL-PNM)과 교차 확인했고 [4, p.4] [4, p.10], 두 기술을 동일한 쟁점·관점·이유 형식으로 병기했으며, 종합은 이미 검증된 Evidence에 한정했다. 그럼에도 저자 보고 성능의 독립 재현은 확인되지 않았다.

추가 확인이 필요한 항목은 다음과 같다.
- **TRL·시장**: MLA 및 ITME의 외부 고객 채택, 운영 기간·규모, ITME 개별 제품 상태.
- **도메인**: MLA의 TTFT·지연시간·최대 동시성·실측 메모리, ITME의 tail latency·cache miss·경합·장애 복구·TCO.
- **품질**: MLA 압축 전후 장문맥 정보 보존과 ITME 계층 이동의 생성 품질 영향.

CXL 일반의 제품·생태계 신호는 ITME 개별 기술의 상용 운용 증거가 아니며, DeepSeek 모델 또는 프레임워크 지원도 MLA 기능 자체의 고객 채택 증거로 일반화하지 않았다.

#### 표 7. 증거 균형표 (중복 제거 후 수집·검증된 Evidence 수)
| 구분 | DeepSeek-V2 MLA | ITME | HW 베이스라인(InfiniGen·CXL-PNM) |
|---|---|---|---|
| 기술 조사 · 논문 청크 | 15 | 23 | 4 |
| 기술 조사 · 웹 문서 | 12 | 4 | 0 |
| 시장 · 웹 문서 | 5 | 4 | 0 |
| 이해관계자 · 웹 문서 | 2 | 3 | 0 |
| 도메인 · 논문 청크 | 15 | 23 | 0 |
| 합계 | 49 | 57 | 4 |

## REFERENCE
- [1] DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, p.1, p.4, p.6, p.7, p.8, p.13, p.32.
- [2] Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J. (SK hynix) (2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv preprint arXiv:2606.12556, p.2, p.3, p.8, p.9, p.10, p.11.
- [4] Kim, D., Lee, M., Kim, J., Kwon, H., Jeong, H., Park, S.-S., et al. (Hanyang University, Samsung Electronics) (2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits. arXiv preprint arXiv:2511.00321, p.4, p.10.
- [W4] docs.vllm.ai (게시일 미확인). mla_attention - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention
- [W5] docs.vllm.ai (게시일 미확인). Attention Backend Feature Support - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/stable/design/attention_backends
- [W22] lmsys.org (게시일 미확인). SGLang v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision - LMSYS Org. lmsys.org, https://www.lmsys.org/blog/2024-09-04-sglang-v0-3
- [W2] computeexpresslink.org (게시일 미확인). Breaking Boundaries in Memory: Highlights from AI Infra Summit and SDC 2025 - Compute Express Link. computeexpresslink.org, https://computeexpresslink.org/blog/breaking-boundaries-in-memory-highlights-from-ai-infra-summit-and-sdc-2025-4198
- [W12] news.skhynix.com (Thu, 01 Oct 2026 23:00:00 GMT). SK hynix Presents a Full Lineup of Memory Solutions Optimized for AI Infrastructure at DTF 2026 – SK hynix Newsroom. news.skhynix.com, https://news.skhynix.com/en/dtf-2026
- [W17] esa.int (게시일 미확인). ESA - Technology Readiness Levels (TRL). esa.int, https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/Technology_Readiness_Levels_TRL
- [W25] mordorintelligence.com (Fri, 11 Sep 2026 00:00:00 GMT). Large Language Model Market Size, Growth & Outlook | Industry Report 2031. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/large-language-model-llm-market
- [W3] docs.dynatrace.com (Fri, 28 Aug 2026 00:00:00 GMT). AI Observability integrations — Dynatrace Docs. docs.dynatrace.com, https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/integrations
- [W19] forbes.com (Tue, 25 Aug 2026 00:00:00 GMT). CXL Growth Shown At The 2026 FMS Conference - Forbes. forbes.com, https://www.forbes.com/sites/tomcoughlin/2026/08/25/cxl-growth-shown-at-the-2026-fms-conference/
- [W24] mordorintelligence.com (Tue, 28 Jul 2026 00:00:00 GMT). CXL Memory Controller IC Market Size, Share & 2031 Growth Trends Report. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/cxl-memory-controller-ic-market
- [W14] rocm.docs.amd.com (게시일 미확인). MLA decoding kernel of the AITER library to accelerate LLM inference — Tutorials for AI developers 5.1. rocm.docs.amd.com, https://rocm.docs.amd.com/projects/ai-developer-hub/en/v5.1/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
- [W13] news.ycombinator.com (게시일 미확인). DeepSeek Open Source FlashMLA – MLA Decoding Kernel for Hopper GPUs | Hacker News. news.ycombinator.com, https://news.ycombinator.com/item?id=43155023
- [W23] marvell.com (게시일 미확인). Marvell Structera™ X, A and S: A Comprehensive CXL Portfolio Powering AI Memory Innovation. marvell.com, https://www.marvell.com/blogs/marvell-structera-cxl-portfolio.html
- [W20] hpcwire.com (게시일 미확인). What Hyperscalers Should Know About CXL. hpcwire.com, https://www.hpcwire.com/2026/08/20/what-hyperscalers-should-know-about-cxl
- [W27] networkworld.com (게시일 미확인). Chip wafer shortage will run through 2030 as AI demand overwhelms supply: SK Hynix chief | Network World. networkworld.com, https://www.networkworld.com/article/4146270/chip-wafer-shortage-will-run-through-2030-as-ai-demand-overwhelms-supply-sk-hynix-chief.html
- [D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 (A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.
