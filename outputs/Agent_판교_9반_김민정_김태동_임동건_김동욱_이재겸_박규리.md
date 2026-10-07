# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-10-07

## SUMMARY
- **TRL 추정(공개 정보 기반):** DeepSeek-V2 MLA와 ITME 개별 기술은 모두 **TRL 4-6**이다. MLA는 DeepSeek-V2 내 구현·내부 평가, ITME는 FPGA 프로토타입·제한 워크로드 평가가 근거이나, 실서비스 운영·독립 재현·상용 채택은 확인되지 않는다 [1, p.6] [1, p.15] [2, p.2] [2, p.11].
- **MLA:** 모델 아키텍처에서 KV를 저랭크 잠재 벡터로 압축한다. 메모리 효율(D3)은 적합이나, 품질 보존(D4)과 운영 안정성(D5)은 근거 부족이다 [1, p.7] [1, p.8].
- **ITME:** CXL-hybrid 계층과 프리페칭으로 장문맥 KV를 배치·이동한다. 서비스 성능(D2)·메모리 효율(D3)은 적합 신호이나, 원격 계층 지연·통합·운영 검증이 제약이다 [2, p.2] [2, p.7] [2, p.9].
- 시장·이해관계자 평가는 엇갈린다. MLA는 생태계 지원이 긍정이나 채택은 근거 부족이고, ITME는 CXL 계열 성장 신호가 있으나 ITME 개별 상용화는 근거 부족이다 [W5] [W6] [W21] [W1].
- 두 기술은 경쟁적 단일 대안이 아니라, KV 저장량을 줄이는 모델 계층 접근과 남은 대용량 상태를 확장 계층에 배치하는 인프라 접근으로서 보완 가능성이 있다. 다만 결합 효과의 직접 근거는 없다.

## 1. 분석 배경
KV cache는 생성 추론에서 과거 토큰의 key·value를 보관해 어텐션 계산을 돕는 상태이며, 일반 MHA에서는 이 저장량이 최대 배치와 시퀀스 길이를 제한하는 병목으로 설명된다 [1, p.7]. 데이터센터·클라우드 장문맥 서빙에서는 대규모 동시 요청, 비용 민감성, 문맥 길이 민감성이 겹쳐 HBM 점유와 KV 이동·재계산 문제가 커진다는 것을 설계 기준으로 삼았다 [D].

본 평가는 하나의 우승 기술을 고르지 않는다. MLA는 저장해야 할 KV 자체를 줄이고, ITME는 대용량 데이터를 계층적으로 배치·이동하므로 작동 계층과 측정 단위가 다르다. 따라서 기술 성숙도, 시장성, 이해관계자, 도메인 적용성에서의 근거와 공백을 병기한다.

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB(≈74.5GiB) 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 약 3.4% |
| 128K | 40 GiB | 320 GiB | 약 53.7% |
| 1M | 320 GiB | 2,560 GiB | 약 429.5% |

산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. 요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). HBM 대비 비율은 80GB = 80×10⁹B ≈ 74.5GiB를 분모로 계산했다(설계서 A-2의 3%·50%·400%는 GiB/GB 단위를 혼용한 근사치라 여기서 바로잡음) [D].

## 2. 기술 선정
#### 선정 방식
- **2안: 조 토의 후 직접 선정**을 적용했다. 선정 기준은 KV cache 병목과의 직접성, 데이터센터·클라우드 장문맥 서빙 관련성, 그리고 소프트웨어·하드웨어의 서로 다른 대응 계층을 함께 검토할 수 있는지였다 [D].

#### 선정 결과
- **DeepSeek-V2 MLA:** key·value를 저차원 잠재 벡터로 공동 압축하도록 어텐션 구조를 재설계한 소프트웨어 접근이다 [1, p.6] [1, p.7]. 기존 비-MLA 모델에 사후 적용하는 경로는 공개 근거로 확인되지 않아 도입성은 조건부로 둔다 [D].
- **ITME:** CXL 기반 DRAM-NVMe hybrid memory와 프리페칭으로 장문맥 KV 및 모델 가중치를 확장 계층에 배치하는 하드웨어 접근이다 [2, p.1] [2, p.2].

비선정 후보의 비교·평가는 검증된 입력이 있을 때에만 다룬다. 본 보고서에서는 InfiniGen과 CXL-PNM을 ITME 한계의 교차 확인용 베이스라인으로만 사용한다 [3, p.4] [4, p.4].

#### 표 2. 비선정 후보와 사유 (설계 단계 팀 판단 — 설계 산출물 A-4) [D]
| 후보 | 진영 | 비선정 사유 | RAG 문서 활용 |
|---|---|---|---|
| InfiniGen | HW | 호스트 메모리 오프로딩으로 신규 메모리 인프라 없이 SW 관리 성격이 강함 | O (ITME 비교용 베이스라인) |
| CXL-PNM | HW | 연산까지 메모리 측으로 옮기는 확장형 접근으로, '공간 확장' 자체의 대표성은 ITME가 더 직접적 | O (ITME 비교용 베이스라인) |
| KIVI | SW | 사후 양자화의 대표 베이스라인이나 공개 채택 근거가 연구·라이브러리 수준 | X (선정 검토만) |
| TurboQuant | SW | 재학습 없이 적용 가능한 최신 양자화이나 공식 상용화 근거가 제한적 | X (선정 검토만) |

## 3. 기술 개요
| 구분 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 진영 | 모델·어텐션 소프트웨어 | CXL-hybrid 메모리·프리페칭 하드웨어 |
| 작동 계층 | 모델 아키텍처 | 메모리·스토리지 계층 |
| 핵심 접근 | KV 저랭크 공동 압축 및 잠재 벡터 캐시 [1, p.7] | CXL-hybrid memory, DMA 프리페칭, 계층 배치 [2, p.1] [2, p.2] |
| 저자 보고 효과 | KV cache 감소 및 효율적 추론 목표 [1, p.6] | NVMe-oF 기준선 대비 처리량 **1.80배** 향상 보고 [2, p.2] |
| 전제조건 | MLA 구조를 포함한 모델·서빙 구현 | CXL-hybrid memory, RDMA, 호스트 스테이징, 프리페칭 통합 [2, p.2] |
| 한계 | MLA 단독 장문맥 품질·운영 안정성 근거 부족 | 원격 계층 지연, 경합, 상용 운영 근거 부족 [2, p.7] [2, p.11] |

#### MLA
MLA는 추론 중 완전한 key·value 대신 압축 잠재 벡터를 캐시하고, 일부 투영을 다른 행렬에 흡수해 명시적 key·value 계산을 줄이는 구조다 [1, p.7]. 저자 보고 결과는 DeepSeek-V2 전체 구성의 결과이므로 MLA 단독 효과로 귀속할 수 없다.

#### ITME
ITME는 압축 기술이 아니라 대용량·예측 가능한 모델 가중치와 장문맥 KV를 CXL-hybrid 확장 계층에 두고 데이터 이동을 연산과 중첩하는 기술이다 [2, p.2] [2, p.4]. 저자 보고 기준 ITME는 재계산 기준 대비 5턴에서 **1.81배** 속도 향상과 NVMe-oF 기준선 대비 처리량 **1.80배** 향상을 보고했다 [2, p.9] [2, p.2].

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | 개별 기술 TRL: 4-6. 판정: 제공 논문은 DeepSeek-V2 내 MLA 구현, 내부 벤치마크 및 저자 보고 성능을 보여주므로 통합 구현·검증 단계로는 평가할 수 있다. 그러나 논문만으로 실제 서비스 운영 또는 상용 채택은 확인되지 않아 TRL 7-9는 부여하지 않는다. [1, p.6] [1, p.7] [1, p.8] [1, p.15] [1, p.21] 계열 TRL: 4-6. 판정: 제공 자료는 MLA가 속한 attention/KV-cache 최적화 계열의 별도 상용 제품군 또는 실제 서비스 채택을 직접 문서화하지 않는다. 따라서 MLA 구현·검증 근거를 넘어서 계열을 TRL 7-9로 판정할 근거가 없다. 이 값은 MLA 자체와 일반 계열의 상용 성숙도를 동일시하지 않는 보수적 상한이다. | TRL 기준 적용: 논문 기반 근거는 구현·검증을 지지할 수 있으나 실운용 채택을 지지하지 않는다. MLA의 저자 보고 결과는 DeepSeek-V2 전체 구성의 결과이며 MLA 단독의 외부 검증 또는 운영 채택 증거가 아니다. 공식 GitHub 발췌문은 FlashMLA가 DeepSeek-V3/V3.2-Exp 모델을 지원한다고 설명하지만, 제공 범위에서 이것이 DeepSeek-V2 MLA 원 아키텍처의 상용 서비스 채택을 직접 입증하지 않으므로 TRL 7-9 판정 근거로 사용하지 않았다. |
| ITME | 개별 기술 TRL: 4-6. 판정: ITME 자체는 hardware prototype 및 ITME(FPGA) 비교 평가, Llama-3.1 기반 워크로드 검증이 문서화되어 있어 프로토타입/유사 운용 조건 검증 단계로 평가할 수 있다. 실서비스 배포·상용 운용 근거가 없으므로 TRL 7-9는 부여하지 않는다. [2, p.2] [2, p.9] [2, p.11] 계열 TRL: 4-6. 판정: 제공 자료는 CXL 기반 메모리 확장 및 pooling을 핵심 기술로 서술하고 CXL-hybrid memory와 PCIe Gen5 SSD/CMM을 사용한 feasibility를 제시하지만, CXL 메모리 모듈 일반의 제품 출시나 상용 운영 사례는 문서화하지 않는다. 따라서 CXL 계열의 일반적 상용 성숙도를 ITME의 FPGA 프로토타입 근거만으로 TRL 7-9로 판정할 수 없다. [2, p.4] [2, p.11] | TRL 기준 적용: FPGA hardware prototype과 제한된 모델·데이터셋 기반 평가는 TRL 4-6의 구현/검증 근거이다. CXL 일반 관련 웹 발췌문은 ITME 개별 아키텍처의 출시 또는 상용 서비스 채택을 문서화하지 않으며, 일부는 ITME/CXL과 직접 관련 없는 메모리 제품 기사이므로 TRL 근거로 사용하지 않았다. |

#### DeepSeek-V2 MLA
개별 기술은 공개 논문에서 DeepSeek-V2 내 MLA 구현과 내부 평가가 확인되어 TRL 4-6으로 추정한다 [1, p.6] [1, p.7] [1, p.15]. 그러나 이는 DeepSeek-V2 전체 구성의 저자 보고이며, 실제 서비스 운영 또는 독립 검증은 확인되지 않는다. MLA가 속한 attention/KV-cache 최적화 계열도 별도 상용 제품군·운영 채택이 문서화되지 않아 동일하게 보수적으로 TRL 4-6으로 둔다. FlashMLA의 DeepSeek-V3/V3.2-Exp 지원은 DeepSeek-V2 MLA의 상용 운영 증거가 아니다 [W9].

#### ITME
ITME 개별 기술은 FPGA 프로토타입과 Llama-3.1 기반 평가가 문서화되어 TRL 4-6으로 추정한다 [2, p.2] [2, p.9] [2, p.11]. CXL 메모리 확장 계열 역시 논문상 feasibility 근거는 있으나, 제공 자료만으로 CXL 일반의 상용 운영 성숙도를 판정할 수 없어 TRL 4-6으로 둔다 [2, p.4] [2, p.11].

두 판단은 모두 **공개 정보 기반 추정**이며, CXL 제품군 일반의 성숙도를 ITME 개별 기술의 성숙도로 전가하지 않았다.

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 근거 부족 | 긍정 |
| 상용화·채택 현황 | 근거 부족 | 근거 부족 |
| 생태계 지지 | 긍정 | 근거 부족 |
| 종합 | 근거 부족 | 근거 부족 |
| 근거 | [W17] [W22] [W19] [W5] [W6] | [W21] [W1] [W25] |

#### DeepSeek-V2 MLA
MLA 기능 자체는 vLLM의 MLA backend 및 DeepSeek-style MLA 문서, SGLang의 MLA 최적화 발표으로 생태계 지지 **긍정** 신호가 있다 [W5] [W6] [W19]. 반면 제공 시장자료는 기업용 또는 일반 LLM 시장이지 MLA가 속한 추론 최적화 시장을 직접 정의하지 않으며, DeepSeek-V2 MLA 자체의 유료 서비스·독립 고객 운영 사례도 근거 부족이다 [W17] [W22]. 따라서 종합은 근거 부족이다.

#### ITME
CXL memory controller IC 시장과 AI 추론·KV cache 수요의 성장 전망은 ITME 상위 계열의 **긍정** 신호다 [W21]. CXL pooling·sharing 활용 논의도 확인된다 [W1] [W25]. 그러나 이 근거는 ITME 개별 제품, 고객 채택, 공식 통합 또는 매출을 직접 보여주지 않는다. CXL 시장과 소프트웨어 추론 최적화 시장은 서로 달라 시장 규모만으로 비교할 수 없으며, ITME 종합은 근거 부족이다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 근거 부족 | 근거 부족 |
| 도입 기업·개발자 | 혼재 | 근거 부족 |
| 투자 업계 | 근거 부족 | 우려 |
| 종합 | 근거 부족 | 혼재 |
| 근거 | [W13] [W11] | [W14] [W23] |

#### DeepSeek-V2 MLA
경쟁 진영과 투자 업계의 직접 발언은 근거 부족이다. 개발자·도입 기업 관점은 **혼재**다. 한 커뮤니티 사용자는 두 p5 H100 노드에서 SGLang의 DeepSeek 지원이 당시 vLLM보다 더 좋고 빨랐다고 경험을 제시했으나, 재현 조건이 완전하지 않은 커뮤니티 의견이다 [W13]. 독립 기술 작성자는 weight absorption과 decoupled RoPE가 수학·특수 최적화 코드의 복잡성을 높인다고 비판했다 [W11]. 실제 도입기업의 운영 경험은 확인되지 않았다.

#### ITME
경쟁 진영과 실제 도입기업·개발자의 ITME 직접 발언은 근거 부족이다. 투자 업계 신호는 **우려**다. Gartner 및 Greyhound Research 관련 발언은 CXL memory pooling이 단기 구조적 메모리 제약의 해소책은 아니며 대규모 전환에는 시간이 걸린다고 평가했다 [W23]. 반면 삼성의 자사 평가는 CXL memory pool이 더 큰 KV cache footprint를 수용하면서 성능을 유지했다고 제시한다 [W14]. 양측은 ITME 자체가 아니라 CXL 계열에 관한 견해이므로 ITME 종합은 혼재다.

### 4.4 도메인 적용성(D1-D7)
| 평가항목 | MLA 판정 | MLA 근거 | ITME 판정 | ITME 근거 |
|---|---|---|---|---|
| D1 워크로드 수용 능력 | 조건부 | [1, p.1] [1, p.21] | 조건부 | [2, p.9] |
| D2 서비스 성능 | 조건부 | [1, p.16] | 적합 | [2, p.2] [2, p.9] |
| D3 메모리 자원 효율 | 적합 | [1, p.1] [1, p.16] | 적합 | [2, p.9] |
| D4 품질·정보 보존 | 근거 부족 | [1, p.1] [1, p.21] | 근거 부족 | [2, p.9] |
| D5 운영 안정성 | 근거 부족 | [1, p.5] [1, p.6] | 조건부 | [2, p.7] [2, p.9] |
| D6 도입·확장 용이성 | 조건부 | [1, p.7] [1, p.12] [1, p.28] | 조건부 | [2, p.1] [2, p.2] [2, p.9] |
| D7 비용 효율성 | 조건부 | [1, p.1] [1, p.16] | 조건부 | [2, p.1] [2, p.9] |

#### DeepSeek-V2 MLA
D3은 적합이다. MLA는 KV를 잠재 벡터로 압축하며, 저자 보고 기준 DeepSeek-V2 전체는 DeepSeek 67B 대비 KV cache를 **93.3%** 줄였다고 제시한다 [1, p.7] [1, p.1]. D1·D2·D6·D7은 조건부다. 모델은 128K 문맥을 지원하고, 저자들은 FP8 변환·KV 평균 6비트 양자화 조건에서 8 H800 노드의 생성 처리량이 50K tokens/s를 넘었다고 보고했지만, MLA 단독의 동시성·TTFT·총비용은 확인되지 않는다 [1, p.4] [1, p.16]. D4·D5는 장문맥 품질 보존과 다중 요청 안정성의 직접 근거가 부족하다.

#### ITME
D2·D3은 적합이다. 저자 보고 기준 ITME는 5턴에서 재계산 대비 **1.81배** 속도 향상과 NVMe-oF 대비 **1.80배** 처리량 향상을 제시하며 [2, p.9] [2, p.2], ITME 구성은 Host/CXL staging으로 10GB를 오프로딩해 안정적 16GB GPU footprint를 유지한다고 보고한다 [2, p.9]. D1·D5·D6·D7은 조건부이며 CXL·RDMA·프리페칭 통합과 원격 지연·경합을 검증해야 한다 [2, p.2] [2, p.7] [2, p.10]. D4는 생성 품질·장문맥 정보 보존 측정이 없어 근거 부족이다.

## 5. 종합 의견
#### 관점 매트릭스
| 관점 | MLA | ITME |
|---|---|---|
| TRL | 4-6: 구현·내부 평가, 운영 채택 미확인 | 4-6: FPGA 검증, 상용 운영 미확인 |
| 시장성 | 근거 부족: 생태계 지지 긍정, 채택 미확인 | 근거 부족: CXL 성장 신호, ITME 채택 미확인 |
| 이해관계자 | 근거 부족: 개발자 반응 혼재 | 혼재: CXL 기대와 전환 우려 병존 |
| 도메인 | D3 적합, D1·D2·D6·D7 조건부, D4·D5 근거 부족 | D2·D3 적합, D1·D5·D6·D7 조건부, D4 근거 부족 |

#### 관점 간 일치
- 두 기술 모두 구현·실험 근거는 있으나 실운용·독립 재현·총비용 근거가 부족하다는 점에서 TRL과 도메인 평가가 일치한다 [1, p.15] [2, p.11].
- 두 기술 모두 품질·정보 보존(D4)의 직접 검증은 근거 부족이다.

#### 상충표: MLA
| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| 생태계와 성숙도 | 시장: 생태계 지지 긍정 [W5] [W6] | TRL: 4-6 | 프레임워크 지원은 DeepSeek-V2 MLA의 실운영 증거가 아님 |
| 메모리와 운영 | D3: 적합 [1, p.7] | D5: 근거 부족 | KV 축소가 동시 요청 안정성을 직접 입증하지 않음 |

#### 상충표: ITME
| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| 계열 성장과 개별 성숙도 | 시장: CXL 성장 신호 긍정 [W21] | TRL: 4-6 | 계열 시장 전망은 ITME 제품화 증거가 아님 |
| 성능과 운영 | D2·D3: 적합 [2, p.2] [2, p.9] | D5·D6: 조건부 | 제한 워크로드 성능이 경합·통합 위험을 포괄하지 않음 |

MLA는 KV 저장량을 줄이고 ITME는 남은 상태를 계층에 배치하므로 보완 계층이다. 직접 결합 실험은 근거 부족이다.

## 6. 시사점
| 이해관계자 | 도입 전 확인 과제 |
|---|---|
| 클라우드 사업자 | 동일 장문맥·동시성 조건에서 HBM·호스트·CXL 사용량, TTFT와 토큰 지연 p50/p95/p99, 캐시 미스·경합·장애 복구를 측정 |
| 모델 개발사 | MLA 단독의 장문맥 품질·회상, 비-MLA 체크포인트 전환 또는 재학습 필요성, 서빙 커널 호환성을 검증 |
| 메모리·HW 벤더 | CXL-hybrid memory, RDMA, 스테이징·프리페처 API의 통합 범위와 원격 지연 은닉 조건을 확인 |
| 투자자 | CXL 계열 성장 전망과 ITME 개별 제품화·고객 도입을 분리하고, MLA 생태계 지원과 실제 고객 채택을 분리해 점검 |

관점별 평가는 달라진다. MLA는 모델 변경을 전제로 메모리 점유를 줄이는 데 직접적이지만 일반적 사후 도입 경로가 불명확하다. ITME는 모델 변경 없이 확장 계층을 제공할 수 있으나 하드웨어·네트워크·프리페칭 통합에 의존한다 [1, p.7] [2, p.2]. 두 접근을 함께 평가하려면 동일 워크로드에서 KV bytes/token, 계층별 점유, 품질, 지연, 총비용을 통일된 기준선으로 측정해야 한다. 공개 근거만으로 결합 효과나 단일 도입안을 제시할 수 없다.

## 7. 한계점
**자료 기준 시점**: 웹 자료 검색·검증일 2026-10-07 (Tavily 검색 결과 기준). 이후 공개된 발표·제품 정보는 반영되지 않았다.

- 모든 TRL 판정은 논문, 프로토타입·특허성 자료, 상용 발표를 포함한 **공개 정보만으로 한 추정**이다. 발표·논문과 실제 채택 사이에는 시차가 있으며, TRL 4-6 구간의 비공개 통합·장기 운영 정보 공백이 남는다.
- 근거 부족: MLA·ITME 모두 독립 재현, 실제 데이터센터 운영, 고객 채택, 장기 안정성, p95/p99 지연, 총소유비용 자료가 부족하다. MLA는 추론 최적화 시장의 직접 정의와 MLA 단독 품질·전환 경로가 부족하다. ITME는 개별 제품 출시·채택, 직접 생태계, 품질 보존 및 CXL/RDMA 경합 자료가 부족하다.
- 도메인 D1의 제한 GPU 메모리별 최대 동시 요청 수와 워크로드 정의도 근거 부족이다. ITME의 저자 보고 **1.81배** 속도 향상은 재계산 기준·5턴 조건이며 [2, p.9], **1.80배** 처리량 향상은 NVMe-oF 기반 분리형 스토리지 기준선 조건이다 [2, p.2]. 일반 서비스 성과로 확대 해석하지 않았다.
- 확증편향 방지 조치로 InfiniGen·CXL-PNM 베이스라인 논문을 사용해 ITME를 교차 확인했고 [3, p.4] [4, p.4], 두 기술을 동일 보고 형식으로 병기했으며, 종합 단계는 이미 검증된 Evidence만 사용하도록 제한했다.
- 이 조치에도 출처 범위와 저자 보고 의존성은 남는다. 따라서 본 보고서는 순위·우승·단일 추천을 제시하지 않는다.

#### 근거 공백 (Supervisor 기록)
- 근거 부족: tech: 기술 성숙도(TRL) 재조사 2회 후에도 근거 부족 — 제공된 발췌문에는 ‘메모리 계층(memory hierarchy)’ 자체의 계층 구조, 저장 장치 간 데이터 이동, 또는 관리 정책에 대한 설명은 없습니다.; 제공된 발췌문에는 온라인 RL 프레임워크의 구체적 알고리즘, 하이퍼파라미터, 하드웨어 구성, 데이터셋 규모, 학습 단계 수 및 재현 가능한 실행 설정이 없습니다.; 제공된 발췌문에는 표준 벤치마크 평가의 전체 결과 수치 및 각 벤치마크의 상세 프롬프트·샷 설정이 완전하게
- 근거 부족: market: 시장성 재조사 2회 후에도 근거 부족 — MLA가 속한 LLM 추론 서빙·추론 최적화 시장을 직접 정의하는 신뢰 가능한 시장조사기관 웹 자료(조사기관, 기준연도, 전망연도, 단위, CAGR 포함); ITME가 속한 CXL 메모리 시장의 시장조사기관 웹 자료(조사기관별 기준연도, 전망연도, 단위, CAGR 포함); DeepSeek-V2 MLA 자체를 명시하는 유료 서비스, 독립 고객사 도입 또는 실제 운영 사례
- 근거 부족: stakeholder: 이해관계자 재조사 2회 후에도 근거 부족 — ITME 이해관계자 관련 고유 출처 최소 2개 및 최신 시도 기준의 직접 근거; OpenAI, Google, Anthropic, Meta 등 경쟁 진영의 MLA 또는 DeepSeek MLA 계열에 대한 식별 가능한 직접 발언; ITME 자체에 대한 경쟁사 또는 경쟁 기술 제공자의 대안·한계·병행 가능성 관련 명시적 발언
- 근거 부족: domain: 도메인 적용성 재조사 2회 후에도 근거 부족 — ‘D1 워크로드’의 정의 및 해당 워크로드의 모델/서빙 구성; 제한된 GPU 메모리의 구체적 용량과 GPU 종류; GPU 메모리 제약별 수용 가능한 동시 요청 수
- 근거 부족: tech: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족
- 근거 부족: market: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족
- 근거 부족: stakeholder: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족
- 근거 부족: domain: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족

#### 표 7. 증거 균형표 (중복 제거 후 수집·검증된 Evidence 수)
| 구분 | DeepSeek-V2 MLA | ITME | HW 베이스라인(InfiniGen·CXL-PNM) |
|---|---|---|---|
| 기술 조사 · 논문 청크 | 33 | 47 | 8 |
| 기술 조사 · 웹 문서 | 6 | 4 | 0 |
| 시장 · 웹 문서 | 6 | 4 | 0 |
| 이해관계자 · 웹 문서 | 4 | 3 | 0 |
| 도메인 · 논문 청크 | 40 | 44 | 0 |
| 합계 | 89 | 102 | 8 |

## REFERENCE
- [1] DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, p.1, p.4, p.5, p.6, p.7, p.8, p.12, p.15, p.16, p.21, p.28.
- [2] Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J. (SK hynix) (2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv preprint arXiv:2606.12556, p.1, p.2, p.4, p.7, p.9, p.10, p.11.
- [3] Lee, W., Lee, J., Seo, J., & Sim, J. (Seoul National University) (2024). InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. arXiv preprint arXiv:2406.19707, p.4.
- [4] Kim, D., Lee, M., Kim, J., Kwon, H., Jeong, H., Park, S.-S., et al. (Hanyang University, Samsung Electronics) (2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits. arXiv preprint arXiv:2511.00321, p.4.
- [W5] docs.vllm.ai (게시일 미확인). mla_attention - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention
- [W6] docs.vllm.ai (게시일 미확인). Attention Backend Feature Support - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/stable/design/attention_backends
- [W21] mordorintelligence.com (Tue, 28 Jul 2026 00:00:00 GMT). CXL Memory Controller IC Market Size, Share & 2031 Growth Trends Report. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/cxl-memory-controller-ic-market
- [W1] computeexpresslink.org (게시일 미확인). Breaking Boundaries in Memory: Highlights from AI Infra Summit and SDC 2025 - Compute Express Link. computeexpresslink.org, https://computeexpresslink.org/blog/breaking-boundaries-in-memory-highlights-from-ai-infra-summit-and-sdc-2025-4198
- [W9] github.com (Wed, 09 Sep 2026 12:00:00 GMT). GitHub - deepseek-ai/FlashMLA: FlashMLA: Efficient Multi-head Latent Attention Kernels · GitHub. github.com, https://github.com/deepseek-ai/FlashMLA
- [W17] gminsights.com (Thu, 01 Oct 2026 10:00:00 GMT). Enterprise LLM Market Size, Forecasts Report 2026-2035. gminsights.com, https://www.gminsights.com/industry-analysis/enterprise-llm-market
- [W22] mordorintelligence.com (Fri, 11 Sep 2026 00:00:00 GMT). Large Language Model Market Size, Growth & Outlook | Industry Report 2031. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/large-language-model-llm-market
- [W19] lmsys.org (게시일 미확인). SGLang v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision - LMSYS Org. lmsys.org, https://www.lmsys.org/blog/2024-09-04-sglang-v0-3
- [W25] penguinsolutions.com (게시일 미확인). CXL 3.0 vs. CXL 2.0: Advancing Composable AI Infrastructure. penguinsolutions.com, https://www.penguinsolutions.com/en-us/resources/blog/cxl-30-vs-cxl-20-key-differences-and-benefits
- [W13] news.ycombinator.com (게시일 미확인). DeepSeek Open Source FlashMLA – MLA Decoding Kernel for Hopper GPUs | Hacker News. news.ycombinator.com, https://news.ycombinator.com/item?id=43155023
- [W11] liorsinai.github.io (게시일 미확인). DeepSeek's Multi-Head Latent Attention - Lior Sinai. liorsinai.github.io, https://liorsinai.github.io/machine-learning/2025/02/22/mla.html
- [W14] semiconductor.samsung.com (게시일 미확인). Breaking AI Memory Limits with CXL Memory Pooling. semiconductor.samsung.com, https://semiconductor.samsung.com/news-events/tech-blog/breaking-ai-memory-limits-with-cxl-memory-pooling
- [W23] networkworld.com (게시일 미확인). Chip wafer shortage will run through 2030 as AI demand overwhelms supply: SK Hynix chief | Network World. networkworld.com, https://www.networkworld.com/article/4146270/chip-wafer-shortage-will-run-through-2030-as-ai-demand-overwhelms-supply-sk-hynix-chief.html
- [D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 (A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.
