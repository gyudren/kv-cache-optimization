# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-09-22

## SUMMARY
- **TRL 추정(공개 정보 기반)**: MLA와 ITME 모두 4–6으로 판정한다. MLA는 DeepSeek-V2 통합 및 벤치마크, ITME는 FPGA 프로토타입과 시스템 평가가 근거이나, 실제 상용 운영·고객 배포 근거는 제공 자료에서 확인되지 않는다.[1, p.6] [1, p.32] [2, p.8] [2, p.9]
- **MLA**는 KV 표현 자체를 저랭크 공동 압축하므로 D3 메모리 자원 효율은 적합이다. 반면 서비스 지연시간·운영 안정성은 근거 부족이며, 저자 보고 성능은 MLA 외 최적화가 결합된 조건을 포함한다.[1, p.7] [1, p.16]
- **ITME**는 계층형 원격 메모리와 프리페치로 용량·I/O 병목을 다루며 D1·D2·D3은 조건부다. 그러나 장기 멀티턴에서 I/O 경합과 재계산 가능성이 확인되어 D5는 제약이다.[2, p.2] [2, p.9] [2, p.10]
- 시장성과 이해관계자 신호는 제공 웹 근거의 작성 주체·확인일·원문 검증 정보가 불충분하여, 신호표의 혼재·우려 판정은 유지하되 독립 검증된 채택 또는 시장 사실로 해석할 수 없다.
- 두 기술은 경쟁적 단일 대안이 아니라, MLA가 모델 내부 캐시량을 줄이고 ITME가 남는 상태의 저장·이동 계층을 확장하는 상호보완적 접근이다. 결합 효과는 근거 부족이다.

## 1. 분석 배경
#### 문제 정의

KV cache는 생성 추론에서 이전 토큰의 key와 value를 저장해 이후 attention 계산을 가속하는 상태다. 표준 MHA는 추론 시 모든 key와 value를 캐시해야 하며, 이 부담은 최대 batch 크기와 시퀀스 길이를 제한하는 배포 병목으로 설명된다.[1, p.7]

데이터센터·클라우드 기반 장문맥 LLM 서빙에서는 대규모 동시 요청, 비용 민감성, 문맥 길이 민감성이 함께 존재한다. 설계 기준상 KV cache는 레이어 수·attention head 수·문맥 길이·동시 사용자 또는 batch·저장 정밀도에 비례해 증가하며, HBM 점유는 동시 요청 수와 최대 batch 감소, 장문맥 제한, GPU 증설 비용, KV 이동·로딩 지연 및 폐기 후 재계산으로 이어질 수 있다.[D]

#### 관점 비교의 이유

본 평가는 단일 우열이나 추천을 정하지 않는다. MLA는 attention 구조에서 캐시해야 할 표현량을 줄이는 접근이고, ITME는 GPU·호스트·SSD 사이 메모리 계층의 용량과 데이터 이동을 다루는 접근이므로, 같은 단위의 성능 수치로 직접 비교하기 어렵다.[1, p.6] [2, p.2]

따라서 기술 성숙도, 시장성, 이해관계자, 데이터센터·클라우드 장문맥 LLM 서빙 도메인 적용성이라는 서로 다른 관점에서 장점·제약·근거 공백을 병기한다. 이는 공통 문제인 KV cache 메모리 병목에 대한 서로 다른 작동 계층의 대응을 구분하기 위한 설계 기준이다.[D]

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 3% |
| 128K | 40 GiB | 320 GiB | 50% |
| 1M | 320 GiB | 2,560 GiB | 400% |

산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. 요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). HBM 대비 비율은 80GB를 기준으로 한 근사치 [D].

## 2. 기술 선정
#### 선정 방식과 기준

선정 방식은 **2안: 조 토의 후 직접 선정**이다. 평가 기준은 TRL, 시장성, 이해관계자, D1–D7 도메인 적용성이며, 목적은 우승 기술 선정이 아니라 발전 단계·장단점·관점 간 일치와 충돌·도입 전 확인사항을 근거와 함께 제시하는 것이다.[D]

#### DeepSeek-V2 MLA 선정 이유

MLA는 key와 value를 저차원 잠재 벡터로 공동 압축하도록 attention 구조를 재설계해 추론 시 KV cache 병목을 줄이는 소프트웨어·모델 아키텍처다.[1, p.6] 추론 시 압축 latent vector를 캐시하는 설계가 명시되어 있어, KV cache 자체의 크기를 줄이는 접근을 평가하기 위해 선정했다.[1, p.7]

설계 검토에서는 기존 모델에 사후 적용하기 어려울 수 있다는 한계를 함께 고려했다. KIVI와 TurboQuant는 검토 의견으로 언급되었으나 본 평가의 직접 선정 대상은 아니다.[D]

#### ITME 선정 이유

ITME는 CXL-hybrid memory를 중간 계층으로 사용하고, SSD 내부 DRAM cache·RDMA·호스트 메모리를 거치는 프리페치로 GPU 연산과 데이터 이동을 중첩하는 하드웨어·시스템 아키텍처다.[2, p.2] 장문맥 KV cache와 모델 가중치의 계층 저장·이동 문제를 평가하기 위해 선정했다.[D]

설계상 ITME는 HBM·호스트 DRAM의 용량 한계를 넘어 장문맥 및 다중 턴 누적 KV cache를 저장하고 재계산을 줄일 가능성을 검토하는 대상이다. 다만 이 가능성은 실제 운영 효과가 아니라 도입 전 검증할 가설로 취급한다.[D]

#### 표 2. 비선정 후보와 사유 (설계 단계 팀 판단 — 설계 산출물 A-4) [D]
| 후보 | 진영 | 비선정 사유 | RAG 문서 활용 |
|---|---|---|---|
| InfiniGen | HW | 호스트 메모리 오프로딩으로 신규 메모리 인프라 없이 SW 관리 성격이 강함 | O (ITME 비교용 베이스라인) |
| CXL-PNM | HW | 연산까지 메모리 측으로 옮기는 확장형 접근으로, '공간 확장' 자체의 대표성은 ITME가 더 직접적 | O (ITME 비교용 베이스라인) |
| KIVI | SW | 사후 양자화의 대표 베이스라인이나 공개 채택 근거가 연구·라이브러리 수준 | X (선정 검토만) |
| TurboQuant | SW | 재학습 없이 적용 가능한 최신 양자화이나 공식 상용화 근거가 제한적 | X (선정 검토만) |

## 3. 기술 개요
#### 기술 대조

| 항목 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 진영 | SW·모델 아키텍처 | HW·시스템 메모리 아키텍처 |
| 작동 계층 | attention 내부 KV 표현 | GPU·호스트·CXL-hybrid memory·SSD 사이 계층 |
| 핵심 접근 | key·value 저랭크 공동 압축 및 latent KV 캐시 | 원격 바이트 주소화 메모리 확장, HW/SW 프리페치, RDMA 데이터 이동 |
| 저자 보고 효과 | DeepSeek-67B 대비 KV cache 93.3% 감소 및 최대 generation throughput 5.76배 향상 | NVMe-oF 기반 분리형 스토리지 기준선 대비 처리량 1.80배 향상 |
| 전제조건 | MLA 구조와 그에 대응하는 모델·가중치·추론 처리 | CXL-hybrid memory, SSD-DRAM cache, RDMA, 프리페치 및 스케줄링 |
| 한계 | 서비스 지연시간·운영 안정성·TCO 직접 근거 부족 | 장기 턴 I/O 경합, 원격 계층 지연, 하드웨어·통합 의존성 |

MLA의 저자 보고 효과는 DeepSeek-V2 전체 구성의 결과이며, MLA 단독 효과나 독립 재현 결과로 해석하지 않는다.[1, p.1] ITME의 처리량 결과도 저자 보고 기준의 특정 기준선·실험 조건 결과다.[2, p.2]

#### MLA: 핵심 접근·효과·제약

MLA는 key와 value를 저랭크로 공동 압축하고, 추론 중 전체 key·value 대신 압축 latent vector를 캐시하도록 설계된다.[1, p.6] [1, p.7] 논문은 MLA의 KV cache가 MHA보다 작은 수준이며, small MoE에서는 MHA의 14%, large MoE에서는 4% 수준이라고 저자 보고한다.[1, p.31]

저자 보고 기준 DeepSeek-V2는 DeepSeek-67B 대비 KV cache 93.3% 감소와 최대 generation throughput 5.76배 향상을 제시한다.[1, p.1] 서비스 관련 서술에서는 FP8 파라미터 변환과 평균 6비트 KV cache 양자화가 함께 적용되었으므로, 해당 결과를 MLA 단독 성능으로 귀속할 수 없다.[1, p.16]

MLA는 모델 attention 구조 자체의 변경을 전제로 한다. 기존 MHA 체크포인트의 변환·재학습 절차, 런타임 통합 작업량, TTFT 및 장기 동시성 안정성은 제공 자료만으로 확인되지 않는다.

#### ITME: 핵심 접근·효과·제약

ITME는 장문맥 KV cache와 모델 가중치처럼 용량이 크고 접근 패턴이 예측 가능한 데이터를 원격 확장 계층에 배치하고, SSD에서 CXL-hybrid memory 내부 DRAM cache로의 HW 프리페치와 RDMA·호스트 메모리를 경유하는 DMA 프리페치를 GPU 연산과 중첩한다.[2, p.2]

저자 보고 기준 ITME는 대규모 LLM 추론에서 NVMe-oF 기반 분리형 스토리지 기준선 대비 처리량 1.80배 향상을 보였다.[2, p.2] 또한 FPGA 프로토타입은 Intel Agilex 7 I-Series FPGA, 32GB DDR4 DRAM cache, PCIe Gen5 NVMe SSD 2개를 사용한 구성으로 기술된다.[2, p.8]

ITME는 CXL-hybrid memory·RDMA·프리페처·스케줄링을 필요로 하며 단순 소프트웨어 설정 변경으로 도입되는 방식은 아니다.[2, p.2] KV cache 증가에 따라 읽기·쓰기 트래픽이 CXL-hybrid memory 내부 I/O 경합을 일으켜 지연 블록의 재계산을 유발할 수 있다는 제약도 보고된다.[2, p.10]

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | technology_trl: 4–6 (판정). 논문은 DeepSeek-V2에 MLA를 설계·통합했고, MLA-MHA 비교 및 모델 벤치마크를 제시한다[1, p.6; p.15; p.32]. 이는 구현 및 실험적 검증 근거이지만, 제공 발췌에는 실제 서비스 운영·상용 채택 근거가 없다. family_trl: 근거 부족. MLA가 속한 attention/KV-cache 최적화 계열 전체의 제품 배포 또는 상용 운용 성숙도를 판정할 직접 근거가 제공되지 않았다. | fact: MLA는 저랭크 key-value 공동 압축으로 추론 시 KV-cache 병목을 완화하도록 설계되었다[1, p.6]. 추론 시 캐시 대상은 압축 latent vector c_t^KV이며, 캐시 원소 수는 층 수 l에 대해 l·d_c로 설명된다[1, p.7]. 저자 보고 기준 DeepSeek-V2는 DeepSeek-67B 대비 KV cache 93.3% 감소 및 최대 generation throughput 5.76배 향상을 보고한다[1, p.1]. MLA-MHA hard benchmark 비교와 공통 평가 설정의 모델 벤치마크가 제시된다[1, p.15; p.32]. interpretation: 아키텍처 통합 및 벤치마크 검증은 TRL 4–6 정의의 ‘통합 구현·프로토타입·유사 운용 환경 시연’에 부합하는 증거이나, 논문만으로 실제 운용을 확인할 수 없다. judgment: MLA 자체는 TRL 4–6으로 보수적으로 추정한다. MLA 계열 전체는 제공 자료만으로 판정하지 않는다. |
| ITME | technology_trl: 4–6 (판정). 논문은 FPGA 기능 프로토타입과 CMM 기반 평가 플랫폼을 기술하고, ShareGPT 256개 동시 대화 및 35-turn 실험을 포함한 성능 검증을 보고한다[2, p.8; p.9; p.11]. 다만 제공 발췌에는 ITME 자체의 제품 출시, 고객 배포 또는 상용 서비스 운용 근거가 없다. family_trl: 근거 부족. 논문은 CXL 기반 메모리 확장·풀링이 현대 데이터센터의 핵심 enabling technology로 부상했다고 서술하지만[2, p.11], CXL 메모리 모듈 또는 CXL 계열의 실제 제품·상용 운용을 입증하는 허용된 직접 근거는 제공되지 않았다. | fact: ITME는 CXL-hybrid memory를 T3.5 계층으로 두고, SSD→내부 DRAM cache→RDMA/host CPU memory의 프리페치와 GPU 연산 중첩을 설명한다[2, p.2]. FPGA 프로토타입은 Intel Agilex 7 I-Series FPGA, 32GB DDR4 DRAM cache, PCIe Gen5 NVMe SSD 2개로 구성되었고, CMM 기반 플랫폼은 약 22GB/s 피크 읽기 처리량으로 기술된다[2, p.8]. 저자 보고 기준 ITME는 NVMe-oF 기반 분리형 스토리지 기준선 대비 1.80배 처리량 향상을 보였으며[2, p.2], FPGA ITME를 포함한 Llama-3.1 8B/70B 비교와 ShareGPT 256개 동시 대화·35-turn workload 평가가 제시된다[2, p.9; p.11]. interpretation: 하드웨어 프로토타입과 시스템 실험은 TRL 4–6의 구현·검증 근거다. 그러나 이것은 ITME 개별 아키텍처의 실제 상용 운용이나 CXL 제품군 일반의 상용 성숙도를 뜻하지 않는다. judgment: ITME 자체는 TRL 4–6으로 보수적으로 추정한다. CXL 메모리 계열의 TRL은 제공 발췌만으로는 근거 부족이다. |

#### 판정 기준과 범위

TRL 4–6은 설계 기준상 코드 공개·도구 탑재 등 구현 근거를 확인하는 범주이며, 본 절의 수치는 공식 인증 등급이 아니라 논문·프로토타입·실험 공개 자료에 한정한 **공개 정보 기반 추정**이다.[D] 통합 구현, 프로토타입, 유사 운용 환경 시연은 이 내부 기준에서 4–6 판단을 위한 근거로 사용한다.[D]

#### DeepSeek-V2 MLA: 4–6 (판정)

MLA는 DeepSeek-V2에 저랭크 key-value 공동 압축 attention으로 설계·통합되어 있으며, MLA와 MHA의 hard benchmark 비교가 제시된다.[1, p.6] [1, p.32] MLA-MHA 비교에서 KV cache per token과 benchmark 결과가 병기되어 구현 및 실험 검증 근거가 존재한다.[1, p.32]

따라서 MLA 자체는 공개 정보 기준 TRL 4–6으로 보수적으로 추정한다. 다만 논문 수치와 서비스 관련 표현은 모두 저자 보고 기준이며, 실제 고객 운영·상용 채택을 독립적으로 확인하는 근거는 제공되지 않았다.[1, p.1] [1, p.16] MLA가 속한 attention 또는 KV-cache 최적화 계열 전체의 TRL은 직접 근거 부족이다.

#### ITME: 4–6 (판정)

ITME는 FPGA 기능 프로토타입과 CMM 기반 평가 환경을 제시한다. FPGA 프로토타입은 Intel Agilex 7 I-Series FPGA 및 32GB DDR4 DRAM cache를 포함하며, CMM 기반 구성은 약 22GB/s 피크 읽기 처리량으로 기술된다.[2, p.8] ShareGPT 256개 동시 대화와 35-turn KV cache 스트레스 시험도 제시된다.[2, p.9]

따라서 ITME 개별 아키텍처는 공개 정보 기준 TRL 4–6으로 보수적으로 추정한다. 이는 프로토타입과 시스템 실험의 존재를 뜻할 뿐, ITME 자체의 제품 출시·고객 배포·현장 운영을 뜻하지 않는다.[2, p.10] [2, p.11]

CXL 메모리 모듈 일반 또는 CXL 제품군 전체의 성숙도를 ITME의 성숙도로 대체할 수 없다. 논문은 상용 CXL 하드웨어의 제한된 가용성 때문에 다수 연구가 시뮬레이션·에뮬레이션에 의존했다고 서술하나, 제공 자료만으로 CXL 계열 일반의 TRL을 판정하기에는 근거 부족이다.[2, p.11]

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 긍정 | 혼재 |
| 상용화·채택 현황 | 혼재 | 혼재 |
| 생태계 지지 | 긍정 | 혼재 |
| 종합 | 혼재 | 혼재 |
| 근거 | [W22] [W5] [W9] [W6] [W12] | [W4] [W15] [W2] [W1] |

#### 평가 범위

시장성 신호표의 MLA 종합 판정과 ITME 종합 판정은 모두 **혼재**다. 다만 제공된 웹 카탈로그는 다수 항목에서 작성 주체, 확인일, 원문 검증용 메타데이터가 충분하지 않아, 아래 판정은 검증된 시장 규모·채택 사실의 재확인이 아니라 신호표를 보수적으로 해석한 결과다.

#### DeepSeek-V2 MLA: 혼재

MLA의 시장 규모·성장성은 **긍정**, 상용화·채택은 **혼재**, 생태계 지지는 **긍정**으로 제시되어 있다. 그러나 [W22]의 시장 전망 수치, [W5]의 DeepSeek 모델 관련 서비스 신호, [W6]·[W12]의 MLA 지원·성능 표현은 본 입력에서 작성 주체·확인일·원문 검증 정보가 충분하지 않다. 따라서 이를 MLA 자체의 시장 규모, 고객 채택, 독립 성능 검증으로 서술하지 않는다.

특히 DeepSeek 모델 지원 사례와 MLA 기능 자체의 직접 지원 사례는 구분해야 한다. 제공 웹 자료만으로는 DeepSeek-V2 MLA의 고객명, 도입 시점, 유료 운영 상태를 확인할 수 없어 상용화·채택 신호는 혼재로 제한한다.

#### ITME: 혼재

ITME의 시장 규모·성장성, 상용화·채택, 생태계 지지는 모두 **혼재**로 제시되어 있다. [W4]의 CXL memory pooling software 시장 수치, [W15]의 CXL 일반 운영 사례, [W2]의 생태계·배포 표현은 ITME 개별 기술의 제품 출시나 고객 운용을 직접 입증하지 않는다. 또한 해당 웹 근거의 검증 메타데이터가 부족하다.

ITME는 특정 CXL-hybrid memory 아키텍처이며, CXL 일반 또는 CXL 메모리 풀링 소프트웨어 시장의 성장 신호를 ITME의 시장 규모·매출·채택으로 전환할 수 없다. ITME 자체의 공식 제품 정의, 출시 상태, 고객 레퍼런스, 공식 런타임 통합 문서는 공개 근거 부족이다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 근거 부족 | 우려 |
| 도입 기업·개발자 | 혼재 | 근거 부족 |
| 투자 업계 | 혼재 | 근거 부족 |
| 종합 | 혼재 | 우려 |
| 근거 | [W23] [W11] [W17] | [W14] [W10] |

#### 평가 범위

이해관계자 신호표에서 MLA 종합은 **혼재**, ITME 종합은 **우려**다. 다만 [W10]·[W11]·[W14]·[W17]·[W23] 등 웹 근거는 원문 검증을 위한 작성 주체·확인일 또는 직접성 정보가 충분하지 않아, 아래는 독립 확인된 기업 입장으로 확정하지 않는다.

#### DeepSeek-V2 MLA: 혼재

경쟁 진영의 MLA 직접 평가는 **근거 부족**이다. 제공 자료만으로 경쟁 기업이나 대체 기술 진영이 MLA 자체의 한계·대체성·병행 가능성을 명시적으로 평가했다는 사실은 확인되지 않는다.

도입 기업·개발자 신호는 **혼재**다. [W23]은 vLLM 관련 MLA 최적화 성능을 언급하고, [W11]은 특정 이용자의 서빙 엔진 비교 경험을 담고 있으나, 전자는 독립 고객 도입 근거가 아니며 후자는 특정 환경의 커뮤니티 경험이다. 따라서 MLA 일반의 운영 성능이나 채택 현황으로 일반화할 수 없다.

투자 업계 신호도 **혼재**다. [W17]은 DeepSeek 계열과 AI 인프라에 관한 분석을 포함하지만 MLA 단독의 수요·수익성·가치평가를 직접 다루는 투자자 근거는 공개 근거 부족이다.

#### ITME: 우려

ITME의 경쟁 진영 신호는 **우려**다. [W14]는 CXL이 HBM을 대체하지 않고 보조 계층을 보완한다는 취지의 CXL 일반 발언을 포함한다. 이는 ITME의 계층형 확장 목적과 관련된 간접 제약 신호일 수 있으나, ITME 자체의 실측 결과나 도입성을 직접 부정하는 증거는 아니다.

도입 기업·개발자와 투자 업계의 ITME 직접 의견은 모두 **근거 부족**이다. ITME를 실제로 도입·운영한 클라우드 사업자 또는 개발자의 운영 경험, 그리고 식별 가능한 투자자·애널리스트의 ITME 직접 평가는 제공되지 않았다.

### 4.4 도메인 적용성(D1-D7)
| 평가항목 | MLA 판정 | MLA 근거 | ITME 판정 | ITME 근거 |
|---|---|---|---|---|
| D1 워크로드 수용 능력 | 조건부 | [1, p.1] [1, p.16] | 조건부 | [2, p.9] |
| D2 서비스 성능 | 근거 부족 | [1, p.16] | 조건부 | [2, p.2] [2, p.9] |
| D3 메모리 자원 효율 | 적합 | [1, p.8] [1, p.31] | 조건부 | [2, p.9] |
| D4 품질·정보 보존 | 조건부 | [1, p.13] | 근거 부족 | [2, p.3] [2, p.7] |
| D5 운영 안정성 | 근거 부족 | - | 제약 | [2, p.9] |
| D6 도입·확장 용이성 | 조건부 | [1, p.6] [1, p.8] | 조건부 | [2, p.2] [2, p.9] |
| D7 비용 효율성 | 조건부 | [1, p.1] [1, p.16] | 조건부 | [2, p.2] [2, p.9] |

#### DeepSeek-V2 MLA

D1은 **조건부**다. 저자 보고 기준 DeepSeek-V2는 128K context를 지원하며, 서비스 배포 서술에서는 MLA와 FP8 파라미터·평균 6비트 KV cache 양자화를 함께 적용해 더 큰 batch를 제공할 수 있다고 설명한다.[1, p.1] [1, p.16] 다만 GPU 용량별 최대 동시 요청 수와 MLA 단독 기여도는 확인되지 않는다.

D2는 **근거 부족**이다. 저자 보고 기준 단일 8×H800 노드에서 generation throughput이 50K tokens/s를 넘는다고 제시되지만, 이 결과는 MLA와 추가 최적화가 결합된 서비스 조건이다.[1, p.16] TTFT, 요청 단위 지연시간, p95/p99 및 메모리 압박 하 지연시간은 제공되지 않았다.

D3는 **적합**이다. MLA는 추론 중 압축 latent vector를 캐시하도록 설계됐고,[1, p.7] 저자 보고 비교에서 small·large MoE의 KV cache가 MHA의 각각 14%, 4% 수준으로 제시된다.[1, p.31]

D4는 **조건부**다. 저자 보고 기준 32K 길이에서 1000 step 추가 학습한 모델이 128K context 평가 및 NIAH에서 견고한 성능을 보였다고 제시된다.[1, p.13] 다만 이 결과는 MLA 단독 효과를 분리하지 않았고 세부 점수도 제공되지 않았다.

D5는 **근거 부족**이다. 동시 멀티턴 요청, cache miss, 경합, 부하별 성능 변동 및 장기 안정성을 직접 다룬 공개 근거가 제공되지 않았다.

D6은 **조건부**다. MLA는 저랭크 KV 공동 압축에 기반한 attention 구조이며, 추론 시 해당 구조의 KV 처리와 모델 구조·가중치가 전제된다.[1, p.6] [1, p.8] 기존 MHA 모델의 변환·재학습 또는 런타임 통합 절차는 근거 부족이다.

D7은 **조건부**다. 저자 보고 KV cache 감소와 더 큰 batch 가능성은 GPU 메모리 요구 감소 가능성을 시사하지만,[1, p.1] [1, p.16] 동등 워크로드의 GPU 수, 인프라 비용, 총소유비용 비교는 제공되지 않았다.

#### ITME

D1은 **조건부**다. ITME는 ShareGPT 기반 256개 동시 대화와 35-turn KV cache 스트레스 시험으로 평가됐다.[2, p.9] 그러나 문맥 길이 상한과 동일 GPU 메모리 조건에서 증가한 동시 요청 수의 직접 비교는 제공되지 않았다.

D2는 **조건부**다. 저자 보고 기준 NVMe-oF 기반 분리형 스토리지 기준선 대비 처리량 1.80배 향상이 제시된다.[2, p.2] 그러나 절대 TTFT·token latency·prompt throughput은 제공되지 않았고, 성능은 CXL-hybrid memory·프리페치·접근 패턴에 의존한다.

D3은 **조건부**다. ITME는 CXL-hybrid memory를 활용하고 호스트 메모리 일부를 staging buffer로 사용한다.[2, p.9] GPU 상주 KV 부담을 줄이는 방향의 실험 근거는 있으나, CXL·호스트·NVMe를 포함한 총 메모리 소비와 token당 KV 바이트는 완전하게 제시되지 않았다.

D4는 **근거 부족**이다. 논문은 장문맥 prefix KV를 ITME 계층에 두는 데이터 배치와 cache miss 시 재계산 정책을 설명하지만,[2, p.2] [2, p.7] 출력 품질, 정확도, 장문맥 정보 보존을 직접 평가한 결과는 제시하지 않는다.

D5는 **제약**이다. KV cache가 턴마다 증가하면서 읽기·쓰기 트래픽이 CXL-hybrid memory 내부 I/O 경합을 일으켜, 일부 블록이 staging buffer에 제때 도달하지 못하고 재계산될 수 있다고 보고한다.[2, p.10] 따라서 장기 멀티턴 운영 안정성은 조건이 아니라 직접 확인된 제약으로 다룬다.

D6은 **조건부**다. ITME는 CXL-hybrid memory, SSD-DRAM cache, RDMA 및 HW/SW 프리페처를 포함한다.[2, p.2] 저자 보고 기준 구현·평가는 최적화된 vLLM v0.17.0에서 수행됐다.[2, p.9] 다른 런타임 호환성과 통합 작업량은 근거 부족이다.

D7은 **조건부**다. ITME는 비용 효율적 TB-scale 확장을 목표로 제시하지만,[2, p.1] CXL·RDMA·GPU·호스트 메모리·스토리지를 합산한 동등 워크로드 비용 또는 TCO 수치는 제공되지 않았다.

## 5. 종합 의견
#### 관점 매트릭스

| 관점 | MLA | ITME |
|---|---|---|
| TRL | 4–6(공개 정보 기반 추정): 통합·벤치마크 근거는 있으나 실제 상용 운영은 근거 부족 | 4–6(공개 정보 기반 추정): FPGA·시스템 실험 근거는 있으나 실제 제품·고객 운용은 근거 부족 |
| 시장 | 혼재: 생태계·채택 웹 신호는 있으나 검증 메타데이터와 MLA 자체 채택 근거가 부족 | 혼재: CXL 일반 신호와 ITME 개별 상용화 근거를 구분해야 함 |
| 이해관계자 | 혼재: 개발자 신호는 있으나 직접 도입·투자 판단 근거가 부족 | 우려: CXL 일반의 HBM 대체 한계 신호가 있으나 ITME 직접 반응은 부족 |
| 도메인 | D3 적합, D1·D4·D6·D7 조건부, D2·D5 근거 부족 | D1·D2·D3·D6·D7 조건부, D4 근거 부족, D5 제약 |

#### 관점 간 일치

- MLA는 TRL과 D3에서, 모델 통합 및 KV cache 절감의 직접 구조·실험 근거가 존재한다는 점에서 일치한다.[1, p.6] [1, p.7] [1, p.31]
- ITME는 TRL과 도메인 관점에서, 프로토타입·시스템 실험은 있으나 CXL-hybrid memory·RDMA·프리페치 구성에 의존하는 조건부 접근이라는 점에서 일치한다.[2, p.2] [2, p.8] [2, p.9]
- 두 기술 모두 실제 고객 도입, 장기 운영, 장애 복구, 다중 테넌트 SLA 및 TCO에 관한 직접 공개 근거가 부족하다.

#### 관점 간 상충: MLA

| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| KV 절감과 서비스 성능 | D3 적합: 압축 latent KV 캐시 및 MHA 대비 작은 cache 근거 | D2 근거 부족: TTFT·지연시간·꼬리 지연 수치 부재 | 메모리 표현량 감소와 서비스 수준 성능은 다른 측정 대상이다.[1, p.7] [1, p.31] |
| 모델·런타임 활동과 상용 채택 | 시장 생태계 지지 긍정 신호 | TRL 4–6: 실제 상용 운영 근거 부족 | 런타임 기능 지원이나 웹상 언급은 고객 배포·운영 신뢰성을 직접 증명하지 않는다. |

#### 관점 간 상충: ITME

| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| 처리량과 장기 안정성 | D2 조건부: 저자 보고 기준선 대비 처리량 개선 | D5 제약: I/O 경합 및 지연 블록 재계산 | 기준선·프리페치 조건의 처리량 이득과 장기 멀티턴 경합은 동시에 존재할 수 있다.[2, p.2] [2, p.10] |
| 계층 확장과 HBM 관계 | D1·D3 조건부: 원격 계층으로 KV 상태를 확장 | 이해관계자 우려: CXL 일반의 HBM 대체 한계 신호 | ITME의 목표는 HBM 완전 대체가 아니라 계층 확장이지만, 원격 계층의 지연·대역폭은 적용 범위를 제한한다.[2, p.2] |
| CXL 일반 신호와 ITME 상용화 | 시장 혼재: CXL 관련 활동 신호 | TRL 4–6: ITME 개별 제품·고객 운영 근거 부족 | CXL 계열의 성숙도와 ITME 개별 기술의 성숙도는 동일하지 않다.[2, p.11] |

#### 두 접근의 관계

MLA는 모델 내부에서 캐시해야 할 KV 상태의 표현량을 줄이고, ITME는 GPU·호스트·SSD 사이에 남는 상태와 가중치를 저장·이동하는 계층을 확장한다.[1, p.7] [2, p.2] 따라서 두 접근은 보완적 계층에 속하며 직접 경쟁 기술로 단정할 수 없다. 두 기술의 결합 성능, 호환성, 비용 효과는 공개 근거 부족이다.

## 6. 시사점
#### 관점별로 달라지는 평가

MLA는 메모리 자원 효율에서는 직접적인 구조·비교 근거가 있으나, 서비스 지연시간과 운영 안정성에서는 근거 공백이 크다.[1, p.7] [1, p.31] ITME는 프로토타입 환경에서 계층 확장과 처리량 개선을 제시하지만, 장기 멀티턴 I/O 경합과 추가 하드웨어·통합 요건이 도입 조건이다.[2, p.2] [2, p.10]

따라서 도입 평가는 메모리 절감, 처리량, 지연시간, 품질, 안정성, 비용을 하나의 수치로 환산하지 말고 동일 모델·문맥 길이·동시성·SLA 조건에서 분리 측정해야 한다.

#### 이해관계자별 도입 전 확인 과제

| 이해관계자 | MLA 확인 과제 | ITME 확인 과제 |
|---|---|---|
| 클라우드 사업자 | 실제 모델·런타임에서 TTFT, token latency, p95/p99, 동시성별 안정성 측정 | CXL·RDMA 토폴로지, 원격 계층 장애·복구, 장기 턴 I/O 경합, 다중 테넌트 격리 측정 |
| 모델 개발사 | MLA 구조·가중치 전제, 장문맥 품질 절제 실험, 기존 모델 전환·재학습 경로 확인 | prefix KV 접근 패턴, cache miss 재계산 영향, 프리페처 제어 인터페이스 검증 |
| 메모리·HW 벤더 | MLA 런타임의 실제 메모리 요구량과 정밀도별 영향 확인 | CXL-hybrid memory·SSD·DRAM cache·NIC 구성별 대역폭, 지연, 장애 조건 검증 |
| 투자자 | MLA 자체의 고객 도입·운영 지속성·수익화 근거 확인 | ITME 개별 제품 정의·출시 상태·고객 레퍼런스·TCO 자료 확인 |

ITME 도입 검증에서는 원격 계층의 이득뿐 아니라, KV cache 증가 시 읽기·쓰기 경합으로 재계산이 발생할 수 있는 조건을 함께 측정해야 한다.[2, p.10] MLA 검증에서는 저자 보고 서비스 수치가 FP8 및 평균 6비트 KV cache 양자화와 결합됐음을 분리해야 한다.[1, p.16]

## 7. 한계점
#### 공개 정보와 판정의 한계

모든 TRL 판정은 논문, 프로토타입 기술, 상용 발표 여부 등 **공개 정보만을 기반으로 한 추정**이다. 공식 TRL 인증, 고객 배포, 현장 운용 또는 제품 신뢰성 시험을 확인한 판정이 아니다.[D] 논문 수치와 벤치마크 결과는 모두 저자 보고 기준으로 취급하며, 독립 재현 또는 실제 운영 성과로 간주하지 않는다.

기술 발표와 실제 채택 사이에는 시차가 있을 수 있다. 공개 자료에서 채택 근거를 찾지 못한 것은 채택 부재의 증명이 아니며, 본 보고서에서는 이를 근거 부족으로 표기한다. 특히 TRL 4–6 구간은 비공개 통합 코드, 고객 PoC, 하드웨어 공급 상태, 현장 운영 자료가 공개되지 않을 수 있어 정보 공백이 크다.

#### 시장·이해관계자 근거의 한계

제공 웹 카탈로그의 다수 항목은 작성 주체, 확인일, 직접 인용문 또는 원문 검증에 필요한 메타데이터가 충분하지 않다. 따라서 시장 규모, 고객 채택, 기업 입장, 투자자 판단에 관한 웹 기반 주장은 본문에서 확정 사실로 확대하지 않았다. CXL 일반, DeepSeek 모델 일반, MLA 기능 자체, ITME 개별 아키텍처를 서로 대체하는 근거로 사용하지 않았다.

#### 확증편향 방지 조치와 남은 한계

실제로 적용한 조치는 다음과 같다.

- ITME의 한계를 교차 확인하기 위해 HW 베이스라인 논문인 InfiniGen 및 CXL-PNM 자료를 검토했다. 다만 제공 발췌에서 InfiniGen의 메커니즘·측정치·명시 한계는 충분하지 않아, ITME와의 정량 우열 비교에는 사용하지 않았다.[4, p.2] [4, p.10]
- MLA와 ITME를 동일한 보고 형식(쟁점·관점 A·관점 B·엇갈리는 이유)으로 병기했다.[D]
- 종합 단계에서는 이미 검증된 Evidence만 사용하고 신규 검색을 수행하지 않도록 제한했다.[D]

그럼에도 두 기술의 결합 효과, 동일 조건의 직접 비교, 실제 TCO, SLA, 장애 복구, 장기 다중 테넌트 운영은 공개 근거 부족이다. 첨부되는 Evidence 균형표의 건수는 인용 유효성이나 독립 재현을 대체하지 않는다.

#### 표 7. 증거 균형표 (중복 제거 후 수집·검증된 Evidence 수)
| 구분 | DeepSeek-V2 MLA | ITME | HW 베이스라인(InfiniGen·CXL-PNM) |
|---|---|---|---|
| 기술 조사 · 논문 청크 | 32 | 39 | 6 |
| 기술 조사 · 웹 문서 | 0 | 5 | 0 |
| 시장 · 웹 문서 | 7 | 6 | 0 |
| 이해관계자 · 웹 문서 | 4 | 3 | 0 |
| 도메인 · 논문 청크 | 39 | 45 | 0 |
| 합계 | 82 | 98 | 6 |

## REFERENCE
- [1] DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, p.1, p.6, p.7, p.8, p.13, p.16, p.31, p.32.
- [2] Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J. (SK hynix) (2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv preprint arXiv:2606.12556, p.1, p.2, p.3, p.7, p.8, p.9, p.10, p.11.
- [4] Kim, D., Lee, M., Kim, J., Kwon, H., Jeong, H., Park, S.-S., et al. (Hanyang University, Samsung Electronics) (2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits. arXiv preprint arXiv:2511.00321, p.2, p.10.
- [W22] precedenceresearch.com (Wed, 09 Sep 2026 00:00:00 GMT). AI Inference-as-a-Service Market Companies, Size & Trends 2026 .... precedenceresearch.com, https://www.precedenceresearch.com/ai-inference-as-a-service-market
- [W5] docs.dynatrace.com (Fri, 28 Aug 2026 00:00:00 GMT). AI Observability integrations — Dynatrace Docs. docs.dynatrace.com, https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/integrations
- [W9] github.com (게시일 미확인). vllm/vllm/model_executor/layers/attention/mla_attention.py at main. github.com, https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py
- [W6] docs.vllm.ai (게시일 미확인). mla_attention - vLLM Documentation. docs.vllm.ai, https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention
- [W12] sgl-project-sglang-93.mintlify.app (게시일 미확인). DeepSeek Models - SGLang. sgl-project-sglang-93.mintlify.app, https://sgl-project-sglang-93.mintlify.app/models/deepseek
- [W4] dataintelo.com (게시일 미확인). CXL Memory Pooling Software Market. dataintelo.com, https://dataintelo.com/report/cxl-memory-pooling-software-market
- [W15] forbes.com (Sat, 18 Jul 2026 00:00:00 GMT). AI Storage & Memory From Backblaze, CoreWeave, Panmnesia, Vast And Cloudera - Forbes. forbes.com, https://www.forbes.com/sites/tomcoughlin/2026/07/18/ai-storage--memory-from-backblaze-coreweave-panmnesia-vast-and-cloudera/
- [W2] computeexpresslink.org (게시일 미확인). Breaking Boundaries in Memory: Highlights from AI Infra Summit and SDC 2025 - Compute Express Link. computeexpresslink.org, https://computeexpresslink.org/blog/breaking-boundaries-in-memory-highlights-from-ai-infra-summit-and-sdc-2025-4198
- [W1] arxiv.org (게시일 미확인). Disaggregated LLM Serving withCXL Shared Memory KV Cache at .... arxiv.org, https://arxiv.org/html/2512.18194v1
- [W23] redhat.com (게시일 미확인). Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM. redhat.com, https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm
- [W11] news.ycombinator.com (게시일 미확인). DeepSeek Open Source FlashMLA – MLA Decoding Kernel for .... news.ycombinator.com, https://news.ycombinator.com/item?id=43155023
- [W17] guinnessgi.com (게시일 미확인). How has DeepSeek affected the AI Market for investors?. guinnessgi.com, https://www.guinnessgi.com/insights/articles/how-has-deepseek-affected-ai-market-investors
- [W14] chosun.com (게시일 미확인). CXL Can't Replace HBM in AI Era. chosun.com, https://www.chosun.com/english/industry-en/2026/09/17/DPBT3YGLRJEXVHCBEOSVQ3GZNI
- [W10] news.sbs.co.kr (게시일 미확인). AI Market: HBM's Dominance Expected to Continue... 'CXL .... news.sbs.co.kr, https://news.sbs.co.kr/english/article.do?news_id=N1008757342
- [D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 (A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.
