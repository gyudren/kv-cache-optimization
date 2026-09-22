# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-09-22

## SUMMARY
- **TRL 추정:** MLA와 ITME 개별 기술은 각각 공개 논문상의 통합 구현·프로토타입 및 실험 근거에 따라 **4–6**으로 보수 추정된다. 다만 MLA 계열과 CXL 메모리 계열 전체의 TRL은 근거 부족이며, 실제 상용 운영은 확인되지 않았다[1, p.6] [1, p.32] [2, p.8] [2, p.9].
- **MLA:** KV cache 자체를 줄이는 구조적 근거와 런타임 생태계 신호는 긍정적이지만, 서비스 지연시간·운영 안정성·개별 고객 도입은 근거 부족 또는 혼재다[1, p.7] [1, p.31] [W6] [W12].
- **ITME:** 프로토타입 처리량 및 계층 확장 가능성은 확인되지만, 장기 멀티턴 I/O 경합, CXL의 HBM 비대체성, ITME 자체 채택 공백이 우려를 만든다[2, p.2] [2, p.10] [W14]. 시장 신호는 CXL 일반의 배포·풀링 활동에 근거한 혼재다[W2].
- 두 접근은 경쟁적 단일 대안이 아니라, MLA는 모델 내부 KV 상태를 축소하고 ITME는 남은 상태의 저장·이동 계층을 확장하는 **보완적 계층**으로 해석된다. 결합 효과는 공개 근거 부족이다.

## 1. 분석 배경
#### 문제 정의

KV cache는 생성 추론에서 이전 토큰의 key와 value를 보관해 후속 attention 계산을 가속하는 상태다. 일반 MHA는 추론 중 모든 key와 value를 캐시해야 하며, 이 부담은 배포 시 최대 batch와 시퀀스 길이를 제한하는 병목으로 제시된다[1, p.7].

설계 기준상 KV cache 규모는 레이어 수·attention head 수·문맥 길이·동시 사용자 또는 batch·저장 정밀도에 비례해 증가한다. HBM 점유가 커지면 동시 요청 수와 최대 batch 감소, 장문맥 요청 제한, GPU 추가 비용, KV 이동·로딩 지연, 폐기 후 재계산 문제가 발생할 수 있다[D].

#### 데이터센터·클라우드 장문맥 조건

본 평가는 대규모 동시 요청, 비용 민감성, 문맥 길이 민감성이 함께 나타나는 데이터센터·클라우드 기반 장문맥 LLM 서빙을 대상으로 한다[D]. ITME 논문도 장문맥 prefix KV와 모델 가중치를 대용량·예측 가능한 데이터로 분류하고, 이를 원격 확장 계층에 배치하는 접근을 제안한다[2, p.2].

#### 비교 원칙

평가 목적은 단일 우열이나 최종 추천이 아니라, 동일한 KV cache 메모리 병목에 대한 소프트웨어 압축과 하드웨어 메모리 확장이 기술 성숙도, 시장성, 이해관계자, 도메인 적용성에서 어떻게 다르게 평가되는지 밝히는 데 있다[D]. 두 기술은 작동 계층과 측정 조건이 달라 논문 수치를 직접 우열 비교로 해석하지 않는다.

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB(≈74.5GiB) 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 약 3.4% |
| 128K | 40 GiB | 320 GiB | 약 53.7% |
| 1M | 320 GiB | 2,560 GiB | 약 429.5% |

산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. 요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). HBM 대비 비율은 80GB = 80×10⁹B ≈ 74.5GiB를 분모로 계산했다(설계서 A-2의 3%·50%·400%는 GiB/GB 단위를 혼용한 근사치라 여기서 바로잡음) [D].

## 2. 기술 선정
#### 선정 방식

기술 선정은 **2안: 조 토의 후 직접 선정** 방식으로 수행했다[D]. 선정 기준은 장문맥 KV cache 병목과의 직접성, 기술 성숙도·상용화·생태계 확인 가능성, 그리고 데이터센터·클라우드 서빙 적용성이다[D].

#### DeepSeek-V2 MLA 선정 이유

MLA는 key와 value를 저랭크 잠재 벡터로 공동 압축하여 추론 시 KV cache 병목을 줄이도록 설계된 attention 구조다[1, p.6]. 기존 모델의 사후 적용이 쉽지 않을 수 있다는 제약에도, 모델 수준에서 KV 상태 자체를 줄이는 접근을 평가하기 위해 선정했다[D].

#### ITME 선정 이유

ITME는 CXL-hybrid memory를 T3.5 계층으로 두고, SSD 내부 DRAM cache·RDMA·호스트 메모리를 거치는 프리페치 파이프라인으로 데이터 이동과 GPU 연산을 중첩하는 시스템 아키텍처다[2, p.2]. 장문맥·다중 턴에서 누적되는 KV cache 및 모델 가중치의 용량·이동 병목을 계층 확장 관점에서 다루므로 선정했다[D].

비선정 후보와 그 비교 사유는 검증된 입력 범위에서 별도로 확정하지 않는다.

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
| 진영 | 소프트웨어·모델 아키텍처 | 하드웨어·시스템 메모리 아키텍처 |
| 작동 계층 | attention 내부 KV 표현 | GPU·호스트·CXL-hybrid memory·SSD 사이 계층 |
| 핵심 접근 | key·value 저랭크 공동 압축 | 원격 바이트 주소화 메모리와 HW/SW 프리페치 |
| 저자 보고 효과 | DeepSeek-67B 대비 KV cache 93.3% 감소, 최대 생성 처리량 5.76배[1, p.1] | NVMe-oF 분리형 스토리지 기준선 대비 처리량 1.80배[2, p.2] |
| 전제조건 | MLA attention 구조와 해당 모델·가중치 | CXL-hybrid memory, SSD, RDMA, 프리페처 및 런타임 통합 |
| 한계 | 서비스 지연시간·운영 안정성·TCO 직접 근거 부족 | 장기 턴 I/O 경합, 원격 계층 지연 및 장비 통합 의존 |

#### MLA

MLA는 key와 value를 저랭크로 공동 압축하고, 추론 시 압축 latent vector를 캐시하는 방식이다[1, p.6] [1, p.7]. 저자 보고 기준으로 DeepSeek-V2는 DeepSeek-67B 대비 KV cache를 93.3% 줄이고 최대 생성 처리량을 5.76배 높였다고 보고한다[1, p.1]. 다만 이 결과는 DeepSeekMoE 및 서비스 최적화가 결합된 모델 결과이므로 MLA 단독 효과로 일반화할 수 없다.

MLA-MHA 비교에서는 소형·대형 MoE 모두에서 MLA의 KV cache가 MHA 대비 각각 14%, 4% 수준이라고 보고됐고[1, p.31], hard benchmark 표에서는 MLA와 MHA의 KV cache 원소 수 및 성능을 함께 제시한다[1, p.32]. 기존 MHA 체크포인트를 MLA로 변환하거나 재학습하는 절차는 공개 발췌에서 확인되지 않는다.

#### ITME

ITME는 예측 가능한 모델 가중치와 장문맥 KV cache를 원격 확장 계층으로 오프로드하고, SSD에서 CXL-hybrid memory 내부 DRAM cache로의 하드웨어 프리페치와 RDMA·호스트 메모리를 통한 다계층 DMA 프리페치를 GPU 연산과 중첩한다[2, p.2].

저자 보고 기준 ITME는 하드웨어 프로토타입 평가에서 NVMe-oF 기반 분리형 스토리지 기준선 대비 처리량 1.80배 향상을 보였다[2, p.2]. 또한 CXL-hybrid memory DRAM cache를 넘는 70B 모델 풋프린트에서도 기준선 대비 1–5% 차이를 유지했다고 보고한다[2, p.10]. 반면 KV cache가 턴마다 증가하면 읽기·쓰기 트래픽이 CXL-hybrid memory 내부 I/O 경합을 유발하고, 지연된 블록은 재계산될 수 있다[2, p.10].

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | 개별 기술 TRL: 4–6 (판정). 논문은 DeepSeek-V2에 MLA를 설계·통합했고, MLA-MHA 비교 및 모델 벤치마크를 제시한다[1, p.6; p.15; p.32]. 이는 구현 및 실험적 검증 근거이지만, 제공 발췌에는 실제 서비스 운영·상용 채택 근거가 없다. 계열 TRL: 근거 부족. MLA가 속한 attention/KV-cache 최적화 계열 전체의 제품 배포 또는 상용 운용 성숙도를 판정할 직접 근거가 제공되지 않았다. | 사실: MLA는 저랭크 key-value 공동 압축으로 추론 시 KV-cache 병목을 완화하도록 설계되었다[1, p.6]. 추론 시 캐시 대상은 압축 latent vector c_t^KV이며, 캐시 원소 수는 층 수 l에 대해 l·d_c로 설명된다[1, p.7]. 저자 보고 기준 DeepSeek-V2는 DeepSeek-67B 대비 KV cache 93.3% 감소 및 최대 generation throughput 5.76배 향상을 보고한다[1, p.1]. MLA-MHA hard benchmark 비교와 공통 평가 설정의 모델 벤치마크가 제시된다[1, p.15; p.32]. 해석: 아키텍처 통합 및 벤치마크 검증은 TRL 4–6 정의의 ‘통합 구현·프로토타입·유사 운용 환경 시연’에 부합하는 증거이나, 논문만으로 실제 운용을 확인할 수 없다. 판정: MLA 자체는 TRL 4–6으로 보수적으로 추정한다. MLA 계열 전체는 제공 자료만으로 판정하지 않는다. |
| ITME | 개별 기술 TRL: 4–6 (판정). 논문은 FPGA 기능 프로토타입과 CMM 기반 평가 플랫폼을 기술하고, ShareGPT 256개 동시 대화 및 35-turn 실험을 포함한 성능 검증을 보고한다[2, p.8; p.9; p.11]. 다만 제공 발췌에는 ITME 자체의 제품 출시, 고객 배포 또는 상용 서비스 운용 근거가 없다. 계열 TRL: 근거 부족. 논문은 CXL 기반 메모리 확장·풀링이 현대 데이터센터의 핵심 enabling technology로 부상했다고 서술하지만[2, p.11], CXL 메모리 모듈 또는 CXL 계열의 실제 제품·상용 운용을 입증하는 허용된 직접 근거는 제공되지 않았다. | 사실: ITME는 CXL-hybrid memory를 T3.5 계층으로 두고, SSD→내부 DRAM cache→RDMA/host CPU memory의 프리페치와 GPU 연산 중첩을 설명한다[2, p.2]. FPGA 프로토타입은 Intel Agilex 7 I-Series FPGA, 32GB DDR4 DRAM cache, PCIe Gen5 NVMe SSD 2개로 구성되었고, CMM 기반 플랫폼은 약 22GB/s 피크 읽기 처리량으로 기술된다[2, p.8]. 저자 보고 기준 ITME는 NVMe-oF 기반 분리형 스토리지 기준선 대비 1.80배 처리량 향상을 보였으며[2, p.2], FPGA ITME를 포함한 Llama-3.1 8B/70B 비교와 ShareGPT 256개 동시 대화·35-turn workload 평가가 제시된다[2, p.9; p.11]. 해석: 하드웨어 프로토타입과 시스템 실험은 TRL 4–6의 구현·검증 근거다. 그러나 이것은 ITME 개별 아키텍처의 실제 상용 운용이나 CXL 제품군 일반의 상용 성숙도를 뜻하지 않는다. 판정: ITME 자체는 TRL 4–6으로 보수적으로 추정한다. CXL 메모리 계열의 TRL은 제공 발췌만으로는 근거 부족이다. |

#### MLA: 개별 기술과 계열을 구분한 공개정보 추정

MLA 자체의 TRL 4–6은 공식 등급이 아니라 공개 논문 기반의 보수적 추정이다. DeepSeek-V2는 MLA를 저랭크 KV 공동 압축 구조로 설계했으며[1, p.6], Table 9는 MHA와 MLA의 hard benchmark 및 KV cache 원소 수 비교를 제시한다[1, p.32]. 이는 통합 구현과 실험 검증 근거이지만 실제 고객 운영이나 상용 서비스의 독립적 근거는 아니다.

MLA가 속한 attention/KV-cache 최적화 계열 전체의 TRL은 **근거 부족**이다. 제공 자료에는 계열 전체의 제품 배포나 상용 운용 성숙도를 직접 판정할 근거가 없다.

#### ITME: 개별 기술과 CXL 계열을 구분한 공개정보 추정

ITME 자체의 TRL 4–6 역시 공개정보에 근거한 보수적 추정이다. 논문은 Intel Agilex 7 FPGA 기반 기능 프로토타입, 32GB DDR4 DRAM cache 및 PCIe Gen5 NVMe SSD 구성을 설명하고[2, p.8], ShareGPT 256개 동시 대화와 35-turn workload를 이용한 평가를 제시한다[2, p.9]. 이는 구현·시스템 실험 근거이나, ITME 자체의 제품 출시·고객 배포·상용 서비스 운용을 확인하지는 않는다.

CXL 메모리 모듈 또는 CXL 계열 일반의 TRL은 **근거 부족**이다. 논문은 상용 CXL 하드웨어 가용성이 제한적이어서 많은 연구가 시뮬레이션이나 에뮬레이션에 의존했다고 서술한다[2, p.11]. CXL 일반의 시장·표준 활동을 ITME 개별 기술의 성숙도로 전환할 수 없다.

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 긍정 | 혼재 |
| 상용화·채택 현황 | 혼재 | 혼재 |
| 생태계 지지 | 긍정 | 혼재 |
| 종합 | 혼재 | 혼재 |
| 근거 | [W22] [W5] [W9] [W6] [W12] | [W4] [W15] [W2] [W1] |

#### MLA: 혼재

MLA가 속한 AI 추론 서비스 수요 환경에는 생성형 AI·LLM 사용 증가, 저지연·확장 가능한 추론 수요, 비용 효율적 하드웨어·소프트웨어 최적화 수요가 성장 동인으로 제시된다[W22]. 생태계 측면에서는 vLLM이 MLAAttention과 MLA 관련 backend를 문서화하고[W6], SGLang은 복수 MLA backend 및 기본 활성화된 MLA 최적화를 명시한다[W12]. 이는 MLA 기능 자체의 직접 구현 지원이라는 긍정 신호다.

그러나 DeepSeek 모델의 Amazon Bedrock 관측성 통합은 DeepSeek-V2 MLA 자체의 고객 도입 증거가 아니다[W5]. MLA 자체에 대해 고객명, 운영 기간, 구매·계약 또는 유료 서비스 적용을 확인하는 근거는 부족하다. 따라서 시장 성장성과 생태계는 긍정이나, 상용화·채택은 혼재이며 종합은 혼재다.

#### ITME: 혼재

CXL memory pooling software 시장 전망은 AI 워크로드, 데이터센터 현대화, CXL 3.0 채택을 동인으로 제시하며 성장 신호를 제공한다[W4]. CXL Consortium은 실제 CXL 배포가 증가하고 memory pooling·sharing이 주요 채택 동인이라고 서술한다[W2]. Forbes 보도는 Meta가 프로덕션 서비스에서 CXL을 이용해 분산 AI 추론용 서버 수와 분산 cache 응답 시간을 줄였다고 전한다[W15].

다만 이들은 CXL 일반 또는 특정 사례의 근거이지 ITME의 제품 출시·고객 배포·런타임 공식 통합을 입증하지 않는다. CXL 공유 KV 연구 구현의 Dynamo·vLLM 통합도 ITME 통합 증거로 볼 수 없다[W1]. 그러므로 CXL 계열의 시장·생태계 신호와 ITME 자체 채택 공백이 함께 존재해 종합은 혼재다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 근거 부족 | 우려 |
| 도입 기업·개발자 | 혼재 | 근거 부족 |
| 투자 업계 | 혼재 | 근거 부족 |
| 종합 | 혼재 | 우려 |
| 근거 | [W23] [W11] [W17] | [W14] [W10] |

#### MLA: 혼재

경쟁 진영이 MLA의 한계나 대체 가능성을 직접 평가한 확인 가능한 자료는 없어 경쟁 진영 평가는 **근거 부족**이다. 개발·도입 측에서는 Red Hat이 vLLM의 MLA 지원 및 최적화를 소개하며, 특정 H200·H100 구성에서 MLA 관련 처리량 개선을 보고했다[W23]. 이는 제공자 측 성능 보고이며, 모든 환경의 독립 검증값은 아니다.

반대로 Hacker News의 식별된 사용자는 2개 p5 H100 노드에서 수행한 개인 비교에서 SGLang의 DeepSeek 지원이 vLLM보다 더 좋고 빨랐다고 작성했다[W11]. 이 사례는 특정 환경의 커뮤니티 경험이므로 MLA 자체의 일반 성능 판정이나 기업 채택 증거는 아니다. 투자·업계 관점에서도 DeepSeek 계열의 저비용 개발 가능성과 인프라 지출 지속이 함께 언급되지만, MLA 단독의 수익성·수요·가치평가 근거는 부족하다[W17].

#### ITME: 우려

ITME 자체에 대한 고객·개발자·투자자의 직접 반응은 **근거 부족**이다. 다만 CXL 계열에 관한 Intel 관계자 인용은 CXL 메모리가 HBM을 대체하기보다 보조 저장계층을 보완하며, GPU의 CXL 접근 속도는 HBM만큼 빠르지 않다고 지적한다[W14]. 이는 ITME 직접 평가가 아니라 CXL 일반의 계층 성능 제약에 관한 간접 근거다.

SBS 보도도 CXL의 역할을 대형 모델에서 자주 쓰이지 않는 데이터를 모으는 용도 정도로 언급하며, 특화 use case를 보지 못했다는 발언을 전한다[W10]. 따라서 경쟁 진영의 간접 우려는 존재하지만, ITME 개별 기술에 대한 도입자·투자자 검증은 확보되지 않아 종합 평가는 우려로 한정한다.

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

#### MLA

MLA의 D1은 **조건부**다. 저자 보고 기준 DeepSeek-V2는 128K context를 지원하며[1, p.1], 서비스 구성에서는 MLA와 FP8 파라미터·평균 6비트 KV cache 양자화를 함께 적용해 더 큰 batch를 처리할 수 있다고 설명한다[1, p.16]. 다만 GPU 용량별 동시 요청 수는 제시되지 않았다.

D2는 **근거 부족**이다. 단일 8×H800 노드에서 생성 처리량이 50K tokens/s를 넘는다고 보고됐지만[1, p.16], TTFT와 요청 단위 지연시간, MLA 단독 기여도는 확인되지 않는다. D3은 **적합**이다. MLA는 KV cache를 직접 줄이는 구조이며[1, p.7], 저자 보고 비교에서 MHA 대비 KV cache가 소형·대형 MoE에서 각각 14%, 4% 수준이다[1, p.31].

D4는 **조건부**다. 32K 추가 학습 뒤 128K context와 NIAH 평가에서 견고한 성능이 보고됐지만, 이 결과를 MLA 단독 효과로 분리할 수 없다[1, p.13]. D5는 **근거 부족**이며, 멀티턴 동시 부하·캐시 미스·p95/p99 지연시간 근거가 없다. D6은 **조건부**다. MLA는 기존 MHA와 다른 attention 구조이므로 해당 모델 구조와 KV 처리 지원이 필요하지만[1, p.6] [1, p.8], 변환·재학습·통합 절차는 공개 근거 부족이다. D7도 **조건부**이며, KV 절감은 자원 절감 가능성을 시사하나 동등 워크로드 TCO는 제시되지 않았다[1, p.1] [1, p.16].

#### ITME

ITME의 D1은 **조건부**다. ShareGPT 256개 동시 대화와 35-turn KV cache 스트레스 시험이 제시됐지만[2, p.9], 특정 GPU 메모리에서 늘어난 최대 동시 요청 수나 context 상한의 직접 비교는 없다. D2는 **조건부**다. 저자 보고 기준 NVMe-oF 분리형 스토리지 기준선 대비 처리량 1.80배 향상이 보고됐지만[2, p.2], 절대 TTFT·token latency·prompt throughput은 부족하다.

D3은 **조건부**다. ITME는 Host memory의 일부를 staging buffer로 사용하고 CXL-hybrid memory 계층을 활용한다[2, p.9]. GPU 상주 메모리 부담 완화 근거는 있으나, CXL·호스트·원격 계층의 총자원 및 bytes/token은 충분히 제시되지 않았다. D4는 **근거 부족**이다. ITME는 장문맥 KV를 계층 배치 대상으로 분류하지만[2, p.2], 출력 품질·정확도·정보 보존의 직접 검증은 없다.

D5는 **제약**이다. KV cache 증가에 따라 읽기·쓰기 트래픽이 CXL-hybrid memory 내부 I/O 경합을 일으켜 일부 데이터가 staging buffer에 제때 도달하지 못하고 재계산이 발생할 수 있다고 보고한다[2, p.10]. D6은 **조건부**다. CXL-hybrid memory, RDMA, HW/SW prefetching 및 런타임 통합이 필요하며[2, p.2], 평가 구현은 최적화된 vLLM v0.17.0에서 수행됐다[2, p.9]. D7도 **조건부**다. 비용 효율적 TB-scale 확장이 목표로 제시되지만[2, p.1], 장비 가격과 총소유비용의 직접 비교는 공개 근거 부족이다.

## 5. 종합 의견
#### 관점 매트릭스

| 관점 | MLA | ITME |
|---|---|---|
| TRL | 4–6 추정; 통합·벤치마크는 있으나 상용 운용 근거 부족[1, p.6] [1, p.32] | 4–6 추정; FPGA·CMM 실험은 있으나 제품·고객 배포 근거 부족[2, p.8] [2, p.9] |
| 시장 | 혼재; 런타임 직접 지원은 긍정, 개별 고객 도입은 근거 부족[W6] [W12] | 혼재; CXL 일반 신호는 있으나 ITME 개별 채택은 근거 부족[W2] [W15] |
| 이해관계자 | 혼재; 구현 보고와 엔진별 경험 차이가 병존[W23] [W11] | 우려; CXL은 HBM 보완 계층이라는 간접 제약 신호[W14] |
| 도메인 | D3 적합, D1·D4·D6·D7 조건부, D2·D5 근거 부족 | D1·D2·D3·D6·D7 조건부, D4 근거 부족, D5 제약 |

#### 관점 간 일치

- MLA는 구조적 KV cache 절감 근거와 D3 적합 판정이 일치한다. 단, 저자 보고 수치는 운영 환경의 독립 검증값이 아니다[1, p.7] [1, p.31].
- MLA의 런타임 기능 지원과 개발자 측 최적화 보고는 생태계 활동이 존재한다는 점에서 일치하지만, 고객 도입을 입증하지는 않는다[W6] [W12] [W23].
- ITME는 프로토타입·시스템 실험 근거와 CXL·RDMA·프리페치 의존성이 함께 확인되므로, TRL 4–6 및 도메인 조건부 평가가 같은 방향을 보인다[2, p.8] [2, p.9] [2, p.2].

#### MLA 관점 상충

| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| 생태계 지원과 상용 운영 | 시장: vLLM·SGLang의 MLA 직접 지원은 긍정[W6] [W12] | TRL: 실제 서비스 운영·채택은 근거 부족 | 기능 지원은 구현 가능성을 보이지만 특정 모델의 고객 운영을 입증하지 않음 |
| KV 절감과 서비스 품질 | D3: 적합[1, p.31] | D2·D5: 근거 부족 | 메모리 절감은 비교 실험이 있으나 TTFT·장기 동시성 안정성은 직접 측정되지 않음 |

#### ITME 관점 상충

| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| CXL 시장 신호와 ITME 성숙도 | 시장: CXL 일반 배포·pooling 신호로 혼재[W2] [W15] | TRL: ITME 개별 상용 운용은 근거 부족 | CXL 일반의 확산은 ITME 자체 제품화·고객 배포를 보장하지 않음 |
| 처리량과 장기 멀티턴 안정성 | D2: 조건부; 1.80배 처리량 보고[2, p.2] | D5: 제약; I/O 경합과 재계산 가능성[2, p.10] | 기준선 대비 처리량 이득과 장기 상태 누적 시 자원 경합은 서로 다른 조건의 관찰값 |
| 메모리 확장과 HBM 관계 | 도메인: 계층 확장 가능성은 조건부[2, p.2] | 이해관계자: CXL은 HBM 비대체 보완 계층[W14] | ITME 목표는 HBM 대체가 아니라 계층 확장이지만, 원격 계층의 속도 특성이 적용 범위를 제한 |

#### 두 접근의 관계

MLA는 모델 내부에서 생성되는 KV 상태의 크기를 줄이고, ITME는 GPU·호스트·SSD 사이에서 남은 상태와 가중치를 저장·이동하는 계층을 확장한다[1, p.7] [2, p.2]. 따라서 두 기술은 동일 단위의 성능 경쟁 대상으로 보기보다 보완 가능한 계층별 접근으로 해석한다. 다만 두 기술의 결합 성능, 호환성, 비용 효과를 직접 검증한 공개 근거는 부족하다.

## 6. 시사점
#### 관점별 분기

MLA는 모델 구조 수준의 KV 절감 근거가 강하지만, 실제 서비스 지연시간·운영 안정성·TCO 검증이 부족하다. ITME는 시스템 계층의 처리량·용량 확장 가능성을 보이나, 추가 하드웨어·프리페치·RDMA 요건과 장기 I/O 경합을 함께 검증해야 한다[1, p.31] [2, p.2] [2, p.10].

| 이해관계자 | MLA 도입 전 확인 과제 | ITME 도입 전 확인 과제 |
|---|---|---|
| 클라우드 사업자 | 모델별 MLA 런타임 지원, TTFT·p95/p99, 멀티테넌트 동시성, GPU당 수용 요청 수 | CXL·RDMA 토폴로지, 장애·복구, 장기 멀티턴 I/O 경합, 계층별 용량·SLA |
| 모델 개발사 | MHA 대비 재학습·가중치 호환성, 장문맥 품질의 MLA 단독 기여 | 모델 가중치·prefix KV 접근 패턴의 예측 가능성, cache miss 시 재계산 영향 |
| 메모리·HW 벤더 | MLA용 backend·kernel 호환성 및 성능 재현 | CXL-hybrid memory·SSD·NIC·프리페처 통합, 읽기 우선 scheduling 검증 |
| 투자자 | MLA 자체 고객 도입, 유료화·운영 지표, DeepSeek 계열과 MLA의 범위 구분 | ITME 제품 정의·출시·고객 레퍼런스, CXL 일반 성장과 ITME 매출의 구분 |

두 경우 모두 동일 모델, 문맥 길이, 동시성, 품질 목표, SLA 및 비용 경계에서 비교 실험을 설계해야 한다. 특히 논문 저자 보고 수치를 실제 운영 효과로 대체하지 않아야 한다.

## 7. 한계점
**자료 기준 시점**: 웹 자료 검색·검증일 2026-09-22 (Tavily 검색 결과 기준). 이후 공개된 발표·제품 정보는 반영되지 않았다.

#### 공개 정보 및 판정 범위

모든 TRL 판정은 논문, 특허, 상용 발표 등 **공개 정보만을 기반으로 한 추정**이다. MLA와 ITME 모두 공식 TRL 등급, 제품 수준 신뢰성 검증, 고객별 실제 운영 근거가 충분하지 않다. 논문 저자 보고 성능은 독립적으로 검증된 운영 사실이 아니라 저자 보고 기준으로 취급했다.

기술 발표·논문 공개와 실제 채택 사이에는 시차가 있을 수 있다. 반대로 공개 자료에서 채택 사례를 찾지 못한 사실은 채택 부재의 증명이 아니다. 특히 TRL 4–6 구간의 내부 프로토타입, 공급망 검증, 고객 평가, 장애·복구, 다중 테넌트 SLA, 비용 자료는 비공개일 수 있어 공개 자료만으로 확인하기 어렵다.

#### 근거 공백

- MLA: TTFT, p95/p99 지연시간, 동시 멀티턴 안정성, 기존 MHA 모델의 전환 절차, 동등 워크로드 TCO가 부족하다.
- ITME: 절대 TTFT·token latency, 장기 턴별 성능·hit rate, 품질·정보 보존, vLLM 외 런타임 호환성, CXL·RDMA 포함 TCO가 부족하다.
- 시장·이해관계자: MLA·ITME 각각의 실명 고객 도입과 운영 책임자 발언이 부족하다. CXL 일반 자료와 ITME 개별 기술 자료, DeepSeek 모델 자료와 MLA 기능 자료를 구분했다.

#### 확증편향 방지 조치와 남은 한계

| 적용 조치 | 수행 내용 | 남은 한계 |
|---|---|---|
| HW 베이스라인 교차 확인 | CXL-PNM 논문을 사용해 ITME와 다른 CXL 접근이 KV 관리·attention 연산 오프로드를 택할 수 있음을 확인했다[4, p.2] | InfiniGen의 메커니즘·측정치·한계에 대한 충분한 발췌는 없었다. |
| 동일 형식 병기 | MLA와 ITME를 쟁점·근거·판정·한계 형식으로 함께 서술했다 | 소프트웨어 압축과 하드웨어 계층 확장은 동일 수치로 환산할 수 없다. |
| 종합 단계의 증거 제한 | 종합 의견은 이미 검증된 Evidence만 사용하고 신규 검색을 추가하지 않았다 | 원자료의 실험 조건·웹 자료의 방법론 품질 차이가 남는다. |

따라서 본 평가는 순위나 단일 추천이 아니라, 공개 근거가 뒷받침하는 장점·제약·확인 과제를 제시하는 범위로 한정한다.

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
- [4] Kim, D., Lee, M., Kim, J., Kwon, H., Jeong, H., Park, S.-S., et al. (Hanyang University, Samsung Electronics) (2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits. arXiv preprint arXiv:2511.00321, p.2.
- [W6] docs.vllm.ai (게시일 미확인). mla_attention - vLLM Documentation. docs.vllm.ai, https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention
- [W12] sgl-project-sglang-93.mintlify.app (게시일 미확인). DeepSeek Models - SGLang. sgl-project-sglang-93.mintlify.app, https://sgl-project-sglang-93.mintlify.app/models/deepseek
- [W14] chosun.com (게시일 미확인). CXL Can't Replace HBM in AI Era. chosun.com, https://www.chosun.com/english/industry-en/2026/09/17/DPBT3YGLRJEXVHCBEOSVQ3GZNI
- [W2] computeexpresslink.org (게시일 미확인). Breaking Boundaries in Memory: Highlights from AI Infra Summit and SDC 2025 - Compute Express Link. computeexpresslink.org, https://computeexpresslink.org/blog/breaking-boundaries-in-memory-highlights-from-ai-infra-summit-and-sdc-2025-4198
- [W22] precedenceresearch.com (Wed, 09 Sep 2026 00:00:00 GMT). AI Inference-as-a-Service Market Companies, Size & Trends 2026 .... precedenceresearch.com, https://www.precedenceresearch.com/ai-inference-as-a-service-market
- [W5] docs.dynatrace.com (Fri, 28 Aug 2026 00:00:00 GMT). AI Observability integrations — Dynatrace Docs. docs.dynatrace.com, https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/integrations
- [W9] github.com (게시일 미확인). vllm/vllm/model_executor/layers/attention/mla_attention.py at main. github.com, https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py
- [W4] dataintelo.com (게시일 미확인). CXL Memory Pooling Software Market. dataintelo.com, https://dataintelo.com/report/cxl-memory-pooling-software-market
- [W15] forbes.com (Sat, 18 Jul 2026 00:00:00 GMT). AI Storage & Memory From Backblaze, CoreWeave, Panmnesia, Vast And Cloudera - Forbes. forbes.com, https://www.forbes.com/sites/tomcoughlin/2026/07/18/ai-storage--memory-from-backblaze-coreweave-panmnesia-vast-and-cloudera/
- [W1] arxiv.org (게시일 미확인). Disaggregated LLM Serving withCXL Shared Memory KV Cache at .... arxiv.org, https://arxiv.org/html/2512.18194v1
- [W23] redhat.com (게시일 미확인). Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM. redhat.com, https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm
- [W11] news.ycombinator.com (게시일 미확인). DeepSeek Open Source FlashMLA – MLA Decoding Kernel for .... news.ycombinator.com, https://news.ycombinator.com/item?id=43155023
- [W17] guinnessgi.com (게시일 미확인). How has DeepSeek affected the AI Market for investors?. guinnessgi.com, https://www.guinnessgi.com/insights/articles/how-has-deepseek-affected-ai-market-investors
- [W10] news.sbs.co.kr (게시일 미확인). AI Market: HBM's Dominance Expected to Continue... 'CXL .... news.sbs.co.kr, https://news.sbs.co.kr/english/article.do?news_id=N1008757342
- [D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 (A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.
