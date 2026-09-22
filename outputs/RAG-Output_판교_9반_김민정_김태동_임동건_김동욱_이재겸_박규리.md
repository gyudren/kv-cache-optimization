# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-09-22

## SUMMARY
- **MLA**는 공개 논문에 설계 통합·비교 벤치마크가 있어 개별 기술 TRL **4–6**으로 추정되며, 계열의 상용 운용 TRL은 **근거 부족**이다. KV cache 절감은 적합 신호이나 서비스 지연·운영 안정성은 근거가 제한적이다.[1, p.6] [1, p.15] [1, p.32]
- **ITME**도 FPGA·CMM 기반 실험 근거로 개별 기술 TRL **4–6**으로 추정되며, CXL 메모리 계열의 상용 운용 TRL은 **근거 부족**이다.[2, p.8] [2, p.9] [2, p.11]
- 시장은 MLA 런타임 구현 지원과 CXL 계열 성장 신호가 있으나, 두 기술 모두 개별 고객 도입·상용 운용 근거가 부족해 **혼재**다.[W6] [W9] [W12] [W2]
- 이해관계자 평가는 MLA가 혼재, ITME가 우려로 엇갈린다. ITME 우려는 CXL 일반의 HBM 비대체·속도 제약에 관한 간접 발언이며 ITME 직접 평가가 아니다.[W14] [W10]
- 두 접근은 각각 모델 내부 KV 표현과 시스템 메모리 계층을 다룬다. 결합 성능·호환성·비용 효과는 **근거 부족**이다.

## 1. 분석 배경
#### 문제 정의
- KV cache는 생성 추론에서 이전 토큰의 key·value 상태를 저장해 재사용하는 메모리 상태다. 표준 MHA는 추론 중 모든 key와 value를 캐시하며, 이 부담은 최대 batch 크기와 시퀀스 길이를 제한하는 병목으로 설명된다.[1, p.7]
- 본 평가의 도메인은 **데이터센터·클라우드 기반 장문맥 LLM 서빙**이다. 대규모 동시 요청, 비용 민감성, 문맥 길이 민감성이 겹쳐 KV cache 문제가 크게 드러나는 조건으로 설정했다.[D]
- 설계 기준상 KV cache는 레이어 수, attention head 수, 문맥 길이, 동시 사용자·batch, 저장 정밀도에 비례해 증가한다. HBM 점유 증가는 동시 요청·최대 batch 감소, 장문맥 제한, GPU 추가 비용, KV 이동·로딩 지연, 폐기 후 재계산 비용으로 이어질 수 있다.[D]

#### 비교 목적
- 분석 질문은 동일한 KV cache 병목에 대한 소프트웨어 압축과 하드웨어 메모리 확장이 TRL, 시장성, 이해관계자, 도메인 적용성에서 어떻게 달리 평가되는지다.[D]
- 두 기술은 작동 계층, 기준선, 실험 구성이 달라 동일 단위의 성능 수치로 순위를 정할 수 없다. 따라서 본 보고서는 우열이나 단일 추천 대신 관점별 장점, 제약, 근거 수준 및 도입 전 검증 과제를 비교한다.[D]

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB(≈74.5GiB) 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 약 3.4% |
| 128K | 40 GiB | 320 GiB | 약 53.7% |
| 1M | 320 GiB | 2,560 GiB | 약 429.5% |

산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. 요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). HBM 대비 비율은 80GB = 80×10⁹B ≈ 74.5GiB를 분모로 계산했다(설계서 A-2의 3%·50%·400%는 GiB/GB 단위를 혼용한 근사치라 여기서 바로잡음) [D].

## 2. 기술 선정
#### 선정 방식
- 기술 선정은 **2안(Human 기반, 조 토의 후 직접 선정)**으로 수행했다.[D]
- 평가 기준은 TRL, 시장성, 이해관계자, 데이터센터·클라우드 장문맥 LLM 서빙 적용성(D1–D7)이며, 소프트웨어 추론 최적화 시장과 하드웨어 CXL 시장의 규모 수치를 직접 우열 근거로 사용하지 않는다.[D]

#### DeepSeek-V2 MLA 선정 이유
- MLA는 key·value를 저랭크 잠재 벡터로 공동 압축하여 추론 시 KV cache 병목을 줄이도록 attention 구조를 재설계한 접근이다.[1, p.6] [1, p.7]
- 모델 구조·가중치에 결부되어 기존 MHA 모델에 사후 적용하기 어렵다는 제약이 예상되지만, 기술 성숙도, 상용화 신호, 생태계, 데이터센터 적용성을 함께 검토하기 위해 선정했다.[D]

#### ITME 선정 이유
- ITME는 CXL-hybrid memory를 원격 확장 계층으로 사용하고, SSD에서 내부 DRAM cache로의 프리페치 및 RDMA·호스트 메모리를 경유하는 DMA 파이프라인을 제안한다.[2, p.2]
- 모델 가중치와 장문맥 KV cache를 계층적으로 다루는 하드웨어·시스템 접근을 검토하기 위해 선정했다.[D]
- KIVI, TurboQuant, InfiniGen, CXL-PNM은 본 보고서의 선정 기술이 아니다. 이 중 CXL-PNM은 KV cache 관리와 attention 계산을 CXL-PNM에 오프로드하는 별도 접근으로 기술되어 ITME의 단순 대체 근거로 사용하지 않았다.[4, p.2]

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
| 진영 | 모델·소프트웨어 아키텍처 | 하드웨어·시스템 메모리 아키텍처 |
| 작동 계층 | attention 내부의 KV 표현 | GPU·호스트·CXL-hybrid memory·SSD 사이의 계층 |
| 핵심 접근 | key·value를 저랭크 잠재 벡터로 공동 압축하고 추론 시 이를 캐시한다.[1, p.6] [1, p.7] | 예측 가능한 모델 가중치·장문맥 KV cache를 원격 확장 계층에 두고 프리페치와 연산을 중첩한다.[2, p.2] |
| 저자 보고 효과 | DeepSeek-67B 대비 KV cache 93.3% 감소 및 최대 generation throughput 5.76배 향상.[1, p.1] | NVMe-oF 기반 분리형 스토리지 기준선 대비 처리량 1.80배 향상.[2, p.2] |
| 전제조건 | MLA 구조와 이에 맞는 모델·가중치 및 KV 처리 경로가 필요하다.[1, p.6] [1, p.8] | CXL-hybrid memory, NVMe SSD, 프리페치, RDMA·호스트 메모리 데이터 경로가 필요하다.[2, p.2] |
| 한계 | 논문 발췌에는 TTFT, p95/p99 지연시간, 멀티턴 운영 안정성 및 TCO의 직접 근거가 없다. | 장기 멀티턴에서 read/write 트래픽이 내부 I/O 경합을 유발할 수 있으며, KV miss는 GPU 재계산으로 처리한다.[2, p.10] [2, p.7] |

#### MLA
- **사실:** MLA는 key·value 공동 압축을 통해 전체 key·value 대신 압축 latent vector를 캐시하도록 설계됐다. MHA의 무거운 KV cache가 배포 시 최대 batch와 시퀀스 길이를 제한한다는 문제를 직접 겨냥한다.[1, p.7]
- **저자 보고:** DeepSeek-V2의 KV cache 감소와 throughput 수치는 MLA 외 FP8 파라미터 및 평균 6-bit KV cache 양자화가 함께 적용된 서비스 조건의 결과다.[1, p.16]
- **한계:** 이 결과만으로 MLA 단독의 지연시간, 비용 또는 운영 안정성 효과를 분리할 수 없다.

#### ITME
- **사실:** ITME는 T3.5 CXL-hybrid memory에 SSD-backed 용량과 DRAM cache를 결합하고, SSD→내부 DRAM→RDMA·호스트 메모리→GPU 이동을 계산과 겹치는 구조를 제시한다.[2, p.2]
- **저자 보고:** ITME는 Llama-3.1 8B·70B와 ShareGPT 기반 워크로드를 사용했고, 256개 동시 대화 및 35-turn KV cache 스트레스 시험을 포함했다.[2, p.9]
- **한계:** 성능은 예측 가능한 접근 패턴, 버퍼 크기, 읽기 우선 스케줄링에 의존한다. 가중치와 KV cache의 동시 전송은 성능 저하를 일으킬 수 있어 효과적 스케줄링이 필요하다.[2, p.10]

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | 개별 기술 TRL: 4–6 (판정). 논문은 DeepSeek-V2에 MLA를 설계·통합했고, MLA-MHA 비교 및 모델 벤치마크를 제시한다[1, p.6] [1, p.15] [1, p.32]. 이는 구현 및 실험적 검증 근거이지만, 제공 발췌에는 실제 서비스 운영·상용 채택 근거가 없다. 계열 TRL: 근거 부족. MLA가 속한 attention/KV-cache 최적화 계열 전체의 제품 배포 또는 상용 운용 성숙도를 판정할 직접 근거가 제공되지 않았다. | 사실: MLA는 저랭크 key-value 공동 압축으로 추론 시 KV-cache 병목을 완화하도록 설계되었다[1, p.6]. 추론 시 캐시 대상은 압축 latent vector c_t^KV이며, 캐시 원소 수는 층 수 l에 대해 l·d_c로 설명된다[1, p.7]. 저자 보고 기준 DeepSeek-V2는 DeepSeek-67B 대비 KV cache 93.3% 감소 및 최대 generation throughput 5.76배 향상을 보고한다[1, p.1]. MLA-MHA hard benchmark 비교와 공통 평가 설정의 모델 벤치마크가 제시된다[1, p.15] [1, p.32]. 해석: 아키텍처 통합 및 벤치마크 검증은 TRL 4–6 정의의 ‘통합 구현·프로토타입·유사 운용 환경 시연’에 부합하는 증거이나, 논문만으로 실제 운용을 확인할 수 없다. 판정: MLA 자체는 TRL 4–6으로 보수적으로 추정한다. MLA 계열 전체는 제공 자료만으로 판정하지 않는다. |
| ITME | 개별 기술 TRL: 4–6 (판정). 논문은 FPGA 기능 프로토타입과 CMM 기반 평가 플랫폼을 기술하고, ShareGPT 256개 동시 대화 및 35-turn 실험을 포함한 성능 검증을 보고한다[2, p.8] [2, p.9] [2, p.11]. 다만 제공 발췌에는 ITME 자체의 제품 출시, 고객 배포 또는 상용 서비스 운용 근거가 없다. 계열 TRL: 근거 부족. 논문은 CXL 기반 메모리 확장·풀링이 현대 데이터센터의 핵심 enabling technology로 부상했다고 서술하지만[2, p.11], CXL 메모리 모듈 또는 CXL 계열의 실제 제품·상용 운용을 입증하는 허용된 직접 근거는 제공되지 않았다. | 사실: ITME는 CXL-hybrid memory를 T3.5 계층으로 두고, SSD→내부 DRAM cache→RDMA/host CPU memory의 프리페치와 GPU 연산 중첩을 설명한다[2, p.2]. FPGA 프로토타입은 Intel Agilex 7 I-Series FPGA, 32GB DDR4 DRAM cache, PCIe Gen5 NVMe SSD 2개로 구성되었고, CMM 기반 플랫폼은 약 22GB/s 피크 읽기 처리량으로 기술된다[2, p.8]. 저자 보고 기준 ITME는 NVMe-oF 기반 분리형 스토리지 기준선 대비 1.80배 처리량 향상을 보였으며[2, p.2], FPGA ITME를 포함한 Llama-3.1 8B/70B 비교와 ShareGPT 256개 동시 대화·35-turn workload 평가가 제시된다[2, p.9] [2, p.11]. 해석: 하드웨어 프로토타입과 시스템 실험은 TRL 4–6의 구현·검증 근거다. 그러나 이것은 ITME 개별 아키텍처의 실제 상용 운용이나 CXL 제품군 일반의 상용 성숙도를 뜻하지 않는다. 판정: ITME 자체는 TRL 4–6으로 보수적으로 추정한다. CXL 메모리 계열의 TRL은 제공 발췌만으로는 근거 부족이다. |

#### 판정 범위
- 아래 TRL은 논문, 공개 구현·발표 자료에 한정한 **공개 정보 기반 추정**이다. TRL 4–6은 구현·프로토타입·실험 검증 신호, TRL 7–9는 제품 출시 또는 상용 서비스 적용 발표를 요구하는 설계 기준을 적용했다.[D]

#### DeepSeek-V2 MLA: 개별 기술 TRL 4–6
- **사실:** DeepSeek-V2 논문은 MLA를 저랭크 KV 공동 압축 attention으로 설계·통합하고, 공통 평가 설정의 모델 벤치마크 및 MLA–MHA hard benchmark 비교를 제시한다.[1, p.6] [1, p.15] [1, p.32]
- **해석:** 구조 통합과 비교 실험은 구현·실험 검증 근거이므로 개별 MLA를 TRL 4–6으로 보수 추정하는 근거가 된다.
- **범위 제한:** DeepSeek-V2 MLA의 실제 고객 배포·상용 서비스 운영을 직접 입증하는 제공 근거는 없다. 또한 MLA가 속한 attention/KV-cache 최적화 **계열의 상용 운용 TRL은 근거 부족**이다.
- 다만 vLLM은 `MLAAttention`과 MLA backend를, SGLang은 복수 MLA backend와 기본 활성화된 MLA 최적화를 명시한다. 이는 **MLA 기능 자체의 런타임 구현 생태계 신호**이지, MLA 계열 전체의 제품 배포·상용 운용 TRL 판정 근거는 아니다.[W6] [W12]

#### ITME: 개별 기술 TRL 4–6
- **사실:** ITME 논문은 Intel Agilex 7 I-Series FPGA 기반 기능 프로토타입, DRAM cache와 PCIe Gen5 NVMe SSD를 결합한 구성을 설명하며, CMM 기반 플랫폼의 약 22 GB/s 피크 읽기 처리량을 기술한다.[2, p.8]
- **사실:** 256개 동시 대화·35-turn 시험과 FPGA ITME를 포함한 Llama-3.1 8B·70B 비교가 제시된다.[2, p.9] [2, p.11]
- **해석:** 프로토타입과 시스템 실험은 개별 ITME의 TRL 4–6 추정을 뒷받침한다. 그러나 ITME 자체의 제품 출시, 고객 배포 또는 상용 서비스 운용은 제공 근거에서 확인되지 않는다.
- **계열 구분:** 논문은 CXL 기반 메모리 확장·풀링의 가능성을 논의하지만, 상용 CXL 하드웨어의 제한된 가용성 때문에 시뮬레이션·에뮬레이션에 의존한 연구도 있었다고 서술한다.[2, p.11] 따라서 CXL 메모리 모듈 또는 CXL 계열 일반의 상용 운용 TRL은 **근거 부족**이며, ITME 개별 TRL과 혼동해서는 안 된다.

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 긍정 | 혼재 |
| 상용화·채택 현황 | 혼재 | 혼재 |
| 생태계 지지 | 긍정 | 혼재 |
| 종합 | 혼재 | 혼재 |
| 근거 | [W22] [W5] [W9] [W6] [W12] | [W4] [W15] [W2] [W1] |

#### DeepSeek-V2 MLA: 종합 혼재
- **시장 규모·성장성(긍정):** AI inference-as-a-service 자료는 생성형 AI·LLM 사용 증가, 저지연·확장형 추론 수요 및 비용 효율적 하드웨어·소프트웨어 최적화를 성장 요인으로 든다. 이는 MLA의 수요 환경에 대한 간접 긍정 신호이며 MLA 자체 시장 규모나 매출은 아니다.[W22]
- **상용화·채택(혼재):** Dynatrace 문서는 Amazon Bedrock의 DeepSeek 모델 R1·V3 관측성을 명시하지만, 이는 DeepSeek-V2 MLA 기능 자체의 고객 채택 증거가 아니다.[W5]
- **생태계 지지(긍정):** vLLM의 MLA 구현은 단일 latent vector KV cache 표현과 MLA backend를 포함하며, 저장소 소스에도 MLA attention 코드가 확인된다.[W6] [W9] SGLang은 DeepSeek 계열 MLA와 다수 MLA backend, 기본 활성화된 MLA 최적화를 문서화한다.[W12]
- **판정:** MLA 기능 자체의 런타임 지원은 직접 신호이나, DeepSeek-V2 MLA의 고객명·도입 시점·운영 상태를 확인하는 자료는 없어 종합은 혼재다.

#### ITME: 종합 혼재
- **시장 규모·성장성(혼재):** CXL memory pooling software 시장 자료는 AI 워크로드·데이터센터 현대화·CXL 3.0 채택을 성장 동인으로 제시한다. 그러나 이는 ITME가 아닌 좁은 정의의 메모리 풀링 소프트웨어 시장 수치이며, ITME 수요·매출 전망으로 일반화할 수 없다.[W4]
- **상용화·채택(혼재):** Forbes 보도는 Meta가 CXL 메모리 확장을 프로덕션 서비스에 사용했다고 전하지만, ITME 사용 여부는 확인하지 않는다.[W15]
- **생태계 지지(혼재):** CXL Consortium은 실제 CXL 배포 증가와 메모리 풀링·공유를 주요 채택 요인으로 언급한다.[W2] 별도 연구 자료는 Dynamo v0.5.0과 vLLM v0.10.1.1 통합을 설명하지만, 이 역시 ITME 공식 통합이나 시장 채택 근거는 아니다.[W1]
- **판정:** CXL 일반의 시장·표준·배포 신호는 존재하지만 ITME 자체의 공식 제품 정의, 출시, 고객 도입, 런타임 지원은 근거 부족이므로 종합은 혼재다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 근거 부족 | 우려 |
| 도입 기업·개발자 | 혼재 | 근거 부족 |
| 투자 업계 | 혼재 | 근거 부족 |
| 종합 | 혼재 | 우려 |
| 근거 | [W23] [W11] [W17] | [W14] [W10] |

#### DeepSeek-V2 MLA: 종합 혼재
- **경쟁 진영(근거 부족):** 제공 자료에는 경쟁 기술·기업 관계자가 MLA 자체의 한계, 대체 또는 병행 가능성을 직접 언급한 근거가 없다.
- **도입 기업·개발자(혼재):** Red Hat 블로그는 vLLM의 MLA 지원과 장문맥 생성 워크로드에서의 처리량·메모리 효율 개선을 보고한다. 이는 제공자 측 보고이며 MLA 자체의 일반화된 고객 운영 성과는 아니다.[W23] 반면 Hacker News의 한 사용자는 두 p5 H100 노드의 개인 비교에서 SGLang의 DeepSeek 지원이 vLLM보다 더 빠르고 좋았다고 썼다. 이는 특정 환경의 커뮤니티 경험으로 일반화할 수 없다.[W11]
- **투자 업계(혼재):** Guinness Global Investors는 DeepSeek의 저비용 모델 개발 가능성이 경쟁을 촉진할 수 있다는 관측과 hyperscaler 인프라 지출 지속을 함께 언급한다. 이는 MLA 단독이 아니라 DeepSeek 계열·AI 인프라에 관한 간접 견해다.[W17]

#### ITME: 종합 우려
- **경쟁 진영(우려):** Chosun 보도는 Intel SoC 아키텍처 관계자의 발언으로, CXL 메모리 결합은 유용할 수 있으나 HBM을 대체하지 않고 보조 저장계층을 보완하며 GPU의 CXL 경유 접근 속도는 HBM만큼 빠르지 않다는 견해를 전한다.[W14] 이는 ITME 직접 평가가 아니라 CXL 일반에 대한 **간접 우려**다.
- SBS 보도도 CXL을 HBM의 대안이 아닌 저장장치 보조 수단으로 다루고, 인용된 관계자의 비활성 데이터 활용 및 특화 사례에 대한 견해를 전한다.[W10] 따라서 두 자료는 ITME의 실측 결과나 고객 도입성을 반증하지 않는다.
- **도입 기업·개발자 및 투자 업계(각 근거 부족):** ITME를 실제 도입한 운영자·개발자 또는 식별 가능한 투자자의 직접 반응은 제공 자료에서 확인되지 않는다.
- **판정 주의:** 신호표의 종합 우려는 경쟁 진영의 CXL 일반 간접 우려를 보존한 결과다. 이를 ITME 전체 이해관계자의 확정적 부정 평가로 확대해서는 안 된다.

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
- **D1 조건부:** DeepSeek-V2는 128K context를 지원하며, 저자 보고 서비스 조건에서 더 큰 batch를 처리할 수 있다고 제시한다. 단, 이 조건은 MLA와 FP8 파라미터·평균 6-bit KV cache 양자화가 함께 적용됐고 GPU 용량별 최대 동시 요청 수는 제시되지 않았다.[1, p.1] [1, p.16]
- **D2 근거 부족:** 저자 보고 기준 단일 8×H800 노드에서 generation throughput 50K tokens/s 초과가 제시되어 처리량 신호는 존재한다.[1, p.16] 그러나 MLA 외 최적화가 결합된 결과이고, TTFT·절대 또는 꼬리 지연시간·메모리 압박 시 지연시간이 없어 D2 전체 판정은 근거 부족을 유지한다.
- **D3 적합:** MLA는 압축 latent vector와 decoupled key를 캐시하며, 저자 보고 MLA의 KV cache는 MHA 대비 소형 MoE에서 14%, 대형 MoE에서 4% 수준이다.[1, p.8] [1, p.31]
- **D4 조건부:** 저자 보고 기준 32K 추가 학습 뒤 128K context 및 NIAH에서 견고한 성능을 제시하지만, 장문맥 품질을 MLA 단독 효과로 분리한 결과는 아니다.[1, p.13]
- **D5 근거 부족:** 멀티턴 동시 요청, 캐시 miss, 자원 경합, 지연시간 분산의 직접 실험은 제공되지 않았다.
- **D6 조건부:** MLA는 MHA와 다른 attention 구조와 KV cache 형식을 사용한다.[1, p.6] [1, p.8] 기존 MHA 체크포인트의 변환·재학습 또는 런타임 통합 절차는 근거 부족이다.
- **D7 조건부:** KV cache 감소와 큰 batch 처리 가능성은 GPU 메모리 요구 절감 가능성을 시사하지만, 동등 워크로드당 인프라 비용·TCO 직접 비교는 없다.[1, p.1] [1, p.16]

#### ITME
- **D1 조건부:** ShareGPT 기반 256개 동시 대화·35-turn KV cache 스트레스 시험을 수행했으나, 동일 GPU 메모리에서 문맥 길이 상한 또는 최대 동시 요청 증가를 직접 비교한 자료는 없다.[2, p.9]
- **D2 조건부:** 저자 보고 기준 NVMe-oF 분리형 스토리지 기준선 대비 처리량 1.80배 향상 신호가 있다.[2, p.2] 다만 절대 TTFT·토큰 지연시간·prompt throughput은 충분히 제시되지 않았고, 성능은 메모리 계층 및 접근 패턴에 의존한다.[2, p.9]
- **D3 조건부:** ITME는 CPU-offload 기준선의 128 GB host memory 대신 CXL-hybrid memory와 일부 host staging buffer를 사용한다.[2, p.9] GPU 상주 용량 절감은 확인되지만, CXL·호스트·원격 계층의 총 자원량과 token당 KV 바이트는 근거 부족이다.
- **D4 근거 부족:** 장문맥 KV cache 배치·복원 및 재계산 정책은 설명되나 출력 품질·정확도·장문맥 정보 보존을 직접 검증한 결과는 없다.[2, p.3] [2, p.7]
- **D5 제약:** KV cache 증가에 따른 read/write 트래픽은 CXL-hybrid memory 내부 I/O 경합을 유발하고 지연 블록의 재계산을 초래할 수 있다.[2, p.10]
- **D6 조건부:** CXL-hybrid memory, RDMA, 하드웨어·사용자 수준 프리페처가 필요하며, 저자 보고 구현은 vLLM v0.17.0 기반이다.[2, p.2] [2, p.9]
- **D7 조건부:** 비용 효율적 TB-scale 확장을 목표로 제시하지만, 장비 가격·네트워크·CXL 구성·TCO의 직접 측정치는 없다.[2, p.1] [2, p.9]

## 5. 종합 의견
#### 관점 매트릭스
| 관점 | MLA | ITME |
|---|---|---|
| TRL | 개별 기술 4–6, 계열 상용 운용 TRL은 근거 부족.[1, p.6] [1, p.32] | 개별 기술 4–6, CXL 계열 상용 운용 TRL은 근거 부족.[2, p.8] [2, p.11] |
| 시장 | 혼재: MLA 런타임 지원은 직접 신호, 개별 고객 도입은 근거 부족.[W6] [W9] [W12] | 혼재: CXL 계열 신호는 있으나 ITME 개별 출시·도입은 근거 부족.[W2] [W15] |
| 이해관계자 | 혼재: 제공자 보고와 커뮤니티 환경 차이가 병존.[W23] [W11] | 우려: CXL 일반의 HBM 비대체·속도 제약은 간접 우려이며 직접 반응은 부족.[W14] [W10] |
| 도메인 | D3 적합, D1·D4·D6·D7 조건부, D2·D5 근거 부족.[1, p.8] [1, p.16] | D1·D2·D3·D6·D7 조건부, D4 근거 부족, D5 제약.[2, p.2] [2, p.9] [2, p.10] |

#### 관점 간 일치
- MLA는 모델 구조 차원의 KV cache 절감 근거가 TRL의 구현·검증 신호와 D3 적합 판정을 함께 뒷받침한다. 다만 정량 결과는 저자 보고 기준이다.[1, p.7] [1, p.31]
- ITME는 프로토타입·시스템 실험이 존재하되 CXL-hybrid memory·RDMA·프리페치에 의존한다는 점에서 TRL 및 도메인 평가가 일치한다.[2, p.2] [2, p.8]
- 두 기술 모두 개별 고객의 장기 상용 운용, SLA, 장애 복구, TCO에 관한 직접 공개 근거가 부족하다.

#### MLA: 관점 간 상충
| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| 런타임 지원과 상용 운용 | 시장: vLLM·SGLang의 MLA 기능 지원은 긍정 신호.[W6] [W9] [W12] | TRL: 개별 기술 4–6, 상용 운용은 근거 부족.[1, p.6] [1, p.32] | 기능 구현은 고객 배포·운영 신뢰성의 직접 증거가 아니다. |
| KV 절감과 서비스 성능 | D3: 적합.[1, p.8] [1, p.31] | D2·D5: 근거 부족.[1, p.16] | 처리량 신호는 있으나 TTFT, 꼬리 지연, 멀티턴 안정성 지표가 부족하다. |

#### ITME: 관점 간 상충
| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| CXL 계열 신호와 ITME 성숙도 | 시장: CXL 배포·생태계 신호로 혼재.[W2] [W15] | TRL: ITME는 4–6, 제품 운용은 근거 부족.[2, p.8] [2, p.11] | CXL 일반의 배포가 ITME 개별 아키텍처의 상용화를 뜻하지 않는다. |
| 처리량과 장기 멀티턴 안정성 | D2: 조건부; 저자 보고 1.80배 처리량 향상.[2, p.2] | D5: 제약; I/O 경합 및 재계산 가능성.[2, p.10] | 기준선 대비 이득과 장기 부하에서의 계층 내 자원 경합은 동시에 존재할 수 있다. |

#### 두 접근의 관계
- MLA는 모델 내부에서 캐시해야 할 KV 상태의 표현량을 줄이고, ITME는 GPU·호스트·CXL-hybrid memory·SSD 사이에서 상태를 저장·이동하는 계층을 다룬다.[1, p.7] [2, p.2]
- 따라서 두 기술은 동일 계층의 직접 경쟁으로 보기보다 **서로 다른 계층의 접근**으로 기술하는 것이 타당하다. 두 기술의 결합 성능, 호환성, 비용 효과는 공개 근거 부족이다.

## 6. 시사점
#### 관점별 시사점
- MLA는 메모리 효율의 구조적 근거와 런타임 구현 신호가 강하지만, 서비스 지연시간·운영 안정성·비용의 직접 검증은 약하다.[1, p.31] [W6] [W12]
- ITME는 장문맥·다중 턴 메모리 계층 확장 실험 신호가 있으나, 원격 계층의 지연·I/O 경합, 장비·통합 요구사항이 평가를 조건부로 만든다.[2, p.2] [2, p.10]
- 따라서 도입 판단은 기술 일반론이 아니라 동일 모델, 문맥 길이, 동시성, SLA 및 비용 조건에서 분리 검증해야 한다.

#### 이해관계자별 도입 전 확인 과제
| 이해관계자 | MLA 확인 과제 | ITME 확인 과제 |
|---|---|---|
| 클라우드 사업자 | TTFT, p95/p99 지연시간, 동시성별 처리량, 장애·복구 및 실제 GPU 메모리 절감 측정 | CXL·RDMA 토폴로지, 원격 계층 경합, long-tail 지연, 다중 테넌트 격리 및 장애 복구 측정 |
| 모델 개발사 | MLA 구조·가중치 요구, 기존 MHA 모델의 변환·재학습 가능성, 장문맥 품질 절제 실험 | 모델별 가중치·KV 접근 예측 가능성, miss 재계산 영향, 프리페치·스케줄링 정책 검증 |
| 메모리·HW 벤더 | MLA runtime backend의 호환성 및 kernel 성능 확인 | CXL-hybrid memory, SSD, NIC, host memory의 구성별 대역폭·경합·운영성 검증 |
| 투자자 | MLA 단독의 고객 도입, 운영 비용, 유지보수 지표 확인 | ITME의 제품 정의·출시 상태·고객 레퍼런스 및 CXL 일반과의 구분 확인 |

- 결합 도입을 검토할 경우에도 MLA의 모델·런타임 요건과 ITME의 하드웨어·네트워크 요건을 별도로 확인해야 하며, 결합 효과를 가정해서는 안 된다.

## 7. 한계점
**자료 기준 시점**: 웹 자료 검색·검증일 2026-09-22 (Tavily 검색 결과 기준). 이후 공개된 발표·제품 정보는 반영되지 않았다.

#### 공개 근거의 한계
- 모든 TRL 판정은 논문, 특허, 상용 발표 등 **공개 정보만을 기반으로 한 추정**이다. 논문·발표가 실제 채택으로 이어지기까지 시차가 있을 수 있으므로, 공개 근거 부족은 기술 또는 배포의 부재를 의미하지 않는다.
- 특히 TRL 4–6 구간의 내부 검증, 신뢰성 시험, 고객 PoC, 공급망 준비도, 운영 장애 기록은 비공개일 수 있다. 이 **비공개 정보 공백** 때문에 프로토타입·실험 근거를 제품·상용 운용 근거로 승격하지 않았다.
- MLA의 정량 결과에는 MLA 외 DeepSeekMoE, FP8 파라미터, KV cache 양자화 등 결합 요인이 있고, ITME의 결과는 특정 기준선·프리페치·메모리 계층 조건에 의존한다.[1, p.16] [2, p.2] [2, p.10]
- 시장·이해관계자 자료 중 일부는 MLA 또는 ITME 개별 기술이 아니라 DeepSeek 모델 계열이나 CXL 일반에 관한 간접 근거다. 이를 개별 기술의 제품 출시·고객 채택 근거로 해석하지 않았다.[W5] [W15] [W14]

#### 확증편향 방지 조치와 남은 한계
| 조치 | 실제 적용 방식 | 남은 한계 |
|---|---|---|
| HW 베이스라인 교차 확인 | InfiniGen·CXL-PNM을 ITME 한계 점검용 베이스라인으로 검토했다. CXL-PNM은 KV 관리·attention 계산을 PNM으로 오프로드하는 별도 접근이다.[4, p.2] | 제공 발췌에는 InfiniGen의 직접 메커니즘·측정 결과가 충분하지 않다. |
| 동일 보고 형식 | 두 기술을 쟁점·관점 A·관점 B·엇갈리는 이유의 동일 구조로 병기했다.[D] | 기술 계층과 실험 기준선이 달라 수치의 직접 비교는 불가하다. |
| 종합 단계의 증거 제한 | 종합 Agent는 이미 검증된 Evidence만 사용하고 신규 검색을 수행하지 않았다.[D] | 검증된 자료군 자체의 시간적·공개 범위 한계는 남는다. |

- 본 보고서는 순위, 최종 승자 또는 단일 추천을 제시하지 않는다.

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
- [1] DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, p.1, p.6, p.7, p.8, p.13, p.15, p.16, p.31, p.32.
- [2] Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J. (SK hynix) (2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv preprint arXiv:2606.12556, p.1, p.2, p.3, p.7, p.8, p.9, p.10, p.11.
- [4] Kim, D., Lee, M., Kim, J., Kwon, H., Jeong, H., Park, S.-S., et al. (Hanyang University, Samsung Electronics) (2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits. arXiv preprint arXiv:2511.00321, p.2.
- [W6] docs.vllm.ai (게시일 미확인). mla_attention - vLLM Documentation. docs.vllm.ai, https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention
- [W9] github.com (게시일 미확인). vllm/vllm/model_executor/layers/attention/mla_attention.py at main. github.com, https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py
- [W12] sgl-project-sglang-93.mintlify.app (게시일 미확인). DeepSeek Models - SGLang. sgl-project-sglang-93.mintlify.app, https://sgl-project-sglang-93.mintlify.app/models/deepseek
- [W2] computeexpresslink.org (게시일 미확인). Breaking Boundaries in Memory: Highlights from AI Infra Summit and SDC 2025 - Compute Express Link. computeexpresslink.org, https://computeexpresslink.org/blog/breaking-boundaries-in-memory-highlights-from-ai-infra-summit-and-sdc-2025-4198
- [W14] chosun.com (게시일 미확인). CXL Can't Replace HBM in AI Era. chosun.com, https://www.chosun.com/english/industry-en/2026/09/17/DPBT3YGLRJEXVHCBEOSVQ3GZNI
- [W10] news.sbs.co.kr (게시일 미확인). AI Market: HBM's Dominance Expected to Continue... 'CXL .... news.sbs.co.kr, https://news.sbs.co.kr/english/article.do?news_id=N1008757342
- [W22] precedenceresearch.com (Wed, 09 Sep 2026 00:00:00 GMT). AI Inference-as-a-Service Market Companies, Size & Trends 2026 .... precedenceresearch.com, https://www.precedenceresearch.com/ai-inference-as-a-service-market
- [W5] docs.dynatrace.com (Fri, 28 Aug 2026 00:00:00 GMT). AI Observability integrations — Dynatrace Docs. docs.dynatrace.com, https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/integrations
- [W4] dataintelo.com (게시일 미확인). CXL Memory Pooling Software Market. dataintelo.com, https://dataintelo.com/report/cxl-memory-pooling-software-market
- [W15] forbes.com (Sat, 18 Jul 2026 00:00:00 GMT). AI Storage & Memory From Backblaze, CoreWeave, Panmnesia, Vast And Cloudera - Forbes. forbes.com, https://www.forbes.com/sites/tomcoughlin/2026/07/18/ai-storage--memory-from-backblaze-coreweave-panmnesia-vast-and-cloudera/
- [W1] arxiv.org (게시일 미확인). Disaggregated LLM Serving withCXL Shared Memory KV Cache at .... arxiv.org, https://arxiv.org/html/2512.18194v1
- [W23] redhat.com (게시일 미확인). Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM. redhat.com, https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm
- [W11] news.ycombinator.com (게시일 미확인). DeepSeek Open Source FlashMLA – MLA Decoding Kernel for .... news.ycombinator.com, https://news.ycombinator.com/item?id=43155023
- [W17] guinnessgi.com (게시일 미확인). How has DeepSeek affected the AI Market for investors?. guinnessgi.com, https://www.guinnessgi.com/insights/articles/how-has-deepseek-affected-ai-market-investors
- [D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 (A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.
