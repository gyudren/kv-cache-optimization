# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-09-22

## SUMMARY
- **TRL(공개정보 추정):** MLA와 ITME 모두 4–6으로 본다. MLA는 DeepSeek-V2 통합·배포 조건 평가, ITME는 CXL-hybrid-memory 기반 평가라는 구현·검증 신호가 있으나, 각 기술 자체의 제품 출시·독립 상용 운영 근거는 부족하다.[1, p.16] [2, p.8]
- **관점의 엇갈림:** MLA는 KV cache 절감과 장문맥 처리의 기술·도메인 신호가 있으나 시장·이해관계자 근거는 부족하다. ITME는 인접 CXL 시장의 성장 신호가 있으나 ITME 자체의 고객 도입·생태계는 확인되지 않아 시장성은 혼재로 판단한다.[1, p.13] [W11] [W13]
- **도메인:** MLA는 메모리 효율에, ITME는 원격 계층을 통한 수용력 확장에 각각 근거가 있다. 다만 두 기술 모두 비용, 운영 안정성 또는 종단간 지연시간의 공개 검증은 제한적이다.[1, p.16] [2, p.6] [2, p.7]
- **관계:** MLA는 모델 내부 KV 표현을 줄이고, ITME는 시스템 메모리 계층을 확장한다. 동일 병목에 대한 서로 다른 계층의 접근이며, 대체재·우열 관계나 결합 효과는 공개 근거만으로 판정할 수 없다.
- **결론:** 단일 추천이나 순위는 도출하지 않는다. 도입 판단은 모델 호환성, CXL·네트워크 구성, 지연시간, 다중 턴 경합, 비용을 해당 운영환경에서 별도로 확인해야 한다.

## 1. 분석 배경
#### 설계 기준
- KV cache는 생성 단계에서 이전 토큰의 key·value를 저장하여 attention 계산을 가속하는 상태이다. 표준 MHA는 추론 중 모든 key와 value를 캐시해야 하므로, 무거운 KV cache가 배치 크기와 시퀀스 길이를 제한하는 병목이 될 수 있다.[1, p.7]
- 본 평가의 도메인은 데이터센터·클라우드 기반 장문맥 LLM 서빙이다. 설계 기준상 이 환경에서는 대규모 동시 요청, 비용 민감성, 문맥 길이 민감성이 함께 존재하며, KV cache가 HBM을 점유하면 동시 요청 수와 최대 batch의 제약, 장문맥 요청 제한, 데이터 이동·재계산 부담이 나타날 수 있다.
- 장문맥 서빙에서 KV cache 문제는 모델 내부 표현량과 GPU 밖 계층으로의 저장·이동 문제를 함께 포함한다. CPU 메모리 오프로드는 배치 크기·시퀀스 길이 제약을 완화할 수 있으나, attention 계산을 위한 대용량 KV cache 전송은 PCIe 대역폭 병목을 유발할 수 있다.[3, p.4]
- 따라서 본 보고서는 하나의 수치로 우열을 정하지 않는다. MLA는 attention 구조 계층, ITME는 원격 메모리·데이터 이동 계층에서 작동하므로, TRL·시장성·이해관계자·도메인 적용성의 근거 수준과 제약을 분리해 비교한다.

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16, 토큰당 320KiB — 설계 산출물 A-2 산식)
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 3% |
| 128K | 40 GiB | 320 GiB | 50% |
| 1M | 320 GiB | 2,560 GiB | 400% |

## 2. 기술 선정
#### 선정 방식
- **2안(Human 기반, 조 토의 후 직접 선정)**을 적용했다. 이는 조사 결과가 아니라 과제의 설계 기준이다.
- 선정 기준은 데이터센터·클라우드 장문맥 LLM 서빙의 KV cache 병목과의 직접성, 서로 다른 작동 계층의 비교 가능성, 공개 문헌에서 확인 가능한 구현·평가 근거의 존재였다.

#### DeepSeek-V2 MLA 선정
- MLA는 저랭크 key-value 공동 압축으로 추론 시 KV cache 병목을 줄이도록 설계된 attention 메커니즘이다.[1, p.6]
- 표준 MHA가 key와 value를 모두 캐시하는 것과 달리, MLA는 압축된 KV latent를 캐시하는 구조를 제시한다.[1, p.7]
- 따라서 모델 내부에서 KV 표현량 자체를 줄이는 SW·모델 아키텍처 접근의 사례로 선정했다.

#### ITME 선정
- ITME는 SSD-backed 용량을 CXL-hybrid-memory 기반 원격 메모리 서버로 제시하여, 대규모 KV cache와 가중치의 계층 배치를 다루는 시스템 접근이다.[2, p.3] [2, p.4]
- GPU 연산과 원격 계층 데이터 검색의 중첩을 통해 접근 지연을 숨기는 구성을 제시한다.[2, p.8]
- 따라서 GPU 메모리 바깥의 용량 확장과 데이터 이동을 다루는 HW·시스템 계층 사례로 선정했다.

#### 비선정 후보
- InfiniGen과 CXL-PNM은 ITME 한계의 교차 확인을 위한 베이스라인 문헌으로 활용했다. CPU 오프로드의 PCIe 병목과 CXL near-memory 처리라는 상이한 접근을 확인할 수 있다.[3, p.4] [4, p.4]
- 이들의 비선정 사유는 기술 사실의 단정이 아니라, 본 과제에서 MLA와 ITME를 대표 사례로 직접 선정한 **설계 판단**이다.

#### 표 2. 비선정 후보와 사유 (설계 산출물 A-4)
| 후보 | 진영 | 비선정 사유 | RAG 문서 활용 |
|---|---|---|---|
| InfiniGen | HW | 호스트 메모리 오프로딩으로 신규 메모리 인프라 없이 SW 관리 성격이 강함 | O (ITME 비교용 베이스라인) |
| CXL-PNM | HW | 연산까지 메모리 측으로 옮기는 확장형 접근으로, '공간 확장' 자체의 대표성은 ITME가 더 직접적 | O (ITME 비교용 베이스라인) |
| KIVI | SW | 사후 양자화의 대표 베이스라인이나 공개 채택 근거가 연구·라이브러리 수준 | X (선정 검토만) |
| TurboQuant | SW | 재학습 없이 적용 가능한 최신 양자화이나 공식 상용화 근거가 제한적 | X (선정 검토만) |

## 3. 기술 개요
| 구분 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 진영 | SW·모델 아키텍처 | HW·시스템 메모리 계층 |
| 작동 계층 | Transformer attention 내부 | 원격 CXL-hybrid-memory 및 데이터 이동 경로 |
| 핵심 접근 | 저랭크 key-value 공동 압축 | 원격 메모리 확장과 계층 간 데이터 배치·검색 |
| 저자 보고 효과 | KV cache 축소 및 배포 처리량 결과 보고 | 원격 계층 기반 처리량 결과 보고 |
| 전제조건 | MLA를 포함한 모델 아키텍처 | CXL-hybrid-memory, 원격 메모리·전송 경로 구성 |
| 한계·확인사항 | 기존 MHA 자산 전환, 지연시간·운영성 공개 근거 부족 | 장기 다중 턴 I/O 경합, 품질·비용·상용 배포 근거 부족 |

#### DeepSeek-V2 MLA
- **핵심 접근(fact):** MLA는 low-rank key-value joint compression을 이용해 추론 시 KV cache 병목을 줄이도록 설계되었다.[1, p.6] MHA에서는 key와 value 전체를 캐시해야 하지만, MLA는 압축 KV latent를 캐시하는 구조를 제시한다.[1, p.7]
- **저자 보고 효과(fact):** DeepSeek-V2는 128K 토큰 문맥 길이를 지원하며, 저자들은 DeepSeek 67B 대비 KV cache 93.3% 감소를 보고했다.[1, p.1] 서비스 배포에서는 FP8 파라미터와 평균 6비트 KV cache 양자화를 사용했으며, 처리량 결과는 이 배포 조건 전체의 결과로 해석해야 한다.[1, p.16]
- **한계·전제조건(interpretation):** MLA 효과를 기존 MHA checkpoint에 사후 적용할 수 있는지, 재학습이 필요한지, 서빙 프레임워크가 직접 지원하는지는 제공 근거만으로 확인할 수 없다. MLA 단독 효과와 DeepSeek-V2 전체 모델·양자화·하드웨어 조건의 효과도 분리되어 있지 않다.

#### ITME
- **핵심 접근(fact):** ITME는 SSD-backed 용량을 직접 접근 가능한 메모리 확장으로 제시하며, CXL-hybrid-memory 기반 원격 메모리 서버로 동작하도록 설명한다.[2, p.3] 하드웨어 관리 DRAM cache로 backend read latency를 완화하는 hybrid 설계를 제시한다.[2, p.4]
- **저자 보고 효과(fact):** 저자들은 NVMe-oF 기반 분리형 스토리지 기준선 대비 1.80배 처리량을 보고했다.[2, p.2] 원격 CXL-hybrid-memory에서 가중치와 prefix KV cache를 검색하는 작업을 GPU 계산과 겹쳐 지연을 은닉하는 방식을 제시한다.[2, p.8]
- **한계·전제조건(fact 및 interpretation):** ITME는 원격 계층 접근, staging, prefetch 정책에 의존한다. 읽기·쓰기 경합은 유효 읽기 대역폭을 낮추고 예측하기 어려운 latency excursion을 만들 수 있어, 논문은 읽기 우선 I/O 스케줄링을 제시한다.[2, p.6] 구체적인 고객 운영, 제품 출하, 장문맥 품질, 총소유비용은 공개 근거 부족이다.

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | technology_trl: 4-6 (추정). 근거: MLA는 DeepSeek-V2에 통합되어 구현되었고, 저자 보고 기준 서비스 배포 설정… | fact: MLA는 low-rank key-value joint compression으로 추론 시 KV-cache 병목을 제거하도록 설계되었다([1, p.6]). 추론 중 압축 KV latent만 캐시하며, key/value up-projection은 query/output projection에 흡수될 수 있다([1, p.7]). 저자 보고 기준 DeepSeek-V2는 DeepSeek 67B 대비 KV cache 93.3% 감소를 보고했다([1, p.1]).… |
| ITME | technology_trl: 4-6 (추정). 근거: FPGA 기반 하드웨어 프로토타입의 기능 실현 가능성 시연과 생산급 SK hynix CM… | fact: ITME는 TB-scale LLM 워크로드를 위한 CXL-hybrid-memory 확장 구조로, GPU 서버가 RDMA를 통해 원격 계층의 모델 가중치와 KV cache에 접근하도록 한다([2, p.2]). 하드웨어 prefetcher와 사용자 수준 prefetch API를 포함하며, FPGA 기반 하드웨어 프로토타입으로 기능 가능성을 시연했다고 저자들이 보고했다([2, p.2]). 저자 보고 기준 NVMe-oF 분리형 스토리지 기준선 대비 처리량… |

#### 판정 기준과 범위
- TRL은 설계 기준에 따라 논문·특허 근거를 1–3, 코드·도구 탑재 등 구현 근거를 4–6, 제품 출시·상용 서비스 적용 발표를 7–9의 신호로 본다.
- 아래 판단은 공식 TRL 인증이 아니라 **논문, 공개 발표, 상용화 공지의 공개정보에만 기초한 추정**이다.

#### DeepSeek-V2 MLA — 4–6 추정
- **fact:** MLA는 DeepSeek-V2의 Transformer attention에 통합된 구조로 제시되며, 저자들은 서비스 배포를 위해 FP8 파라미터와 평균 6비트 KV cache 양자화를 사용했다고 보고했다.[1, p.4] [1, p.16]
- **fact:** 동일 논문은 128K 문맥 평가에서 견고한 성능을 보고한다.[1, p.13]
- **judgment:** 통합 구현과 배포 조건 평가가 문서화되어 있어 4–6으로 추정한다. 다만 MLA 아키텍처 자체의 제품 출시, 독립 고객 도입, 상용 서비스 운영을 확인하는 공개 근거가 없으므로 7–9는 판정하지 않는다.
- **계열과의 구분:** attention/KV-cache 최적화 계열 전체의 TRL은 별도 제품·운영 근거가 없어 근거 부족이다. DeepSeek 모델의 공개·사용과 MLA 기능 자체의 직접 상용 지원은 동일하지 않다.

#### ITME — 4–6 추정
- **fact:** ITME는 CXL-hybrid-memory를 통한 원격 메모리 확장 구조와, GPU 계산 및 원격 계층 검색의 중첩을 제시한다.[2, p.3] [2, p.8]
- **fact:** 저자들은 ITME의 처리량 평가 결과를 보고한다.[2, p.2]
- **judgment:** 아키텍처와 평가가 공개되어 있어 ITME 개별 기술은 4–6으로 추정한다. 그러나 ITME 정확 아키텍처의 제품 출시, 고객 배포, 상용 서비스 운영을 문서화하는 공개 근거가 부족하므로 7–9는 판정하지 않는다.
- **계열과의 구분:** CXL은 메모리 확장·풀링의 핵심 기술로 논의되지만, 상용 CXL 하드웨어의 제한된 가용성 때문에 다수 연구가 시뮬레이션·에뮬레이션에 의존했다고 서술된다.[2, p.11] 이는 CXL 일반 또는 CXL 메모리 모듈의 TRL을 ITME 개별 기술에 그대로 부여할 근거가 아니다.

#### 공개정보 공백
- MLA와 ITME 모두 TRL 4–6 이후의 비공개 통합, 고객 검증, 장애·운영 지표는 확인할 수 없다. 공개 근거 부족은 미도입의 증명이 아니다.

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 근거 부족 | 긍정 |
| 상용화·채택 현황 | 근거 부족 | 근거 부족 |
| 생태계 지지 | 근거 부족 | 근거 부족 |
| 종합 | 근거 부족 | 근거 부족 |
| 근거 | [W7] [W6] | [W11] [W13] |

#### DeepSeek-V2 MLA — 근거 부족
- **시장 규모·성장(fact):** MLA 또는 MLA가 속한 SW 추론 최적화 시장의 시장 규모, 기준연도, 전망연도, CAGR을 직접 제시하는 검증된 자료는 제공되지 않았다.
- **채택(fact):** 제공된 DeepSeek 관련 시장 보도는 기업가치·자금조달 맥락의 정보이며, MLA 기능 자체의 유료 서비스 적용, 고객 도입 또는 장문맥 서빙 채택을 식별하지 않는다.[W6] [W7]
- **생태계(judgment):** vLLM·SGLang·클라우드 서비스·공식 저장소가 MLA를 직접 지원한다는 공개 근거가 부족하다. DeepSeek 모델의 사용 또는 기업 관련 보도는 MLA 기능의 독립 지원 사례로 해석하지 않는다.
- **판정:** 시장성은 **근거 부족**이다. 필요한 확인 자료는 MLA 직접 지원 릴리스 노트, 고객·운영 사례, SW 추론 최적화 시장의 정의된 조사자료다.

#### ITME — 혼재
- **인접 시장 신호(fact):** CXL 메모리 컨트롤러 IC 시장 자료는 초기 상업적 배치와 시장 성장 전망을 다룬다.[W11] 이는 CXL 관련 시장 환경의 신호이지만 ITME 자체 시장의 수치가 아니다.
- **개별 기술 채택(fact):** ITME 관련 보도는 SK hynix의 AI 메모리 아키텍처 진전 맥락을 언급하지만, ITME의 제품 출하, 고객명, 운영 규모, 실제 서비스 적용을 직접 확인하지 않는다.[W13]
- **생태계(judgment):** ITME를 직접 지원하는 추론 엔진, 클라우드, 서버 OEM, CXL 장치 또는 표준화 문서의 공개 근거가 부족하다.
- **판정:** 인접 CXL 시장의 긍정 신호와 ITME 개별 상용화 근거 부족이 공존하므로 시장성은 **혼재**다. CXL 일반 시장 규모로 ITME의 수요·매출·도입을 대체할 수 없다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 근거 부족 | 근거 부족 |
| 도입 기업·개발자 | 근거 부족 | 근거 부족 |
| 투자 업계 | 근거 부족 | 근거 부족 |
| 종합 | 근거 부족 | 근거 부족 |
| 근거 | [W2] | [W5] [W4] [W1] |

#### 평가 범위
- S1은 경쟁 기술·기업, S2는 개발자·도입·운영자, S3는 투자자·애널리스트의 기술 **직접 발언 또는 운영 경험**을 요구한다.
- 제공된 웹 자료 중 일부는 기술 일반 설명 또는 제3자 논평이므로, 직접 이해관계자 평가와 구분한다.

#### DeepSeek-V2 MLA — 근거 부족
- **S1 경쟁 기술·기업:** MLA를 직접 대상으로 장점·제약·대체 관계를 밝힌 경쟁사 또는 대체기술 제공자의 귀속 발언은 확인되지 않았다.
- **S2 개발자·도입자:** PyTorch 한국 사용자 모임 자료는 MLA의 KV cache 절감 구조와 사용 방법을 소개하지만, 특정 기업의 실제 운영 도입 경험이나 재현 가능한 운영 성과는 아니다.[W2]
- **S3 투자자·애널리스트:** MLA 자체의 수익성, 채택성, 위험을 직접 평가한 투자자·애널리스트 발언은 확인되지 않았다.
- **판정:** S1–S3 모두 **근거 부족**이며, 전체 이해관계자 관점도 **근거 부족**이다.

#### ITME — 근거 부족
- **S1 경쟁 기술·기업:** ITME 자체를 대상으로 한 경쟁사 또는 대체기술 제공자의 공식 평가 발언은 확인되지 않았다.
- **S2 개발자·도입자:** CXL 일반 설명 자료는 DDR·PCIe의 제약과 CXL 풀링을 설명하지만, ITME의 실제 통합·운영 경험은 아니다.[W4] ITME 관련 소개 페이지의 성능 서술도 도입기업의 독립 운영 평가로 볼 수 없다.[W8]
- **S3 투자자·애널리스트:** 메모리·HBM 경쟁에 관한 인용은 ITME의 성장성이나 사업성을 직접 평가한 발언이 아니다.[W1] CXL 일반에 관한 업계 콘텐츠 역시 ITME에 대한 투자자 판단으로 사용할 수 없다.[W5]
- **판정:** S1–S3 모두 **근거 부족**이며, 전체 이해관계자 관점도 **근거 부족**이다.

#### 해석상 주의
- 위의 [W1]–[W5], [W8]은 배경 또는 간접 신호일 뿐이다. ITME 개별 기술의 도입·운영·투자자 의견으로 환원하지 않았다.

### 4.4 도메인 적용성(D1-D7)
| 평가항목 | MLA 판정 | MLA 근거 | ITME 판정 | ITME 근거 |
|---|---|---|---|---|
| ITME D1 워크로드 수용 능력 | 근거 부족 | - | 적합 | - |
| ITME D2 서비스 성능 | 근거 부족 | - | 조건부 | - |
| ITME D3 메모리 자원 효율성 | 근거 부족 | - | 적합 | - |
| ITME D4 품질·정보 보존 | 근거 부족 | - | 근거 부족 | - |
| ITME D5 운영 안정성 | 근거 부족 | - | 조건부 | - |
| ITME D6 도입·확장 용이성 | 근거 부족 | - | 조건부 | - |
| ITME D7 비용 효율성 | 근거 부족 | - | 조건부 | - |
| MLA D1 워크로드 수용 능력 | 적합 | - | 근거 부족 | - |
| MLA D2 서비스 성능 | 조건부 | - | 근거 부족 | - |
| MLA D3 메모리 자원 효율성 | 적합 | - | 근거 부족 | - |
| MLA D4 품질·정보 보존 | 적합 | - | 근거 부족 | - |
| MLA D5 운영 안정성 | 근거 부족 | - | 근거 부족 | - |
| MLA D6 도입·확장 용이성 | 조건부 | - | 근거 부족 | - |
| MLA D7 비용 효율성 | 조건부 | - | 근거 부족 | - |

#### DeepSeek-V2 MLA
- **D1 워크로드 수용 능력 — 적합:** DeepSeek-V2는 128K 토큰 문맥을 지원한다고 제시되며, 저자 보고 기준 KV cache 93.3% 감소가 제시된다.[1, p.1] 단, 최대 동시 요청 수와 최대 batch는 공개 근거 부족이다.
- **D2 서비스 성능 — 조건부:** 저자들은 배포 조건에서 처리량 결과를 보고했다.[1, p.16] 그러나 TTFT, 종단간 지연시간, 백분위 지연시간이 제공되지 않아 처리량 결과만으로 응답시간 품질을 확정할 수 없다.
- **D3 메모리 자원 효율성 — 적합:** MLA는 추론 시 압축 KV latent를 캐시하는 구조이며, 배포에서는 평균 6비트 KV cache 양자화가 사용되었다.[1, p.7] [1, p.16] 다만 HBM·호스트 DRAM·외부 계층별 사용량은 근거 부족이다.
- **D4 품질·정보 보존 — 조건부:** 128K에서의 견고한 결과는 DeepSeek-V2 전체 모델의 결과로 보고되었다.[1, p.13] MLA 단독의 인과 효과로 분리되지 않았으므로 조건부로 판단한다.
- **D5 운영 안정성 — 근거 부족:** 다중 턴 동시성, cache miss, 자원 경합, tail latency 및 성능 변동의 운영 지표가 부족하다.
- **D6 도입·확장 용이성 — 조건부:** MLA는 Transformer attention 구조에 통합된 방식이다.[1, p.4] 기존 MHA 모델의 checkpoint 변환·재사용 및 재학습 요건은 공개 근거 부족이다.
- **D7 비용 효율성 — 조건부:** KV cache 절감은 GPU 메모리 자원 절감 가능성을 시사하지만, 동등 워크로드 기준 비용·TCO·달러당 처리량의 직접 근거는 없다.[1, p.1] [1, p.16]

#### ITME
- **D1 워크로드 수용 능력 — 적합:** ITME 평가는 다중 턴 대화와 KV cache 메모리 압력 조건을 포함한다.[2, p.9] 다만 최대 지원 동시 요청·문맥 길이의 일반화 가능한 한계는 근거 부족이다.
- **D2 서비스 성능 — 조건부:** 저자들은 NVMe-oF 기준선 대비 처리량 결과를 보고했다.[2, p.2] 그러나 절대 TTFT, 종단간 지연시간, 백분위 지연시간이 부족하다.
- **D3 메모리 자원 효율성 — 적합:** ITME는 원격 CXL-hybrid-memory 계층에 대용량 상태를 배치하고, GPU 계산과 검색을 중첩하는 구조를 제시한다.[2, p.3] [2, p.8] 계층별 총 사용량과 토큰당 KV byte는 근거 부족이다.
- **D4 품질·정보 보존 — 근거 부족:** 원격 KV 저장·복원 및 prefetch가 생성 품질이나 장문맥 정보 보존에 미치는 영향을 직접 측정한 공개 결과가 없다.
- **D5 운영 안정성 — 조건부:** KV cache miss는 GPU 재계산으로 처리하도록 설계되었다.[2, p.7] 읽기·쓰기 경합은 읽기 대역폭 저하와 예측하기 어려운 latency excursion을 유발할 수 있어, 읽기 우선 I/O 정책이 제시된다.[2, p.6]
- **D6 도입·확장 용이성 — 조건부:** CXL-hybrid-memory 원격 서버, 계층 간 staging 및 전송 경로가 필요한 구조다.[2, p.3] [2, p.4] 구체적 장비 호환성, 네트워크 토폴로지, 프레임워크 변경 범위는 근거 부족이다.
- **D7 비용 효율성 — 조건부:** 메모리 확장을 비용 효율적으로 지향하는 구조이나, ITME 인프라의 TCO, 요청당 비용, 동등 워크로드 비용 비교는 제공되지 않았다.[2, p.1] [2, p.8]

#### 판정 연결
- 위 D1–D7 판정은 본 절의 인용 근거를 요약한 것이다. MLA D4는 모델 전체 결과와 MLA 단독 효과를 분리할 수 없으므로 일관되게 **조건부**로 둔다.

## 5. 종합 의견
#### 관점 매트릭스
| 관점 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| TRL | 4–6 추정: 통합·배포 조건 평가가 있으나 상용 채택 근거 부족 | 4–6 추정: 공개 아키텍처·평가가 있으나 제품·고객 운영 근거 부족 |
| 시장성 | 근거 부족: MLA 자체 시장·채택·생태계 자료 부족 | 혼재: 인접 CXL 시장 신호는 있으나 ITME 개별 채택 부족 |
| 이해관계자 | 근거 부족: S1–S3 직접 발언 부족 | 근거 부족: S1–S3 직접 발언 부족 |
| 도메인 | 메모리 효율은 적합, 성능·도입·비용은 조건부, 운영성은 근거 부족 | 수용력·메모리 효율은 적합, 성능·운영·도입·비용은 조건부, 품질은 근거 부족 |

#### 관점 간 일치
- 두 기술 모두 구현·평가 신호는 있으나, 개별 기술의 상용 출시·독립 고객 채택 근거가 부족하다. 따라서 TRL 7–9로 확장하지 않는다는 점에서 TRL·시장 판단이 일치한다.[1, p.16] [2, p.8]
- 두 기술 모두 KV cache 관련 메모리 병목 완화 근거는 있으나, 동등 워크로드 기준 비용과 TCO가 부족하다. 메모리 절감 또는 처리량 결과만으로 경제성을 확정하지 않는다는 점에서 도메인 판단이 일치한다.[1, p.16] [2, p.2]
- 두 기술 모두 처리량 관련 저자 보고는 있으나, 운영 판단에 필요한 절대 TTFT·종단간·백분위 지연시간의 공개 근거는 제한적이다.[1, p.16] [2, p.6]

#### MLA의 관점 간 상충
| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| 통합 검증과 시장 채택 | TRL·도메인: 통합 구조와 배포 조건 결과가 있어 긍정 신호 | 시장: MLA 자체의 고객 도입·생태계는 근거 부족 | 논문 내 평가와 독립 제품·고객 검증은 서로 다른 증거를 요구한다. |
| 메모리 효율과 전환 용이성 | D3: 압축 KV cache 구조로 적합 | D6: 기존 MHA 자산 전환은 조건부 | 운용 중인 MLA 모델의 효율과 기존 모델을 MLA로 전환하는 난이도는 별개다. |

#### ITME의 관점 간 상충
| 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|
| CXL 시장 환경과 ITME 상용화 | 시장 환경: 인접 CXL 시장 성장 신호 | TRL·시장 채택: ITME 개별 제품·고객 근거 부족 | CXL 일반 시장과 특정 ITME 아키텍처의 상용화는 동일하지 않다. |
| 메모리 확장과 장기 운영성 | D1·D3: 원격 계층 수용력·메모리 효율에 적합 | D5: I/O 경합에 따라 조건부 | 용량 확장 효과는 staging·prefetch·읽기/쓰기 경합 제어에 의존한다.[2, p.6] [2, p.7] |

#### 두 접근의 관계
- **interpretation:** MLA는 GPU에 저장되는 KV 표현량을 줄이고, ITME는 GPU 밖 계층의 저장 용량과 이동 경로를 다룬다.[1, p.6] [2, p.3]
- 따라서 두 접근은 경쟁적 단일 대안이라기보다 보완 가능한 계층 접근으로 해석할 수 있다. 다만 두 기술의 결합 시 메모리 절감 중복, 지연시간 상호작용, 품질 영향, 비용 효과를 직접 검증한 근거는 없으므로 결합 효과는 **근거 부족**이다.

## 6. 시사점
#### 관점에 따라 달라지는 평가
- 모델 개발 관점에서는 MLA의 KV 표현 압축이 직접적인 메모리 효율 신호가 될 수 있으나, 기존 MHA 모델 전환성과 MLA 단독 품질 효과는 별도 검증이 필요하다.[1, p.6] [1, p.13]
- 인프라 관점에서는 ITME가 원격 계층 확장 선택지를 제공하지만, 성능은 prefetch·staging·읽기/쓰기 경합 관리에 조건부로 의존한다.[2, p.6] [2, p.7]
- 시장 관점에서는 DeepSeek 모델 관련 보도와 CXL 인접 시장 자료가 존재하더라도, 이는 각각 MLA와 ITME의 직접 채택 증거가 아니다.[W6] [W11]

#### 이해관계자별 도입 전 확인 과제
| 이해관계자 | MLA 확인 과제 | ITME 확인 과제 |
|---|---|---|
| 클라우드 사업자 | 장문맥·동시성별 TTFT, 종단간·tail latency, 서빙 엔진 지원 여부 | CXL·RDMA·NIC 토폴로지, staging 용량, 경합 시 tail latency, 장애 격리 |
| 모델 개발사 | 기존 MHA checkpoint 호환성, 재학습 필요 여부, MLA 단독 품질 검증 | 원격 KV 복원·재계산이 품질과 스케줄링에 미치는 영향 |
| 메모리·HW 벤더 | MLA 워크로드에서의 KV 절감이 실제 메모리 요구량에 미치는 효과 | 장치 호환성, 읽기 우선 정책, CXL 계층·네트워크 병목, 장기 I/O 내구성 |
| 투자자 | MLA 기능 자체의 유료 도입, 고객 유지, 생태계 지원 자료 | ITME 제품화 상태, 고객 검증, 공급망, TCO 및 대체 메모리 계층과의 비교 |

- 위 과제는 추천 조건이 아니라 공개 근거의 공백을 해소하기 위한 검증 항목이다. 특히 비용은 두 기술 모두 동등 워크로드 기준으로 확인해야 하며, 서로 다른 모델·하드웨어·기준선의 저자 보고 수치를 직접 비교해서는 안 된다.

## 7. 한계점
#### 공개정보와 TRL의 한계
- **모든 TRL 판정은 논문, 특허, 상용 발표 등 공개정보에만 기초한 추정**이며, 공식 TRL 인증이 아니다.
- 논문 발표·기술 공개와 실제 고객 채택·운영 사이에는 시차가 있을 수 있다. 공개 채택 근거가 없다는 사실은 기술 또는 제품의 부재를 의미하지 않는다.
- TRL 4–6 구간에서는 비공개 통합, 고객 평가, 장애율, 운영 비용, 장기 안정성 정보가 공개되지 않는 공백이 크다. MLA와 ITME 모두 7–9 판정에 필요한 제품 출시·상용 운영 자료가 부족하다.

#### 수치와 적용 범위의 한계
- MLA와 ITME의 정량 결과는 저자 보고 기준이며, 모델, 하드웨어, 정밀도, 기준선, 워크로드와 측정 조건이 다르다. 따라서 수치를 같은 단위의 직접 우열 비교로 사용할 수 없다.[1, p.16] [2, p.2]
- MLA의 배포 결과에는 모델 전체 구조와 양자화 조건이 함께 포함되어 있어 MLA 단독 효과로 분리할 수 없다.[1, p.16]
- ITME는 원격 계층의 읽기·쓰기 경합 및 cache miss 처리 정책의 영향을 받는다.[2, p.6] [2, p.7] 품질 보존, 장기 다중 턴 안정성, TCO의 직접 근거는 부족하다.

#### 확증편향 방지 조치와 남은 한계
| 조치 | 실제 적용 내용 | 남은 한계 |
|---|---|---|
| HW 베이스라인 교차 확인 | InfiniGen과 CXL-PNM 문헌을 사용해 CPU 오프로드의 PCIe 병목과 CXL near-memory 접근을 확인했다.[3, p.4] [4, p.4] | 이는 ITME의 직접 우위 또는 한계의 정량 증명이 아니다. |
| 동일 형식 병기 | 두 기술을 쟁점·관점 A·관점 B·이유 형식으로 병기했다. | 증거량과 공개 범위 자체의 비대칭은 해소되지 않는다. |
| 종합 단계의 증거 제한 | 종합 판단은 이미 검증된 Evidence만 사용하고 신규 검색·신규 수치를 추가하지 않았다. | 제공된 Evidence의 원문 발췌 범위와 자료 시점에 의존한다. |

- 시장성·이해관계자 평가는 특히 직접 근거가 부족하다. MLA 기능 자체와 DeepSeek 모델 일반, ITME 개별 기술과 CXL 일반 시장을 구분했으나, 이 구분으로 인해 다수 항목은 근거 부족으로 남는다.
- 본 보고서는 순위, 우승 기술 또는 단일 추천을 제시하지 않는다.

## REFERENCE
- DeepSeek-AI(게시일 미확인). *DeepSeek-V2*. 원문 PDF.
- ITME 저자 미확인(게시일 미확인). *ITME 관련 원문 PDF*. 원문 PDF.
- InfiniGen 저자 미확인(게시일 미확인). *InfiniGen 관련 원문 PDF*. 원문 PDF.
- CXL-PNM 저자 미확인(게시일 미확인). *CXL-PNM 관련 원문 PDF*. 원문 PDF.
- TechCrunch(2026-05-06). *DeepSeek could hit $45B valuation from its first investment round*. TechCrunch, https://techcrunch.com/2026/05/06/deepseek-could-hit-45b-valuation-from-its-first-investment-round/
- TechCrunch(2026-07-14). *DeepSeek reportedly in talks to raise $1.5B, then IPO*. TechCrunch, https://techcrunch.com/2026/07/14/deepseek-reportedly-in-talks-to-raise-1-5b-then-ipo/
- Mordor Intelligence(게시일 미확인). *CXL Memory Controller IC Market*. Mordor Intelligence, https://www.mordorintelligence.com/industry-reports/cxl-memory-controller-ic-market
- TrendForce(2026-09-18). *OpenAI, Intel Reportedly Downplay CXL as HBM Replacement, Citing Use-Case and Data-Transfer Limits*. TrendForce, https://www.trendforce.com/news/2026/09/18/news-openai-intel-reportedly-downplay-cxl-as-hbm-replacement-citing-use-case-and-data-transfer-limits
- PyTorch 한국 사용자 모임(게시일 미확인). *DeepSeek-V2 MoE*. https://discuss.pytorch.kr/t/deepseek-v2-moe/4366
- HyperAccel(게시일 미확인). *What is CXL*. HyperAccel, https://hyper-accel.github.io/posts/what-is-cxl
- AlphaXiv(게시일 미확인). *ITME 관련 논문 소개 페이지*. AlphaXiv, https://www.alphaxiv.org/abs/2606.12556
- 주간조선(게시일 미확인). *CXMT·HBM 경쟁 관련 기사*. 주간조선, http://weekly.chosun.com/news/articleView.html?idxno=40084
- 삼성전자 반도체 뉴스룸(게시일 미확인). *AI 반도체 시장의 미래를 엿보다: CXL에 대해*. 삼성전자 반도체 뉴스룸, https://news.samsungsemiconductor.com/kr/behind-the-chip-ai-%EB%B0%98%EB%8F%84%EC%B2%B4-%EC%8B%9C%EC%9E%A5%EC%9D%98-%EB%AF%B8%EB%9E%98%EB%A5%BC-%EC%97%BF%EB%B3%B4%EB%8B%A4-cxl%EC%97%90-%EB%8C%80%ED%95%B4

#### 표 7. 증거 균형표 (중복 제거 후 수집·검증된 Evidence 수)
| 구분 | DeepSeek-V2 MLA | ITME | HW 베이스라인(InfiniGen·CXL-PNM) |
|---|---|---|---|
| 기술 조사 · 논문 청크 | 36 | 47 | 5 |
| 시장 · 웹 문서 | 4 | 2 | 0 |
| 이해관계자 · 웹 문서 | 2 | 5 | 0 |
| 도메인 · 논문 청크 | 37 | 44 | 0 |
| 합계 | 79 | 98 | 5 |

## REFERENCE
- [1] DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, p.1, p.4, p.6, p.7, p.13, p.16.
- [2] Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J. (SK hynix) (2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv preprint arXiv:2606.12556, p.1, p.2, p.3, p.4, p.6, p.7, p.8, p.9, p.11.
- [3] Lee, W., Lee, J., Seo, J., & Sim, J. (Seoul National University) (2024). InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. arXiv preprint arXiv:2406.19707, p.4.
- [4] Kim, D., Lee, M., Kim, J., Kwon, H., Jeong, H., Park, S.-S., et al. (Hanyang University, Samsung Electronics) (2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits. arXiv preprint arXiv:2511.00321, p.4.
- [W11] mordorintelligence.com (Tue, 28 Jul 2026 00:00:00 GMT). CXL Memory Controller IC Market Size, Share & 2031 Growth .... mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/cxl-memory-controller-ic-market
- [W13] trendforce.com (Fri, 18 Sep 2026 04:00:00 GMT). [News] OpenAI, Intel Reportedly Downplay CXL as HBM Replacement, Citing Use-Case and Data-Transfer Limits. trendforce.com, https://www.trendforce.com/news/2026/09/18/news-openai-intel-reportedly-downplay-cxl-as-hbm-replacement-citing-use-case-and-data-transfer-limits
- [W7] techcrunch.com (Tue, 14 Jul 2026 16:45:23 GMT). DeepSeek reportedly in talks to raise $1.5B, then IPO - TechCrunch. techcrunch.com, https://techcrunch.com/2026/07/14/deepseek-reportedly-in-talks-to-raise-1-5b-then-ipo/
- [W6] techcrunch.com (Wed, 06 May 2026 17:20:34 GMT). DeepSeek could hit $45B valuation from its first investment round - TechCrunch. techcrunch.com, https://techcrunch.com/2026/05/06/deepseek-could-hit-45b-valuation-from-its-first-investment-round/
- [W2] discuss.pytorch.kr (게시일 미확인). DeepSeek-V2: 강력하고 경제적이며 효율적인 전문가 혼합 .... discuss.pytorch.kr, https://discuss.pytorch.kr/t/deepseek-v2-moe/4366
- [W5] news.samsungsemiconductor.com (게시일 미확인). [Behind the CHIP] AI 반도체 시장의 미래를 엿보다. CXL에 .... news.samsungsemiconductor.com, https://news.samsungsemiconductor.com/kr/behind-the-chip-ai-%EB%B0%98%EB%8F%84%EC%B2%B4-%EC%8B%9C%EC%9E%A5%EC%9D%98-%EB%AF%B8%EB%9E%98%EB%A5%BC-%EC%97%BF%EB%B3%B4%EB%8B%A4-cxl%EC%97%90-%EB%8C%80%ED%95%B4
- [W4] hyper-accel.github.io (게시일 미확인). AI 시대의 필수 소비재, 메모리 이해하기 4편: CXL 이해하기. hyper-accel.github.io, https://hyper-accel.github.io/posts/what-is-cxl
- [W1] weekly.chosun.com (게시일 미확인). 미국 압력과 중국 추격… K반도체, CXL로 대응한다 - 주간조선. weekly.chosun.com, http://weekly.chosun.com/news/articleView.html?idxno=40084
- [W8] alphaxiv.org (게시일 미확인). ITME: Inference Tiered Memory Expansion with .... alphaxiv.org, https://www.alphaxiv.org/abs/2606.12556
