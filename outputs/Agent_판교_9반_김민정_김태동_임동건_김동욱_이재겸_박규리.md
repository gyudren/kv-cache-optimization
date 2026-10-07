# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-10-07

## SUMMARY
- **DeepSeek-V2 MLA**는 공개 논문 기반 보수적 추정으로 **TRL 4-6**이다. 저랭크 KV 공동 압축 및 MHA 통제 비교는 확인되지만, 특정 MLA 아키텍처의 실제 서비스 운용·상용 채택은 공개 근거 부족이다 [1, p.6] [1, p.31] [1, p.32].
- **ITME**도 FPGA 프로토타입과 CMM·SSD 기반 평가를 근거로 **TRL 4-6**으로 추정된다. 다만 ITME 자체의 데이터센터 서비스 운용·제품 채택은 확인되지 않는다 [2, p.2] [2, p.11].
- MLA는 메모리 효율·품질 통제 비교에서 긍정 신호가 있으나, 운영 안정성과 채택 증거가 부족하다. ITME는 계층 확장·처리량 저자 보고가 있으나, 경합·캐시 미스의 구현 병목과 투자 관점의 우려가 병존한다 [1, p.31] [1, p.32] [2, p.11] [W24].
- 두 기술은 경쟁하는 단일 대안이 아니라, MLA는 **KV 표현량 축소**, ITME는 **잔여 KV의 계층형 수용·이동**을 맡는 상호 다른 계층의 접근이다 [1, p.7] [2, p.2].

## 1. 분석 배경
KV cache는 생성 추론에서 이전 토큰의 key·value를 저장해 어텐션을 가속하지만, MHA에서는 이 캐시가 최대 batch와 시퀀스 길이를 제한하는 배포 병목이 될 수 있다 [1, p.7]. 본 도메인은 대규모 동시 요청, 비용 민감성, 장문맥 요구가 공존하는 데이터센터·클라우드 장문맥 LLM 서빙이다 [D]. 설계 기준상 KV cache는 레이어·head·문맥 길이·동시 사용자 또는 batch·저장 정밀도에 따라 커지며, HBM 점유는 동시 요청과 장문맥 수용을 제약한다 [D].

따라서 본 평가는 우열 선정이 아니라, 같은 병목에 대해 모델 구조를 바꾸는 MLA와 메모리 계층을 확장하는 ITME가 기술 성숙도, 시장성, 이해관계자 및 도메인 적용성에서 각각 어떤 근거와 제약을 보이는지 비교한다 [D].

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB(≈74.5GiB) 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 약 3.4% |
| 128K | 40 GiB | 320 GiB | 약 53.7% |
| 1M | 320 GiB | 2,560 GiB | 약 429.5% |

산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. 요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). HBM 대비 비율은 80GB = 80×10⁹B ≈ 74.5GiB를 분모로 계산했다(설계서 A-2의 3%·50%·400%는 GiB/GB 단위를 혼용한 근사치라 여기서 바로잡음) [D].

## 2. 기술 선정
선정 방식은 **2안: 조 토의 후 직접 선정**이다 [D]. SW는 DeepSeek-V2 MLA를 선정했다. MLA는 key·value를 저차원 잠재 벡터로 공동 압축하고 추론 시 이를 캐시하는 구조이므로 KV 표현량 자체를 줄이는 사례이기 때문이다 [1, p.7]. 단, 기존 MHA 모델에 사후 적용 가능한지와 재학습 요구사항은 공개 근거 부족이다.

HW는 SK hynix의 ITME를 선정했다. ITME는 CXL-hybrid 확장 계층과 프리페치를 통해 장문맥 KV cache 및 가중치를 수용·이동하는 구조를 제시하기 때문이다 [2, p.1] [2, p.2]. 두 기술은 같은 병목을 서로 다른 계층에서 다루므로 함께 비교한다 [D]. KIVI·TurboQuant 등 비선정 후보는 설계 검토 의견만 있으며, 본 보고서에서는 검증된 비교 근거가 없어 성능 판단을 하지 않는다 [D].

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
| 진영 | 모델·어텐션 SW | CXL-hybrid 계층 메모리 HW/SW |
| 작동 계층 | KV 표현·어텐션 구조 | GPU·호스트·CXL-hybrid·원격 계층 |
| 핵심 접근 | key·value 저랭크 공동 압축, latent KV 캐시 | 예측 가능한 가중치·장문맥 KV의 확장 계층 배치·프리페치 |
| 저자 보고 효과 | MHA 대비 small MoE KV cache 14%, large MoE 4% | NVMe-oF 기준선 대비 처리량 1.80배 |
| 전제조건 | MLA 구조·가중치 및 추론 구현 | CXL-hybrid, RDMA·DMA, 프리페치 제어 |
| 한계 | 단독 지연·처리량 및 운영 안정성 근거 부족 | 경합·캐시 미스에서 FPGA 구현 병목 |

#### DeepSeek-V2 MLA
- **fact:** 추론 시 압축 KV latent를 캐시하고, key/value 업프로젝션은 각각 query/output 프로젝션에 흡수할 수 있다 [1, p.7].
- **저자 보고:** attention 외 구조가 동일한 비교에서 KV cache 원소 수는 small MoE의 MHA 110.6K·MLA 15.6K, large MoE의 MHA 860.2K·MLA 34.6K였다 [1, p.31] [1, p.32]. 품질·메모리 결과는 단일 저자 통제 비교이며, 장문맥 정보 보존의 직접 측정은 근거 부족이다.

#### ITME
- **fact:** 활성화·working KV는 GPU 또는 호스트 계층에, 가중치·장문맥 KV는 원격 확장 계층에 두고 프리페치로 이동을 연산과 중첩한다 [2, p.2].
- **저자 보고:** FPGA 프로토타입·CMM·PCIe Gen5 SSD 기반 평가에서 NVMe-oF 기준선 대비 처리량 1.80배를 보고했다 [2, p.2] [2, p.11]. 다만 비교 기준선은 이상적 NVMe-oF 상한 시나리오로 설명되며 일반 배포 효과로 일반화할 수 없다 [2, p.10].

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | TRL 4-6(공개 논문 기반 보수적 추정). DeepSeek-V2 논문은 MLA를 포함한 모델 구현과 MHA 대비 통제 비교를 제시하므로 통합 구현·검증 수준의 근거는 있다([1, p.1] [1, p.31] [1, p.32]). 그러나 제공된 MLA 원문만으로는 특정 MLA 아키텍처의 실제 서비스 운용 또는 상용 채택을 확인할 수 없으므로 TRL 7-9로 올릴 수 없다. | technology_trl 근거: MLA의 저랭크 KV 공동 압축 구조와 캐시 방식([1, p.6-7]), attention mechanism 외에는 동일한 small/large MoE 비교 및 KV cache·벤치마크 결과([1, p.31-32]). 계열 TRL: MLA가 속한 일반 Transformer attention/KV-cache 최적화 계열의 별도 제품군 성숙도는 제공된 원문만으로 독립 판정할 수 없다. 논문은 MQA·GQA를 관련 방법으로 언급하지만([1, p.6]), 이 발췌문에는 해당 계열의 제품 출시·상용 운용 근거가 없다. 따라서 family_trl은 '근거 부족'이며, MLA 자체 논문 검증 수준과 혼동하지 않는다. 제공된 GitHub 발췌문은 FlashMLA가 DeepSeek-V3/V3.2-Exp를 구동한다고 하나, 발췌된 부분은 sparse-attention/DSA 설명이고 DeepSeek-V2 MLA 아키텍처 자체의 상용 서비스 채택을 직접 입증하지 않는다. |
| ITME | TRL 4-6(공개 논문 기반 보수적 추정). ITME 논문은 FPGA 기반 하드웨어 프로토타입, SK hynix CMM 및 PCIe Gen5 SSD 기반 평가, LLM 추론 처리량 비교를 보고하므로 구현·유사 운용 조건 검증의 근거가 있다([2, p.2] [2, p.11]). 다만 실제 데이터센터 서비스 또는 상용 제품으로서 ITME 자체의 운용·채택을 문서화한 근거는 제공되지 않아 TRL 7-9는 부여할 수 없다. | technology_trl 근거: CXL-hybrid 계층·프리페치 설계([2, p.2] [2, p.11]), FPGA 프로토타입 및 Llama-3.1 8B/70B 동일 설정 비교([2, p.11]), NVMe-oF 기준선 대비 처리량 보고([2, p.2]). 계열 TRL: CXL 메모리 또는 CXL-hybrid 메모리 제품군 일반의 성숙도는 ITME 자체와 분리되어야 한다. 제공된 ITME 원문은 SK hynix CMM을 사용한 실험을 말하지만([2, p.11]), 일반 CXL 메모리 제품군의 제품 출시·상용 운용을 직접 문서화하지 않는다. 제공된 웹 발췌문도 ITME 개별 기술의 출시·운용을 입증하지 않으며, 일부는 CXL과 무관한 DDR5/HBM 생산 기사다. 따라서 family_trl은 '근거 부족'이다. |

#### DeepSeek-V2 MLA
**판정 근거:** 공개 논문은 MLA의 저랭크 KV 공동 압축, 캐시 방식 및 MHA 대비 통제 비교를 제시한다 [1, p.7] [1, p.31] [1, p.32]. 따라서 구현·검증 근거는 있으나, 특정 DeepSeek-V2 MLA의 실서비스 운용 또는 상용 채택은 확인되지 않아 TRL 7-9 근거는 부족하다. MQA·GQA 언급은 MLA가 속한 일반 attention/KV 최적화 제품군의 상용 성숙도를 입증하지 않으므로 **family TRL은 근거 부족**이다 [1, p.6].

#### ITME
**판정 근거:** ITME 논문은 CXL-hybrid 계층·프리페치 설계, FPGA 프로토타입, CMM·SSD 기반 LLM 추론 평가를 제시한다 [2, p.2] [2, p.11]. 그러나 ITME 자체의 상용 제품 또는 현장 데이터센터 운용은 확인되지 않는다. SK hynix CMM 사용 실험은 일반 CXL 메모리·CXL-hybrid 제품군의 출시·운용을 직접 입증하지 않으므로 **family TRL은 근거 부족**이다 [2, p.11].

모든 TRL은 논문·공개 발표 범위의 **공개 정보 기반 추정**이며, 공식 TRL 인증이 아니다.

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 긍정 | 혼재 |
| 상용화·채택 현황 | 근거 부족 | 근거 부족 |
| 생태계 지지 | 긍정 | 근거 부족 |
| 종합 | 근거 부족 | 근거 부족 |
| 근거 | [W23] [W20] [W4] [W6] [W5] | [W22] [W8] [W1] [W26] [W14] [W16] |

#### DeepSeek-V2 MLA
시장 규모·성장성의 **긍정** 신호는 LLM 시장 성장 전망에 관한 간접 근거다 [W23]. 이는 MLA 전용 또는 추론 최적화 시장 규모가 아니다. 생태계 지지는 vLLM이 MLA prefill/decode backend와 DeepSeek-style MLA를 문서화하고, SGLang이 MLA 최적화를 발표한 점에서 **긍정**이다 [W5] [W6] [W20]. 반면 유료 서비스, 고객 사용량, 계약, DeepSeek 모델 제공과 구분된 MLA 자체의 상용 채택은 **근거 부족**이다 [W4]. 따라서 종합은 근거 부족이다.

#### ITME
CXL memory controller IC 시장에서 AI 추론·RAG·KV cache 워크로드 성장 전망이 제시되어 시장 신호는 **혼재**다 [W22]. CXL 메모리 풀링·공유의 실제 배포가 추진력을 얻는다는 계열 신호도 있다 [W1]. 그러나 이는 ITME의 매출·제품화·고객 도입을 뜻하지 않으며, ITME와 서빙 프레임워크의 직접 통합도 근거 부족이다. 그러므로 상용화·채택 및 생태계는 근거 부족, 종합도 근거 부족이다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 근거 부족 | 긍정 |
| 도입 기업·개발자 | 근거 부족 | 근거 부족 |
| 투자 업계 | 근거 부족 | 우려 |
| 종합 | 근거 부족 | 혼재 |
| 근거 | [W17] [W12] | [W13] [W24] |

#### DeepSeek-V2 MLA
경쟁 진영, 도입 기업·개발자, 투자 업계의 MLA 자체 직접 평가는 모두 **근거 부족**이다. NVIDIA 게시물은 DeepSeek-V4 및 서빙 프레임워크를 언급하지만 MLA 또는 DeepSeek-V2 MLA를 직접 평가하지 않는다 [W17]. 개인 개발자 게시물의 DeepSeek 서빙 경험도 MLA 기능, 조건 및 재현 절차를 분리하지 않는다 [W12].

#### ITME
경쟁 진영은 **긍정** 신호가 있으나 CXL 계열에 대한 간접 근거다. Samsung은 자사 평가에서 DRAM 용량을 넘는 KV cache 수요에서 CXL memory pool이 더 큰 footprint를 수용하며 안정적 성능을 유지했다고 주장했다 [W13]. ITME 자체에 대한 직접 비교는 아니다. 투자 업계는 **우려** 신호가 있다. Gartner 분석가와 Greyhound Research 분석가는 CXL memory pooling 등이 최적화에는 기여하나 단기 구조적 메모리 제약을 제거하지 못하며 전환이 느릴 수 있다고 평가했다 [W24]. 도입 기업·개발자의 ITME 직접 운영 경험은 근거 부족이므로 종합은 혼재다.

### 4.4 도메인 적용성(D1-D7)
| 평가항목 | MLA 판정 | MLA 근거 | ITME 판정 | ITME 근거 |
|---|---|---|---|---|
| D1 워크로드 수용 능력 | 조건부 | [1, p.1] [1, p.7] [1, p.8] | 조건부 | [2, p.9] |
| D2 서비스 성능 | 조건부 | [1, p.16] [1, p.1] | 조건부 | [2, p.2] [2, p.9] [2, p.11] |
| D3 메모리 자원 효율 | 적합 | [1, p.1] [1, p.7] [1, p.8] | 조건부 | [2, p.2] [2, p.6] [2, p.9] |
| D4 품질·정보 보존 | 적합 | [1, p.31] | 근거 부족 | [2, p.1] [2, p.2] [2, p.11] |
| D5 운영 안정성 | 근거 부족 | [1, p.7] [1, p.16] | 제약 | [2, p.11] |
| D6 도입·확장 용이성 | 조건부 | [1, p.7] [1, p.8] | 조건부 | [2, p.1] [2, p.2] [2, p.8] |
| D7 비용 효율성 | 조건부 | [1, p.16] [1, p.1] | 조건부 | [2, p.9] |

#### DeepSeek-V2 MLA
D1·D2·D6·D7은 **조건부**다. 모델은 장문맥 지원과 KV cache 축소를 제시하지만, 최대 동시 요청·TTFT·메모리 압박 지연·금전 비용의 직접 검증이 부족하다 [1, p.1] [1, p.7] [1, p.16]. D3은 저랭크 캐시 구조와 MHA 통제 비교에 근거해 **적합**이다 [1, p.7] [1, p.31] [1, p.32]. D4도 통제 비교의 품질 저하 부재에 근거해 **적합**이나, 이는 저자 보고·단일 출처이고 장문맥 정보 보존 직접 평가는 근거 부족이라는 한계를 함께 둔다 [1, p.31] [1, p.32]. D5는 동시성·경합·장기 안정성 근거 부족이다.

#### ITME
D1·D2·D3·D6·D7은 **조건부**다. 다중 대화·장기 turn 평가와 처리량 저자 보고는 있으나 절대 TTFT·꼬리 지연·계층별 점유량·TCO가 충분하지 않다 [2, p.2] [2, p.9] [2, p.11]. D4는 생성 품질·정확도·장문맥 보존률 직접 평가가 없어 **근거 부족**이다. D5는 CXL-hybrid 경합과 캐시 미스 시 FPGA 대역폭 저하가 보고되어 **제약**이다 [2, p.10] [2, p.11].

## 5. 종합 의견
| 관점 | MLA | ITME |
|---|---|---|
| TRL | TRL 4-6 | TRL 4-6 |
| 시장 | 근거 부족 | 근거 부족 |
| 이해관계자 | 근거 부족 | 혼재 |
| 도메인 | 메모리 효율·품질 적합, 나머지 조건부 또는 근거 부족 | 안정성 제약, 품질 근거 부족, 나머지 조건부 |

두 기술 모두 논문·프로토타입 근거는 있으나, 개별 기술의 현장 채택과 장기 운용은 확인되지 않는다는 점에서 관점이 일치한다. MLA는 메모리 절감의 통제 비교가 비교적 명확하고 [1, p.31] [1, p.32], ITME는 용량 확장·프리페치의 시스템 실험이 있으나 경합 병목이 확인된다 [2, p.10] [2, p.11].

| 기술 | 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|---|
| MLA | 메모리·품질 대 운영 검증 | D3·D4 적합 | TRL·시장·이해관계자 근거 부족 | 통제 벤치마크와 실서비스 채택은 요구 근거가 다름 |
| MLA | 시스템 처리량 | D2 조건부 | MLA 단독 지연·TTFT 근거 부족 | 배포 성능은 FP8·KV 양자화 등 결합 효과 [1, p.16] |
| ITME | 처리량 대 안정성 | D2 조건부 | D5 제약 | 전체 처리량과 경합·캐시 미스는 다른 조건의 결과 [2, p.2] [2, p.11] |
| ITME | CXL 계열 신호 대 개별 상용화 | 시장·공급사 신호 존재 | ITME 제품화·채택 근거 부족 | 계열 전망은 ITME 개별 성과를 보장하지 않음 [W1] [W22] |

MLA는 캐시 **표현량**을 줄이고 ITME는 남은 캐시의 **수용·이동 계층**을 확장한다. 따라서 상호보완 가능성은 있으나, 결합 구성의 성능·호환성·비용을 직접 비교한 공개 근거는 없다.

## 6. 시사점
평가의 차이는 기술 계층과 확인 가능한 근거의 차이에서 나온다. 모델 개발 관점에서는 MLA의 구조적 KV 절감이 중심이고 [1, p.7], 인프라 관점에서는 ITME의 CXL-hybrid 배치·프리페치와 원격 접근 지연이 중심이다 [2, p.2]. 시장·채택 관점에서는 두 기술 모두 개별 기술의 고객 운영 근거가 부족하다.

| 이해관계자 | 도입 전 확인 과제 |
|---|---|
| 클라우드 사업자 | 실제 요청 분포에서 동시성, TTFT, 평균·p95/p99 지연, 캐시 미스·경합, 계층별 메모리 점유와 TCO를 측정 |
| 모델 개발사 | MLA 적용 모델·가중치 조건, 기존 MHA/GQA 전환·재학습 필요성, 장문맥 정보 보존을 검증 |
| 메모리·HW 벤더 | CXL-hybrid·RDMA·DMA·프리페치 API, 장애·복구, 경합 시 대역폭 및 서빙 엔진 통합을 검증 [2, p.2] [2, p.5] |
| 투자자 | CXL 계열 전망과 ITME 개별 제품화·고객 도입을 분리하고, 단기 구조 제약 및 도입 시차를 확인 [W24] |

이는 단일 추천이 아니라, 각 조직이 자신의 병목 위치와 검증 가능한 운영 조건에 맞춰 확인해야 할 과제다.

## 7. 한계점
**자료 기준 시점**: 웹 자료 검색·검증일 2026-10-07 (Tavily 검색 결과 기준). 이후 공개된 발표·제품 정보는 반영되지 않았다.

모든 TRL 판정은 논문, 특허, 상용 발표 등 **공개 정보만에 근거한 추정**이다. 기술 발표와 실제 채택 사이에는 시차가 있을 수 있으며, TRL 4-6 구간의 비공개 구현·신뢰성·현장 운영 정보에는 공백이 있다.

#### 근거 부족
- 기술: MLA의 단독 처리량·지연 조건, 별도 메모리 계층 메커니즘, 상세 하드웨어·분산 학습·전체 평가 설정이 부족하다.
- 시장: MLA 전용 추론 최적화 시장의 방법론 포함 원문 자료, MLA 자체 유료 서비스·고객 운영·계약, DeepSeek 모델 제공과 구분된 MLA 기능 지원 자료가 부족하다.
- 이해관계자: MLA 직접 경쟁사 발언, 식별 가능한 데이터센터·클라우드 도입 사례, 재현 가능한 운영 벤치마크가 부족하다.
- 도메인: GPU 총메모리·가중치 점유, 요청별 문맥·생성 길이, KV cache 바이트 정밀도·런타임 오버헤드가 부족하다.
- MLA D3·D4 적합은 저자 통제 비교 중심의 단일 출처에 의존한다. 특히 D4의 장문맥 정보 보존 직접 측정은 근거 부족이며, 저자 보고와 독립 검증을 구분했다 [1, p.31] [1, p.32].

확증편향 방지 조치로 ITME는 HW 베이스라인 논문을 교차 확인했고 [3, p.12] [4, p.1], 두 기술은 동일 보고 형식으로 병기했으며, 종합 단계에서는 이미 검증된 Evidence만 사용했다. 그럼에도 결합 구성, 장기 운용, 독립 재현, 비용·채택 자료의 공백은 남는다.

#### 근거 공백 (Supervisor 기록)
- 근거 부족: tech: 기술 성숙도(TRL) 재조사 2회 후에도 근거 부족 — 제공된 인용문에는 별도의 ‘메모리 계층(memory hierarchy)’ 메커니즘에 대한 설명이 없다.; 제공된 발췌문에는 하드웨어 종류·대수, 분산 학습의 구체적 토폴로지, 학습 총 스텝/시간, 평가 데이터셋별 전체 결과, SFT 및 RL의 세부 구현 설정은 없습니다.; MLA의 처리량(throughput) 수치 및 비교 조건
- 근거 부족: market: 시장성 재조사 2회 후에도 근거 부족 — MLA가 속한 LLM 추론 서빙·추론 최적화 시장을 직접 정의하고, 기준연도·전망연도·단위·CAGR·발행일·조사 방법론을 제시하는 원문 시장조사 웹 자료; DeepSeek-V2 MLA 자체를 명시한 유료 서비스 출시, 고객 운영 사례, 사용량, 계약 또는 독립 보도; Amazon Bedrock·Dynatrace 등에서 DeepSeek 모델 제공과 별개로 MLA 기능 자체를 지원한다고 명시한 공식 문서
- 근거 부족: stakeholder: 이해관계자 재조사 2회 후에도 근거 부족 — MLA 또는 DeepSeek의 MLA 서빙에 대해 경쟁 기업·경쟁 기술 제공자가 직접 발언한 공식 발표, 인터뷰 또는 기술 문서; 식별 가능한 데이터센터·클라우드 도입 기업 또는 운영 조직의 MLA 직접 도입 경험과 호환성·유지보수·품질·성능 자료; 서빙 프레임워크·모델 버전, 하드웨어 구성, 워크로드, 측정 지표 및 재현 절차를 갖춘 MLA 관련 공식 이슈 또는 벤치마크
- 근거 부족: domain: 도메인 적용성 재조사 2회 후에도 근거 부족 — GPU 메모리 총용량 및 모델 가중치가 차지하는 메모리; 요청별 컨텍스트 길이와 생성 길이; KV 캐시의 실제 바이트 정밀도/데이터형 및 런타임 오버헤드
- 근거 부족: tech: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족
- 근거 부족: market: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족
- 근거 부족: stakeholder: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족
- 근거 부족: domain: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족
- 근거 부족: domain: 품질 평가 미달, 재조사 한도 소진 — bias_control: 4.4 MLA D4는 “[1, p.31]”의 저자 통제 벤치마크 결과 하나에 의존해 ‘적합’으로 판정한다. 같은 절에서 장문맥 정보 보존의 직접 측정 부재 또는 저자 보고·단일 출처라는 반대 방향의 한계를 해당 판정에 병기하지 않았다. D3의 ‘적합’도 사실상 동일 저자 논문 단일 출처의 KV 비교에 의존한다. 반면 ITME에는 원
- 근거 부족: domain: 품질 평가 미달, 재조사 한도 소진 — bias_control: 4.4 도메인 적용성 표와 MLA 단락에서 D3·D4를 각각 “적합”으로 판정하지만, 두 판정의 직접 근거는 사실상 저자 통제 비교인 [1, p.31-32] 중심이다. 본문 후반에 “단일 저자 논문”, “장문맥 정보 보존의 직접 측정은 부족”이라고 적었으나, D4의 ‘품질·정보 보존’ 적합 판정과 같은 위치에서 저자 보고·단일 출처

#### 표 7. 증거 균형표 (중복 제거 후 수집·검증된 Evidence 수)
| 구분 | DeepSeek-V2 MLA | ITME | HW 베이스라인(InfiniGen·CXL-PNM) |
|---|---|---|---|
| 기술 조사 · 논문 청크 | 29 | 49 | 5 |
| 기술 조사 · 웹 문서 | 6 | 4 | 0 |
| 시장 · 웹 문서 | 6 | 6 | 0 |
| 이해관계자 · 웹 문서 | 3 | 3 | 0 |
| 도메인 · 논문 청크 | 39 | 44 | 0 |
| 합계 | 83 | 106 | 5 |

## REFERENCE
- [1] DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, p.1, p.6, p.7, p.8, p.16, p.31, p.32.
- [2] Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J. (SK hynix) (2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv preprint arXiv:2606.12556, p.1, p.2, p.5, p.6, p.8, p.9, p.10, p.11.
- [3] Lee, W., Lee, J., Seo, J., & Sim, J. (Seoul National University) (2024). InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. arXiv preprint arXiv:2406.19707, p.12.
- [4] Kim, D., Lee, M., Kim, J., Kwon, H., Jeong, H., Park, S.-S., et al. (Hanyang University, Samsung Electronics) (2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits. arXiv preprint arXiv:2511.00321, p.1.
- [W24] networkworld.com (게시일 미확인). Chip wafer shortage will run through 2030 as AI demand overwhelms supply: SK Hynix chief | Network World. networkworld.com, https://www.networkworld.com/article/4146270/chip-wafer-shortage-will-run-through-2030-as-ai-demand-overwhelms-supply-sk-hynix-chief.html
- [W23] mordorintelligence.com (Fri, 11 Sep 2026 00:00:00 GMT). Large Language Model Market Size, Growth & Outlook | Industry Report 2031. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/large-language-model-llm-market
- [W20] lmsys.org (게시일 미확인). SGLang v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision - LMSYS Org. lmsys.org, https://www.lmsys.org/blog/2024-09-04-sglang-v0-3
- [W4] docs.dynatrace.com (Fri, 28 Aug 2026 00:00:00 GMT). AI Observability integrations — Dynatrace Docs. docs.dynatrace.com, https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/integrations
- [W6] docs.vllm.ai (게시일 미확인). Attention Backend Feature Support - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/stable/design/attention_backends
- [W5] docs.vllm.ai (게시일 미확인). mla_attention - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention
- [W22] mordorintelligence.com (Tue, 28 Jul 2026 00:00:00 GMT). CXL Memory Controller IC Market Size, Share & 2031 Growth Trends Report. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/cxl-memory-controller-ic-market
- [W8] finance.biggo.com (Tue, 06 Oct 2026 18:17:28 GMT). AI Memory Shortage Could Last Years; Morgan Stanley Names Nvidia, Micron Among 6 Outperformers — BigGo Finance. finance.biggo.com, https://finance.biggo.com/news/eba2a946-29a5-4439-9bcb-99029f17b123
- [W1] computeexpresslink.org (게시일 미확인). Breaking Boundaries in Memory: Highlights from AI Infra Summit and SDC 2025 - Compute Express Link. computeexpresslink.org, https://computeexpresslink.org/blog/breaking-boundaries-in-memory-highlights-from-ai-infra-summit-and-sdc-2025-4198
- [W26] penguinsolutions.com (게시일 미확인). CXL 3.0 vs. CXL 2.0: Advancing Composable AI Infrastructure. penguinsolutions.com, https://www.penguinsolutions.com/en-us/resources/blog/cxl-30-vs-cxl-20-key-differences-and-benefits
- [W14] vtechworks.lib.vt.edu (게시일 미확인). Disaggregated LLM Serving with CXL Shared Memory KV .... vtechworks.lib.vt.edu, https://vtechworks.lib.vt.edu/server/api/core/bitstreams/fb5697b4-124d-4dca-be49-dab91e3e0c2a/content
- [W16] asteralabs.com (게시일 미확인). How CXL Transforms RAG and KV Cache Performance. asteralabs.com, https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance
- [W17] facebook.com (게시일 미확인). NVIDIA AI - ✨ DeepSeek-V4 is here — a million-token.... facebook.com, https://www.facebook.com/NVIDIAAI/posts/-deepseek-v4-is-here-a-million-token-context-16t-parameter-powerhouse-optimized-/1400447275452883
- [W12] news.ycombinator.com (게시일 미확인). DeepSeek Open Source FlashMLA – MLA Decoding Kernel for Hopper GPUs | Hacker News. news.ycombinator.com, https://news.ycombinator.com/item?id=43155023
- [W13] semiconductor.samsung.com (게시일 미확인). Breaking AI Memory Limits with CXL Memory Pooling. semiconductor.samsung.com, https://semiconductor.samsung.com/news-events/tech-blog/breaking-ai-memory-limits-with-cxl-memory-pooling
- [D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 (A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.
