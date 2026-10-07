# KV cache 최적화 기술 다관점 평가 보고서

DeepSeek-V2 MLA(SW 압축) vs ITME(HW 메모리 확장) — 데이터센터·클라우드 장문맥 LLM 서빙

판교 9반 2조 · 김동욱, 김민정, 김태동, 박규리, 이재겸, 임동건 · 2026-10-07

## SUMMARY
- **TRL 추정:** MLA는 공개 저장소의 서빙 커널 통합 진술을 근거로 TRL 7, ITME는 FPGA 프로토타입·평가 근거를 바탕으로 TRL 4-6으로 추정된다. 두 판단은 공개 정보 기반 추정이다 [W8] [2, p.10] [2, p.11].
- **도메인:** MLA는 KV cache 표현 축소의 저자 보고 근거가, ITME는 확장 계층과 프리페치의 저자 보고 근거가 있어 모두 `조건부` 신호가 우세하다. MLA의 종단 간 서빙 성능·안정성, ITME의 품질 보존은 각각 근거 부족이다 [1, p.31] [1, p.32] [2, p.2] [2, p.10].
- **관점 차이:** MLA는 생태계 구현 근거가 있으나 시장의 고객 채택 근거는 직접적이지 않다. ITME의 CXL 계열 시장 신호는 존재하지만 ITME 개별 기술의 제품화·고객 운영 근거로 확장할 수 없다 [W3] [W4] [W5] [W13] [W22].
- **관계:** MLA는 모델의 KV 표현을 줄이고, ITME는 남은 KV cache와 가중치의 저장·이동 계층을 확장한다. 같은 병목을 다루되 대체재가 아니라 상이한 계층의 접근이며, 결합 효과는 근거 부족이다.
- 본 평가는 우열이나 단일 도입안을 제시하지 않으며, 데이터센터·클라우드 장문맥 서빙이라는 설계 조건에서 관점별 확인 과제를 제시한다 [D].

## 1. 분석 배경
#### 설계 기준
KV cache는 생성 단계에서 이전 토큰의 Key·Value를 보관해 어텐션 계산을 돕는 상태다. 설계 기준상 그 규모는 레이어 수, attention head 수, 문맥 길이, 동시 사용자·batch, 저장 정밀도에 비례해 증가한다. HBM 점유가 커지면 동시 요청·최대 batch 및 장문맥 요청이 제약되고, GPU 증설, KV 이동·로딩 지연, 폐기 후 재계산 문제가 발생할 수 있다 [D].

평가 도메인은 대규모 동시 요청, 비용 민감성, 문맥 길이 민감성이 겹치는 데이터센터·클라우드 기반 장문맥 LLM 서빙이다 [D]. MLA와 ITME는 각각 모델 구조와 메모리 인프라에서 이 병목에 접근하므로, 동일 단위의 성능 우열 대신 기술 성숙도, 시장성, 이해관계자 및 도메인 적용성에서 신호와 한계를 비교한다 [D].

#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]
| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB(≈74.5GiB) 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 약 3.4% |
| 128K | 40 GiB | 320 GiB | 약 53.7% |
| 1M | 320 GiB | 2,560 GiB | 약 429.5% |

산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. 요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). HBM 대비 비율은 80GB = 80×10⁹B ≈ 74.5GiB를 분모로 계산했다(설계서 A-2의 3%·50%·400%는 GiB/GB 단위를 혼용한 근사치라 여기서 바로잡음) [D].

## 2. 기술 선정
#### 선정 방식과 기준
기술은 **2안: 조 토의 후 직접 선정** 방식으로 정했다. 평가는 TRL, 시장성, 이해관계자, D1-D7 도메인 기준을 적용하며, 목적은 우승 기술 선정이 아니라 관점별 장점·제약·근거 수준을 드러내는 것이다 [D].

- **DeepSeek-V2 MLA:** Key·Value를 저차원 잠재 벡터로 압축하는 어텐션 구조 재설계로, KV cache 자체를 줄이는 소프트웨어 접근이라는 점에서 선정했다. 기존 모델에 사후 적용하기 어려울 수 있다는 설계상 제약도 함께 검토 대상이다 [D].
- **ITME:** CXL 기반 DRAM-NVMe hybrid memory와 프리페치를 활용하여 원격 메모리 계층을 구성하는 하드웨어 접근이라는 점에서 선정했다. 장문맥·다중 턴 KV cache의 수용과 이동 문제를 검토 대상으로 삼았다 [D].

KIVI와 TurboQuant는 검토 의견으로만 언급되었으며, 제공된 검증 입력에는 비선정 사유를 뒷받침하는 기술 비교 근거가 없다 [D].

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
| 진영 | 모델·추론 소프트웨어 | CXL-hybrid-memory 기반 인프라 |
| 작동 계층 | 어텐션의 K·V 표현 | GPU·호스트·원격 메모리·저장장치 경로 |
| 핵심 접근 | K·V 공동 저랭크 잠재 벡터를 저장한다 [1, p.7] | 원격 바이트 주소형 메모리, RDMA, 내부 프리페치와 다계층 DMA를 사용한다 [2, p.2] [2, p.8] |
| 저자 보고 효과 | MHA 대비 KV cache가 소형 MoE에서 14%, 대형 MoE에서 4%라고 보고했다 [1, p.31] [1, p.32] | NVMe-oF 기준선 대비 처리량 1.80배 향상을 보고했다 [2, p.2] |
| 전제조건 | MLA 구조 모델·가중치 및 이를 처리하는 추론 구현이 필요하다 [1, p.28] | CXL-hybrid memory, RDMA, 프리페치 제어 경로가 필요하다 [2, p.2] |
| 한계 | 제공 근거에 종단 간 처리량·TTFT·운영 안정성 측정값이 없다 [1, p.6] [1, p.31] | I/O 경합 시 지연 청크의 재계산이 발생할 수 있다 [2, p.10] |

#### MLA
MLA는 추론 시 모든 K·V를 저장하는 MHA의 KV cache 부담을 줄이기 위해 K·V 공동 저랭크 압축을 사용한다 [1, p.6] [1, p.7]. 저자 보고 기준, 비교 모델에서 KV cache 요소 수와 일반 벤치마크 결과가 함께 제시되지만, 실제 데이터센터 서빙의 지연시간·처리량 수치는 제공되지 않았다 [1, p.31] [1, p.32].

#### ITME
ITME는 예측 가능한 가중치와 prefix KV cache 접근을 이용해 SSD에서 내부 DRAM cache로, 이어 RDMA·호스트 메모리를 거쳐 GPU로 데이터를 미리 이동하는 구조다 [2, p.2] [2, p.8]. 저자 보고 처리량 결과는 있으나, 원격 계층의 읽기·쓰기 경합과 재계산 조건, 품질 보존 검증 공백을 함께 고려해야 한다 [2, p.10] [2, p.11].

## 4. 관점별 평가
### 4.1 기술 성숙도(TRL)
| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |
|---|---|---|
| DeepSeek-V2 MLA | 기술 자체: TRL 7. 제공 웹 근거에서 DeepSeek의 공식 FlashMLA 저장소는 해당 최적화 어텐션 커널이 DeepSeek-V3 및 DeepSeek-V3.2-Exp 모델을 구동한다고 명시한다. 이는 MLA 자체가 모델 서빙용 커널/백엔드에 통합된 운영 근거로 해석할 수 있다. 다만 이 근거만으로 외부 고객 대상 API의 SLA, 배포 규모, 독립 운영 재현까지는 확인되지 않는다 ([W8]). 계열: TRL 7. MLA 계열은 DeepSeek-V3 및 DeepSeek-V3.2-Exp를 구동하는 최적화 커널의 공개 통합 근거가 있다 ([W8]). 다만 계열 전반의 다중 공급자 채택 또는 독립 재현은 제공 근거에서 확인되지 않는다. | 기술 자체 판정 기준: 문헌만으로는 구현·검증 수준인 TRL 1-6만 뒷받침할 수 있으나, 공식 GitHub 저장소가 FlashMLA를 DeepSeek-V3 및 DeepSeek-V3.2-Exp를 구동하는 최적화 어텐션 커널이라고 명시한다. 이에 따라 MLA 자체에 대해 모델 서빙 커널 통합·운영 채택이 확인된 것으로 보아 TRL 7로 추정했다 ([W8]). 이는 해당 저장소의 진술에 근거한 판정이며, 상용 API 고객 규모나 독립 사업자의 운영 채택을 뜻하지 않는다. 논문 구현·검증 근거: MLA는 저랭크 K·V 공동 압축으로 추론 시 KV cache 병목을 줄이도록 설계되었고 [1, p.6] [1, p.7], MHA와 비교한 KV cache 및 벤치마크 결과가 보고되었다 [1, p.31] [1, p.32]. |
| ITME | 기술 자체: TRL 4-6. FPGA 기반 프로토타입과 실제 시스템 환경에서의 하드웨어 제어 로직·소프트웨어 프리페처 검증이 보고되었고, CMM 기반 평가 플랫폼 및 FPGA 프로토타입의 성능을 비교하였다 [2, p.8] [2, p.10] [2, p.11]. 그러나 ITME 자체의 제품 출시, 실제 서비스 운영, 외부 고객 채택을 문서화한 웹 근거는 제공되지 않았다. 계열: TRL 4-6. 제공 근거는 CXL-hybrid-memory를 이용한 ITME 프로토타입 검증과 상용 SK Hynix CMM·PCIe Gen5 SSD를 사용한 평가를 보여 준다 [2, p.11]. 그러나 제공된 웹 자료는 ITME가 속한 CXL 메모리 계열의 제품 출시 또는 상용 서비스 운용을 직접 문서화하지 않으므로, 계열에 TRL 7-9를 부여할 근거는 부족하다. 이는 ITME 개별 기술의 미상용화와 CXL 계열 일반의 시장 부재를 동일시하는 판단은 아니다. | 기술 자체 판정 기준: ITME는 CMM 기반 성능 평가와 FPGA 기반 프로토타입으로 하드웨어 실현 가능성을 검증한 단계이다 [2, p.8] [2, p.10] [2, p.11]. 이 근거는 통합 구현·유사 운용 환경 검증에 해당하므로 TRL 4-6으로 추정했다. ITME 개별 기술의 상용 제품 출시, 실제 서비스 운영, 외부 고객 채택은 제공 PDF 및 웹 자료에서 확인되지 않는다. 계열 판정 기준: 상용 SK Hynix CMM 및 PCIe Gen5 SSD가 ITME 평가에 사용된 사실은 확인되지만 [2, p.11], 이는 ITME의 상용화 증거가 아니다. 또한 제공 웹 자료에는 CXL 메모리 제품 자체의 출시·상용 운영을 직접 확인하는 자료가 없다. 따라서 CXL-hybrid-memory 계열의 TRL 7-9 판정도 보류했다. |

공개 정보에 한정한 추정이다. TRL은 기술 성숙도를 1~9 단계로 표현하는 척도이며, 여기서 TRL 7 및 TRL 4-6은 표의 공개 근거 수준을 요약하는 용도로만 사용했다 [W20].

#### DeepSeek-V2 MLA
기술 자체와 MLA 계열 모두 TRL 7 추정의 직접 근거는 DeepSeek 공식 FlashMLA 저장소가 해당 최적화 커널이 DeepSeek-V3 및 DeepSeek-V3.2-Exp 모델을 구동한다고 밝힌 점이다 [W8]. 즉, 공개된 모델 서빙 커널 통합·구동 진술은 확인된다. 다만 이는 DeepSeek-V2 MLA의 외부 고객 SLA, 배포 규모 또는 독립 재현을 확인하는 자료는 아니다. 논문은 MLA의 저랭크 K·V 공동 압축과 비교 벤치마크를 제시한다 [1, p.6] [1, p.31].

#### ITME
ITME 자체와 CXL-hybrid-memory 계열은 TRL 4-6으로 추정했다. 근거 수준은 CMM 기반 평가와 FPGA 프로토타입을 통한 하드웨어 제어 로직·소프트웨어 프리페처 검증이다 [2, p.8] [2, p.10] [2, p.11]. 상용 SK hynix CMM과 PCIe Gen5 SSD의 평가 사용은 확인되지만, 이는 ITME의 제품 출시 증거가 아니다 [2, p.11]. 제공 자료로는 ITME 개별 기술의 외부 고객 채택, 운영 규모, 독립 재현은 확인되지 않는다. CXL 제품군 일반의 성숙도와 ITME 개별 설계의 성숙도를 동일시하지 않았다.

### 4.2 시장성
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 시장 규모·성장성 | 긍정 | 혼재 |
| 상용화·채택 현황 | 긍정 | 혼재 |
| 생태계 지지 | 긍정 | 혼재 |
| 종합 | 근거 부족 | 근거 부족 |
| 근거 | [W29] [W27] [W3] [W17] [W4] [W5] [W26] | [W23] [W22] [W13] [W28] [W2] |

한계: 표의 MLA ‘상용화·채택 현황: 긍정’은 DeepSeek R1/V3의 관측·호스팅 지원을 근거로 한 간접 신호일 뿐, DeepSeek-V2 MLA 기능 자체의 고객 채택을 직접 확인하지 못하므로 해당 판정의 직접성을 제한한다.

#### DeepSeek-V2 MLA
시장 규모·성장성의 `긍정`은 LLM 시장 전망 및 MLOps·LLMOps 범위에 서빙·추론 최적화가 포함된다는 상위 시장 신호에 근거한다 [W29] [W27]. 이는 MLA 전용 시장 규모·점유율·성장률은 아니다. 상용화·채택의 `긍정`은 Amazon Bedrock의 DeepSeek R1/V3 관측 지원과 방화벽 뒤 vLLM 클러스터 호스팅 언급에 근거한 모델·호스팅 신호다 [W3] [W17]. MLA 기능 자체의 고객 채택 증거로 해석할 수 없다. 생태계 지지의 `긍정`은 vLLM의 MLA backend와 SGLang의 MLA 최적화 구현에 근거한다 [W4] [W5] [W26]. 다만 세 하위 판정 모두 반대 방향의 시장 근거는 제공 자료에서 확인되지 않았으며, 종합은 `근거 부족`으로 유지한다.

#### ITME
`혼재` 신호는 ITME 자체가 아니라 CXL 계열 자료에서 온다. CXL 메모리 풀링·공유의 배포 확대 언급, CXL 제품 전시 및 메모리 확장 전망은 긍정 신호다 [W2] [W13] [W23]. 반면 CXL 시장 자료는 AI 추론·KV cache 수요의 성장 전망을 제시하면서도 ITME 개별 기술의 출시·고객 운영을 확인하지 않는다 [W28]. 따라서 ITME의 시장성 종합은 `근거 부족`이며, CXL 계열 신호를 ITME 채택으로 확대하지 않는다.

### 4.3 이해관계자
| 평가 대상 | DeepSeek-V2 MLA | ITME |
|---|---|---|
| 경쟁 진영 | 근거 부족 | 근거 부족 |
| 도입 기업·개발자 | 혼재 | 근거 부족 |
| 투자 업계 | 근거 부족 | 혼재 |
| 종합 | 근거 부족 | 혼재 |
| 근거 | [W16] [W14] [W15] | [W7] [W31] [W24] |

#### DeepSeek-V2 MLA
개발자·도입 기업 관점은 `혼재`다. AMD는 MLA decode가 지연을 누적시키는 병목이 될 수 있다고 설명하는 한편, AITER 커널이 vLLM·SGLang에 통합돼 AMD Instinct GPU에서 가속을 제공한다고 안내한다 [W16]. 개발자 커뮤니티에서는 특정 환경에서 SGLang의 DeepSeek 지원이 vLLM보다 더 빠르다는 경험담이 제시됐지만, 이는 독립 벤치마크가 아닌 개인 의견이다 [W14]. 경쟁 진영 및 투자 업계가 MLA 자체를 명시적으로 평가한 직접 자료는 부족하다. SemiAnalysis의 KV cache 절감 설명은 분석 의견이지 투자 판단의 직접 근거는 아니다 [W15]. 따라서 종합은 `근거 부족`이다.

#### ITME
ITME 자체에 대한 경쟁사·운영자 직접 발언은 확인되지 않아 해당 항목은 `근거 부족`이다. 투자 업계의 `혼재`는 CXL 메모리 계열에 대한 간접 반응이다. Morgan Stanley 전망을 인용한 보도는 메모리 지출 비중 확대 신호를 전하지만 [W7], Gartner 및 Greyhound Research 인용 발언은 CXL 메모리 풀링이 단기 구조적 제약을 제거하지 못할 수 있다고 본다 [W31]. 운영자는 CPU·BIOS·펌웨어·스위치 상호운용성, 소프트웨어·관리 도구, 실부하 성능을 확인해야 한다는 일반 CXL 도입 조언도 제시된다 [W24]. 이는 ITME 개별 기술의 사용자 경험이 아니다.

### 4.4 도메인 적용성(D1-D7)
| 평가항목 | MLA 판정 | MLA 근거 | ITME 판정 | ITME 근거 |
|---|---|---|---|---|
| D1 워크로드 수용 능력 | 조건부 | [1, p.4] [1, p.32] | 조건부 | [2, p.9] [2, p.10] [2, p.2] |
| D2 서비스 성능 | 근거 부족 | [1, p.6] [1, p.5] | 조건부 | [2, p.10] [2, p.2] |
| D3 메모리 자원 효율 | 조건부 | [1, p.8] [1, p.32] | 조건부 | [2, p.2] [2, p.9] [2, p.8] |
| D4 품질·정보 보존 | 조건부 | [1, p.31] [1, p.32] | 근거 부족 | [2, p.1] [2, p.3] |
| D5 운영 안정성 | 근거 부족 | [1, p.6] [1, p.31] | 조건부 | [2, p.9] [2, p.10] |
| D6 도입·확장 용이성 | 조건부 | [1, p.28] [1, p.31] | 조건부 | [2, p.2] [2, p.8] |
| D7 비용 효율성 | 조건부 | [1, p.32] [1, p.6] | 조건부 | [2, p.10] [2, p.8] |

#### DeepSeek-V2 MLA
D1·D3·D4·D6·D7은 `조건부`다. 저자 보고 기준 MLA는 MHA 대비 KV cache 요소 수를 줄였고, 비교 표에는 일반 벤치마크 결과가 제시된다 [1, p.31] [1, p.32]. 이는 메모리 여지와 품질 유지 가능성의 근거지만, 실제 HBM 점유량·동시 요청 수·총비용은 확인되지 않는다. MLA는 별도 어텐션 구조로 문서화돼 있어 도입 시 MLA 모델·가중치와 대응 추론 구현의 호환성을 확인해야 한다 [1, p.28]. D2 서비스 성능과 D5 운영 안정성은 종단 간 처리량·TTFT·경합·장기 운영 측정이 없어 `근거 부족`이다 [1, p.6] [1, p.31].

#### ITME
D1·D2·D3·D5·D6·D7은 `조건부`다. 저자 보고 기준 ITME는 장기 다중 대화 평가에서 원격 CXL-hybrid memory를 활용하고, NVMe-oF 기준선 대비 처리량 향상을 제시했다 [2, p.9] [2, p.2]. 그러나 턴 증가에 따른 I/O 경합은 일부 청크 재계산을 유발할 수 있다 [2, p.10]. CXL-hybrid memory·RDMA·프리페치 API 및 관련 시스템 통합이 필요하다 [2, p.2] [2, p.8]. D4는 출력 품질·장문맥 정보 보존을 직접 측정한 자료가 없어 `근거 부족`이다 [2, p.1] [2, p.3].

## 5. 종합 의견
| 관점 | MLA | ITME |
|---|---|---|
| TRL | TRL 7 추정: 공개 서빙 커널 통합·구동 진술 [W8] | TRL 4-6 추정: 프로토타입·평가 근거 [2, p.10] [2, p.11] |
| 시장 | 종합 `근거 부족`: 생태계 지원은 확인되나 고객 채택은 간접 근거 [W3] [W4] | 종합 `근거 부족`: CXL 계열 신호는 있으나 ITME 직접 근거는 부족 [W2] [W13] |
| 이해관계자 | 종합 `근거 부족`: 개발자 관점은 `혼재` [W14] [W16] | 종합 `혼재`: 투자자 관점은 CXL 계열 간접 신호 [W7] [W31] |
| 도메인 | `조건부`: 메모리 절감 근거와 운영 성능 공백 병존 [1, p.31] [1, p.32] | `조건부`: 용량·처리량 저자 보고와 경합 조건 병존 [2, p.2] [2, p.10] |

관점 간 일치는 MLA의 KV 표현 축소와 ITME의 계층형 메모리 확장이라는 작동 원리의 확인에 있다 [1, p.7] [2, p.2]. 다만 시장·이해관계자 관점은 각각 MLA 기능 및 ITME 개별 기술에 대한 직접 채택 자료가 부족해 기술·도메인 근거보다 약하다.

| 기술 | 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 |
|---|---|---|---|---|
| MLA | 커널 통합과 서빙 성능 | TRL 7 추정 [W8] | D2 `근거 부족` [1, p.6] | 커널 구동 진술은 종단 간 처리량·TTFT를 측정하지 않는다. |
| MLA | 생태계와 구현 편차 | 시장 생태계 `긍정` [W4] [W26] | 개발자·도입 기업 `혼재` [W14] [W16] | 지원 존재와 환경별 성능 일관성은 다른 주장이다. |
| ITME | CXL 계열과 개별 ITME | 시장 하위 신호 `혼재` [W2] [W13] | TRL 4-6 추정 [2, p.10] | CXL 계열 제품·시장 신호는 ITME 프로토타입의 상용화 증거가 아니다. |
| ITME | 처리량과 운영 경합 | D2 `조건부` [2, p.2] | D5 `조건부` [2, p.10] | 저자 보고 처리량과 I/O 경합·재계산 조건이 함께 제시된다. |

MLA는 KV 표현량을 줄이고 ITME는 메모리 수용·이동 경로를 확장한다. 따라서 두 접근은 보완 가능한 계층이지만, 실제 결합의 호환성·성능·비용을 검증한 자료는 없다.

## 6. 시사점
관점별 평가는 달라진다. 모델 개발사는 MLA의 메모리 절감 구조와 생태계 구현을 볼 수 있으나, 기존 모델 전환성과 종단 간 성능을 별도 검증해야 한다. 인프라 운영자는 ITME의 용량 확장 가능성을 볼 수 있으나, 원격 계층 경합과 시스템 통합을 별도 검증해야 한다 [1, p.7] [2, p.2] [2, p.10].

| 이해관계자 | 도입 전 확인 과제 |
|---|---|
| 클라우드 사업자 | 동일 장문맥·동시성 조건에서 TTFT, 처리량, tail latency, 요청 실패율 및 장애 복구를 측정한다. MLA의 기능 지원과 고객 채택을 구분하고, ITME의 CXL·RDMA 경합을 측정한다. |
| 모델 개발사 | MLA 모델·가중치와 추론 backend 호환성, 기존 checkpoint 전환·재학습 필요 여부, 장문맥 정보 보존을 확인한다 [1, p.28] [W4]. |
| 메모리·HW 벤더 | CXL-hybrid memory, NIC, SSD, 프리페처 및 관리 경로의 상호운용성과 실부하 성능을 검증한다 [2, p.2] [W24]. |
| 투자자 | MLA의 생태계 신호와 직접 고객 채택을 구분하고, CXL 계열 시장 전망과 ITME 개별 기술의 제품화·운영 근거를 분리해 검토한다 [W3] [W23] [W31]. |

두 기술을 비교할 때는 메모리 절감 요소 수와 원격 계층 처리량을 같은 단위의 우열로 환산하지 말고, 동일 모델·문맥·동시성·인프라·품질 목표에서 별도 측정해야 한다.

## 7. 한계점
**자료 기준 시점**: 웹 자료 검색·검증일 2026-10-07 (Tavily 검색 결과 기준). 이후 공개된 발표·제품 정보는 반영되지 않았다.

모든 TRL 판정은 논문, 특허성 연구 자료, 상용화 발표 등 **공개 정보만으로 한 추정**이다. 발표·논문 시점과 실제 고객 채택 사이에는 시차가 있을 수 있으며, 특히 TRL 4-6 구간의 비공개 통합·운영 정보는 확인하기 어렵다. 논문 수치와 성능은 저자 보고 기준이며 독립 재현으로 간주하지 않았다.

#### 관점별 근거 부족
- **시장성:** MLA 전용 시장 규모·점유율, MLA 기능 자체의 고객 채택, 부정·반대 방향의 시장 근거가 부족하다. 특히 R1/V3 관측·호스팅 지원은 DeepSeek-V2 MLA 기능 자체의 채택을 직접 입증하지 않는다 [W3] [W17]. ITME는 개별 제품 출시·고객 운영·시장 규모가 부족하며, CXL 계열 근거를 ITME로 일반화할 수 없다.
- **이해관계자:** MLA의 경쟁사·투자자 직접 발언과 ITME 자체의 경쟁사·운영자·투자자 직접 발언이 부족하다.
- **도메인:** MLA의 실서빙 처리량·TTFT·운영 안정성, ITME의 품질 보존·절대 지연시간·총소유비용, 그리고 두 기술의 통제된 병행 실험이 부족하다.

확증편향을 줄이기 위해 ITME는 HW 베이스라인 논문으로 교차 확인했고, 두 기술에 동일한 보고 형식과 관점 기준을 적용했으며, 종합은 이미 검증된 Evidence로 제한했다. 그럼에도 출처 범위와 공개 정보 공백 때문에 채택 규모, 비용 및 장기 운영성을 확정할 수 없다.

#### 근거 공백 (Supervisor 기록)
- **시장성** — 근거 부족: 종합 단계에서 추가 근거가 필요하다고 판단함; 4.2 시장성 표에서 MLA의 ‘상용화·채택 현황’을 ‘긍정’으로 판정한 근거 [W3]·[W17]은 DeepSeek R1/V3 모델의 관측·호스팅 지원을 말할 뿐 DeepSeek-V2 MLA 기능 자체의 채택을 직접 뒷받침하지 않는다. 본문도 이를 ‘적용 모델 및 기능 지원 근거’로 묶어 MLA 채택 신호로 확장한다. 또한 4.2의 MLA 시장 규모·성장성; MLA: 판정이 긍정 한쪽뿐(반대 방향 근거 미탐색); 4.2의 MLA ‘시장 규모·성장성’, ‘상용화·채택 현황’, ‘생태계 지지’는 모두 ‘긍정’인데, 7장 ‘MLA: 판정이 긍정 한쪽뿐(반대 방향 근거 미탐색)’이라고 공백을 기록한다. 본문이 간접성·한계를 설명하고 종합을 ‘근거 부족’으로 낮춘 점은 보완적이나, 하위 시장성 판정의 반대 방향 근거 부재는 해소되지 않았다.
- **이해관계자** — 근거 부족: 종합 단계에서 추가 근거가 필요하다고 판단함

#### 표 7. 증거 균형표 (중복 제거 후 수집·검증된 Evidence 수)
| 구분 | DeepSeek-V2 MLA | ITME | HW 베이스라인(InfiniGen·CXL-PNM) |
|---|---|---|---|
| 기술 조사 · 논문 청크 | 12 | 26 | 2 |
| 기술 조사 · 웹 문서 | 12 | 4 | 0 |
| 시장 · 웹 문서 | 7 | 5 | 0 |
| 이해관계자 · 웹 문서 | 3 | 3 | 0 |
| 도메인 · 논문 청크 | 20 | 23 | 0 |
| 합계 | 54 | 61 | 2 |

## REFERENCE
- [1] DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, p.4, p.5, p.6, p.7, p.8, p.28, p.31, p.32.
- [2] Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J. (SK hynix) (2026). ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories. arXiv preprint arXiv:2606.12556, p.1, p.2, p.3, p.8, p.9, p.10, p.11.
- [W8] github.com (Wed, 09 Sep 2026 12:00:00 GMT). GitHub - deepseek-ai/FlashMLA: FlashMLA: Efficient Multi-head Latent Attention Kernels · GitHub. github.com, https://github.com/deepseek-ai/FlashMLA
- [W3] docs.dynatrace.com (Fri, 28 Aug 2026 00:00:00 GMT). AI Observability integrations — Dynatrace Docs. docs.dynatrace.com, https://docs.dynatrace.com/docs/observe/dynatrace-for-ai-observability/integrations
- [W4] docs.vllm.ai (게시일 미확인). mla_attention - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/attention/mla_attention
- [W5] docs.vllm.ai (게시일 미확인). Attention Backend Feature Support - vLLM. docs.vllm.ai, https://docs.vllm.ai/en/stable/design/attention_backends
- [W13] news.skhynix.com (Thu, 01 Oct 2026 23:00:00 GMT). SK hynix Presents a Full Lineup of Memory Solutions Optimized for AI Infrastructure at DTF 2026 – SK hynix Newsroom. news.skhynix.com, https://news.skhynix.com/en/dtf-2026
- [W22] forbes.com (Sat, 18 Jul 2026 00:00:00 GMT). AI Storage & Memory From Backblaze, CoreWeave, Panmnesia, Vast And Cloudera - Forbes. forbes.com, https://www.forbes.com/sites/tomcoughlin/2026/07/18/ai-storage--memory-from-backblaze-coreweave-panmnesia-vast-and-cloudera/
- [W20] esa.int (게시일 미확인). ESA - Technology Readiness Levels (TRL). esa.int, https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/Technology_Readiness_Levels_TRL
- [W29] mordorintelligence.com (Fri, 11 Sep 2026 00:00:00 GMT). Large Language Model Market Size, Growth & Outlook | Industry Report 2031. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/large-language-model-llm-market
- [W27] meticulousresearch.com (Mon, 28 Sep 2026 04:00:00 GMT). MLOps & LLMOps Platforms Market Size & Forecast 2036. meticulousresearch.com, https://www.meticulousresearch.com/product/mlops-llmops-platforms-market-6912
- [W17] switchboard-ai.com (Wed, 23 Sep 2026 00:00:00 GMT). Switchboard — AI platform for customer operations. switchboard-ai.com, https://switchboard-ai.com/platform
- [W26] lmsys.org (게시일 미확인). SGLang v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision - LMSYS Org. lmsys.org, https://www.lmsys.org/blog/2024-09-04-sglang-v0-3
- [W23] forbes.com (Tue, 25 Aug 2026 00:00:00 GMT). CXL Growth Shown At The 2026 FMS Conference - Forbes. forbes.com, https://www.forbes.com/sites/tomcoughlin/2026/08/25/cxl-growth-shown-at-the-2026-fms-conference/
- [W28] mordorintelligence.com (Tue, 28 Jul 2026 00:00:00 GMT). CXL Memory Controller IC Market Size, Share & 2031 Growth Trends Report. mordorintelligence.com, https://www.mordorintelligence.com/industry-reports/cxl-memory-controller-ic-market
- [W2] computeexpresslink.org (게시일 미확인). Breaking Boundaries in Memory: Highlights from AI Infra Summit and SDC 2025 - Compute Express Link. computeexpresslink.org, https://computeexpresslink.org/blog/breaking-boundaries-in-memory-highlights-from-ai-infra-summit-and-sdc-2025-4198
- [W16] rocm.docs.amd.com (게시일 미확인). MLA decoding kernel of the AITER library to accelerate LLM inference — Tutorials for AI developers 5.1. rocm.docs.amd.com, https://rocm.docs.amd.com/projects/ai-developer-hub/en/v5.1/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
- [W14] news.ycombinator.com (게시일 미확인). DeepSeek Open Source FlashMLA – MLA Decoding Kernel for Hopper GPUs | Hacker News. news.ycombinator.com, https://news.ycombinator.com/item?id=43155023
- [W15] newsletter.semianalysis.com (게시일 미확인). DeepSeek Debates: Chinese Leadership On Cost, True Training Cost, Closed Model Margin Impacts. newsletter.semianalysis.com, https://newsletter.semianalysis.com/p/deepseek-debates
- [W7] finance.yahoo.com (게시일 미확인). The AI ‘Memory Wall’ Is About to Get a Lot Taller: These 3 Stocks Will Win Big. finance.yahoo.com, https://finance.yahoo.com/markets/stocks/articles/ai-memory-wall-lot-taller-122127554.html
- [W31] networkworld.com (게시일 미확인). Chip wafer shortage will run through 2030 as AI demand overwhelms supply: SK Hynix chief | Network World. networkworld.com, https://www.networkworld.com/article/4146270/chip-wafer-shortage-will-run-through-2030-as-ai-demand-overwhelms-supply-sk-hynix-chief.html
- [W24] hpcwire.com (게시일 미확인). What Hyperscalers Should Know About CXL. hpcwire.com, https://www.hpcwire.com/2026/08/20/what-hyperscalers-should-know-about-cxl
- [D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 (A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.
