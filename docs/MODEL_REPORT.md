# 📊 모델 개발 보고서: 낚시성 기사 탐지 (Clickbait Detector)

> **"제목과 본문의 시맨틱 정렬을 통한 고성능 뉴스 신뢰도 분석"**

---

## 1. 모델 개발 과정 (Model Development Process)

### 📈 데이터셋 구성 (Dataset Composition)
*   **분류 클래스 (Classes)**: 2가지
    *   **정상 기사 (Normal)**: 제목이 본문 내용을 충실히 반영하는 기사
    *   **낚시성 기사 (Clickbait)**: 클릭 유도를 위해 내용을 왜곡하거나 과장한 제목의 기사
*   **데이터 출처**: 낚시성 기사 탐지 데이터 (146번 데이터셋)
*   **데이터 규모**: 고효율 학습을 위한 20% 샘플링 데이터셋 (`processed_data_20pct.parquet`)

### ⌨️ 텍스트 전처리 (Text Preprocessing)
*   **토크나이저**: `klue/bert-base` 전용 워드피스(WordPiece) 토크나이저
*   **길이 정규화**: 최대 **512 토큰** (Truncation & Dynamic Padding 적용)
*   **데이터 결합**: `[CLS] 제목 [SEP] 본문 [SEP]` 구조의 문장 쌍(Sentence Pair) 입력 방식 사용
*   **배치 처리**: GPU 메모리 최적화를 위한 배치 사이즈 16 적용

### 🤖 모델 로드 및 학습 (Model Load & Training)
*   **최종 모델**: `clickbait_detector_final`
*   **프레임워크**: Hugging Face **Transformers & Trainer** API
*   **학습 환경**: PyTorch 기반 CUDA 가속 학습 (fp16 적용)
*   **최고 성능 지표 (Best Metrics)**:
    *   **검증 정확도(Accuracy)**: `99.04%`
    *   **F1-Score**: `97.68%`

---

## 2. 모델 아키텍처 특징 (Architecture Features)

### 🌟 KLUE-BERT (Korean Language Understanding Evaluation BERT)
![KLUE-BERT Architecture](klue_bert_architecture.png)
*   **Deep Neural Network**: 12계층의 깊은 트랜스포머 인코더 구조
*   **한국어 특화**: 뉴스, 위키, 구어체 등 한국어 데이터셋으로 사전 학습되어 문맥 파악 능력이 우수함
*   **어텐션 메커니즘**: 제목과 본문 간의 관계(Cross-Attention)를 분석하여 낚시성 패턴 효과적 탐지
*   **즉각적인 예측**: 사전 학습된 가중치를 활용하여 미세 조정(Fine-tuning) 후 즉시 실서비스 투입 가능

### 🔍 아키텍처 상세 설명 (Graph Details)
*   **입력 레이어 (Title & Content)**: 제목과 본문을 `[SEP]` 토큰으로 결합하여 입력하며, `[CLS]` 토큰이 문맥의 함축적 요약을 담당합니다.
*   **12 레이어 인코더**: 깊은 신경망 구조를 통해 단어 간의 연관성을 다각도로 분석(Self-Attention)합니다.
*   **문맥 교차 분석**: 제목의 자극적인 단어와 본문의 실제 정보 간의 괴리를 포착하여 낚시성을 판별합니다.
*   **최종 분류 (Binary Header)**: `[CLS]` 백터의 정보를 바탕으로 정상(Normal)과 낚시성(Clickbait)을 최종 결정합니다.

---
*Created by Kwangmyung Moon*
