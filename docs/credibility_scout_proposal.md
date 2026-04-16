# 프로젝트 제안서: Credibility Scout (가짜 뉴스 및 선동 탐지기)

## 1. 프로젝트 개요 (Overview)
**Credibility Scout**는 정보 과잉 시대에 사용자가 접하는 뉴스 리포트나 커뮤니티 게시글의 신뢰도를 데이터 기반으로 분석하는 **지형 지능형 뉴스 분석 플랫폼**입니다. 단순히 '진실/거짓'을 가리는 것을 넘어, 텍스트 내의 **논리적 오류, 감정적 조작, 정치적 편향성**을 딥러닝 기술로 정밀 분석하여 사용자에게 비판적 사고의 근거를 제공합니다.

---

## 2. 핵심 해결 과제 (Core Challenges)
1.  **교묘한 선동(Loaded Language):** 명백한 거짓은 아니지만, 특정 감정(공포, 분노)을 자극하여 판단을 흐리게 하는 수사법 탐지.
2.  **논리적 허점(Logical Fallacies):** 결론과 무관한 근거를 대거나 인신공격성 발언으로 논점을 흐리는 기법 식별.
3.  **정보의 불균형(Bias):** 특정 관점의 정보만 선택적으로 제공하여 발생하는 확증 편향 방지.

---

## 3. 시스템 아키텍처 (Hybrid Architecture)
본 프로젝트는 속도와 효율성을 위한 **SLM(Small Language Model)**과 복잡한 추론을 위한 **LLM(Large Language Model)**을 결합한 하이브리드 구조를 채택합니다.

- **Fast Track (SLM):** BERT/DeBERTa를 활용한 감정 및 논리적 오류 전처리 분석.
- **Deep Track (LLM + RAG):** 외부 팩트 체크 소스 검색 및 최종 논리적 마감 리포트 생성.

---

## 4. 딥러닝 모델 학습 및 고도화 전략
프로젝트의 기술적 차별화를 위해 다음의 모델들을 직접 파인튜닝(Fine-tuning)합니다.

- **논리적 오류 분류기:** `LogiLogi` 데이터셋 등을 활용해 기만적 논증 식별.
- **감정 분석 모델:** `GoEmotions` 데이터를 활용해 자극적인 어조 수치화.
- **RAG 리랭커:** 주장과 증거 간의 밀접한 연관성을 판단하는 교차 인코더(Cross-Encoder) 학습.

---

## 5. 기술 스택 (Tech Stack)
- **Frontend:** Next.js, Tailwind CSS
- **Backend:** FastAPI, Docker
- **AI/ML:** PyTorch, Hugging Face, LangChain
- **DB:** Supabase (pgvector 포함)

---

## 6. 향후 확장 계획 (Roadmap)
1.  **MVP 개발:** 텍스트 분석 엔진 구축.
2.  **RAG 통합:** 실시간 뉴스 검색 및 검증 기능.
3.  **서비스화:** 크롬 익스텐션 및 대시보드 배포.
