# 💬 Gemma 3 팩트체크 상담 엔진 분석 보고서

> **본 문서는 `gemma_chat.py`에 구현된 대화형 AI 상담 시스템의 구조와 작동 원리를 상세히 설명합니다.**

---

## 1. 개요 (Overview)
*   **사용 모델**: `gemma-3-4b-it` (Google Open Model)
*   **목표**: 뉴스 분석 리포트 확인 후 발생하는 사용자의 의문점을 실시간 대화로 해결하고, 심층적인 팩트체크 가이드를 제공함.

---

## 2. 핵심 로직 분석 (Core Logic)

### 🧠 맥락 주입 (Context Injection)
단순한 챗봇과 달리, `GemmaChatAgent`는 대화 시작 시 **현재 분석 중인 상황을 완벽히 학습**합니다.

*   **주입되는 데이터**:
    *   기사 본문 전체
    *   BERT 모델의 판별 결과 (Fake/Real) 및 신뢰도 수치
    *   Gemini가 생성한 전문 해설 리포트 요약
*   **기대 효과**: 사용자가 "이 결과의 근거가 뭐야?"라고 물었을 때, 앞선 분석 내용을 바탕으로 일관성 있는 답변 가능.

### 🗨️ 세션 관리 (Session Management)
*   `start_chat(history=[])`를 통해 독립적인 대화 세션을 생성합니다.
*   연속적인 질문(Multi-turn)에도 이전 대화 내용을 기억하여 자연스러운 상담이 가능합니다.

---

## 3. 핵심 코드 상세 분석 (Code Deep Dive)

이 시스템의 두뇌 역할을 하는 `GemmaChatAgent` 클래스의 주요 로직을 분석합니다.

### 🔌 초기화 및 엔진 로드 (`__init__`)
```python
def __init__(self, api_key):
    genai.configure(api_key=api_key)
    # 팩트체크 상담에 최적화된 Gemma 3 지시 튜닝(IT) 모델 로드
    self.model = genai.GenerativeModel('gemma-3-4b-it')
```
*   **역할**: Google Gemini API 설정을 완료하고, 대화 능력이 극대화된 `gemma-3-4b-it` 모델을 인스턴스화합니다. 상담 세션(`chat_session`)은 초기에는 비워둡니다.

### 🧠 맥락 주입 및 세션 활성화 (`start_conversation`)
가장 핵심적인 부분으로, 챗봇에게 분석 상황을 완벽히 인지시킵니다.
```python
def start_conversation(self, context_text, detection_result):
    # (1) 시스템 프롬프트(Persona) 정의
    system_instruction = f"당신은 '팩트체크 보조 전문가' Gemma입니다..."
    
    # (2) 대화 세션 생성 (이전 기록이 없는 깨끗한 상태로 시작)
    self.chat_session = self.model.start_chat(history=[])
    
    # (3) 분석 정보를 처음부터 알고 시작하도록 맥락 주입
    initial_prompt = f"{system_instruction}\n\n[분석정보] {context_text}..."
    return self.chat_session.send_message(initial_prompt).text
```
*   **역할**: 
    1.  **Persona**: 챗봇에게 '비평 전문가'라는 인격을 부여합니다.
    2.  **Context Injection**: 분석된 뉴스 내용과 판별 결과를 **처음부터 알고 대화를 시작**하게 함으로써 "왜?"라는 질문에 정확히 답할 수 있는 기반을 마련합니다.

### 🗨️ 실시간 응답 유지 (`send_message`)
```python
def send_message(self, message):
    # 기존 대화 흐름(history)을 유지하며 사용자 메시지에 응답
    response = self.chat_session.send_message(message)
    return response.text
```
*   **역할**: 사용자가 입력한 채팅 메시지를 처리하며, `chat_session`이 살아있어 이전 대화 내용을 모두 기억한 채 답변을 이어갑니다.

---

## 4. 챗봇의 주요 특성 (Key Features)

| 특성 | 설명 |
| :--- | :--- |
| **전문성** | 뉴스 비평가 및 데이터 과학자의 관점에서 논리적으로 답변함 |
| **추론 능력** | 인공지능이 왜 그렇게 판단했는지(단어 선택, 문체 등)를 추론하여 해설함 |
| **교육적 가치** | 독자가 뉴스에 대해 비판적 사고를 가질 수 있도록 팩트체크 요령을 안내함 |
| **응답속도** | 4B 경량 모델을 사용하여 실시간에 가까운 빠른 답변 제공 |

---

## 5. 시스템 통합 워크플로우 (Integration)

1.  **Phase 1 (Detection)**: BERT 모델이 뉴스의 진위 확률을 계산.
2.  **Phase 2 (Explanation)**: Gemini 1.5가 분석 리포트 초안 작성.
3.  **Phase 3 (Consulting)**: **Gemma 3**가 위 모든 정보를 바탕으로 사용자와 1:1 상담 수행.

---
*Created by Kwangmyung Moon / AI-Lawyer Project Team*
