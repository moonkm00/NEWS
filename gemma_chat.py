import google.generativeai as genai
import os

class GemmaChatAgent:
    def __init__(self, api_key):
        """
        Gemma 3 모델을 활용하여 사용자와 팩트체크 대화를 수행하는 클래스
        """
        if not api_key:
            raise ValueError("API Key가 필요합니다.")
        genai.configure(api_key=api_key)
        
        # 최속/최고 효율 모델인 Gemma 3 4B 사용
        self.model = genai.GenerativeModel('gemma-3-4b-it')
        self.chat_session = None

    def start_conversation(self, context_text, detection_result):
        """
        기사 본문과 탐지 결과를 바탕으로 대화 시작 (시스템 프롬프트 설정 역할)
        """
        system_instruction = f"""
        당신은 '팩트체크 보조 전문가' Gemma입니다. 
        당신의 역할은 사용자가 입력한 기사의 신뢰도 분석 결과를 바탕으로 대화를 나누는 것입니다.

        [분석 정보]
        - 분석 대상 기사: {context_text}
        - 판별 결과: {detection_result['status']} ({detection_result['confidence']:.1f}% 신뢰도)
        - 분석 요약: {detection_result['explanation']}

        [대화 원칙]
        1. 공손하고 친절하며 분석적인 태도를 유지하세요.
        2. 사용자가 기사의 특정 부분에 대해 물어보면 뉴스비평가 및 데이터 과학자의 관점에서 논리적으로 답변하세요.
        3. 단순한 결과(참/거짓)를 넘어서, 왜 인공지능이 그렇게 판단했는지(문체, 단어 선택, 논리적 비약 등)를 함께 추론하여 해설하세요.
        4. 사용자가 불안해하거나 의심스러워하는 부분에 대해 팩트체크 가이드를 제공하세요.
        5. 반드시 한국어로 답변하세요.
        """
        
        # Gemma 3는 대화 모드를 지원합니다.
        self.chat_session = self.model.start_chat(history=[])
        # 첫 메시지로 맥락 주입 (사용자에게는 보이지 않게 처리하거나 첫 인사로 활용)
        initial_prompt = f"{system_instruction}\n\n위 정보를 숙지했습니다. 인사와 함께 대화를 시작하세요."
        response = self.chat_session.send_message(initial_prompt)
        return response.text

    def send_message(self, message):
        """
        사용자 메시지를 보내고 답변 수신
        """
        if self.chat_session is None:
            return "분석이 먼저 완료되어야 챗봇과 대화할 수 있습니다."
        
        response = self.chat_session.send_message(message)
        return response.text
