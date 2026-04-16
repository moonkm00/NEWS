import google.generativeai as genai
import os

class NewsExplainer:
    def __init__(self, api_key):
        """
        Gemini API를 활용하여 가짜 뉴스 판별 결과에 대한 해설을 생성하는 클래스
        """
        if not api_key:
            raise ValueError("Gemini API Key가 필요합니다.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-flash-latest')

    def generate_explanation(self, text, is_fake, confidence):
        """
        뉴스 본문과 판별 결과를 바탕으로 해설 리포트 생성
        """
        status = "가짜 뉴스(Fake)" if is_fake else "진짜 뉴스(Real)"
        
        prompt = f"""
        당신은 뉴스 팩트체크 전문가이자 비평가입니다. 
        인공지능 판별 모델이 다음 뉴스를 분석한 결과, 이 뉴스는 {confidence:.1f}%의 확률로 [{status}]라고 판단했습니다.
        
        [뉴스 본문]
        {text}
        
        [수행 작업]
        1. 이 뉴스가 {status}로 판단된 이유를 뉴스 문체, 논리적 흐름, 내용의 현실성 측면에서 분석하세요.
        2. 만약 가짜 뉴스로 판단되었다면, 어떤 부분이 전형적인 선동이나 조작의 특징을 보이는지 지적하세요.
        3. 진짜 뉴스로 판단되었다면, 어떤 점이 신뢰도를 높이는지 설명하세요.
        4. 독자가 이 뉴스를 접할 때 주의해야 할 점을 한 줄로 요약하세요.
        
        결과는 일반 독자가 이해하기 쉽게 친절하면서도 전문적인 어조로 작성해주세요.
        출력은 반드시 한국어로 해주세요.
        """
        
        response = self.model.generate_content(prompt)
        return response.text

    def analyze_news_from_scratch(self, text):
        """
        로컬 모델 없이 Gemini가 직접 탐지와 해설을 동시에 수행
        """
        prompt = f"""
        당신은 뉴스 팩트체크 전문가입니다. 다음 뉴스 기사를 분석하여 진짜 뉴스인지 가짜 뉴스인지 판별하고 그 이유를 설명하세요.
        
        [뉴스 본문]
        {text}
        
        [출력 양식 (JSON 형식을 지켜주세요)]
        {{
            "is_fake": true 또는 false,
            "confidence": 0~100 사이의 숫자,
            "explanation": "상세한 분석 및 해설 내용"
        }}
        
        결과는 반드시 한국어로 작성하세요.
        """
        
        response = self.model.generate_content(prompt)
        # 텍스트에서 JSON 부분만 추출하여 파싱 (간단하게 처리)
        import json
        import re
        
        try:
            # ```json ... ``` 블록이나 단순 JSON 텍스트 추출
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return result
            else:
                return {"is_fake": False, "confidence": 50, "explanation": "분석 실패: 형식을 맞추지 못했습니다."}
        except Exception as e:
            # 에러 발생 시 상세 정보 반환
            return {
                "is_fake": False, 
                "confidence": 0, 
                "explanation": f"분석 중 오류 발생: {str(e)}\n모델의 응답: {response.text[:200]}..."
            }
