import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NewsDetector:
    def __init__(self, model_path="./clickbait_detector_final"):
        """
        학습된 탐지 모델을 로드하여 예측을 수행하는 클래스
        """
        print(f"\n[AI Detector] 로컬 딥러닝 모델({model_path}) 로드 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        print("[AI Detector] 모델 로드 완료! 분석 준비가 되었습니다.\n")

    def predict(self, title, content):
        """
        제목과 본문을 입력받아 [SEP]로 구분하여 결합 후 확률과 레이블을 반환
        """
        inputs = self.tokenizer(title, content, return_tensors="pt", truncation=True, max_length=256, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
        # label 0: Fake (Clickbait), label 1: Real (Normal)
        fake_prob = probs[0][0].item()
        is_fake = fake_prob > 0.5
        
        return is_fake, fake_prob
