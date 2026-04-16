import streamlit as st
import os
from dotenv import load_dotenv
from detector import NewsDetector
from explainer import NewsExplainer
from gemma_chat import GemmaChatAgent

# 환경 변수 로드 및 터미널 로그
load_dotenv()
print("\n[AI Detector System] 애플리케이션 초기화 중...")

# 페이지 설정
st.set_page_config(page_title="AI 가짜 뉴스 탐지기", page_icon="🕵️", layout="wide")

st.title("🕵️ 하이브리드 가짜 뉴스 탐지기")
print("[AI Detector System] UI 렌더링 중...")
st.markdown("""
이 도구는 **KcELECTRA** 모델로 뉴스의 신뢰도를 실시간 분석하고, 
**Gemini 1.5**와 **Gemma 3**를 통해 분석 리포트 및 대형 언어 모델과의 대화 기능을 제공하는 하이브리드 AI 시스템입니다.
""")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "gemma_agent" not in st.session_state:
    st.session_state.gemma_agent = None

# 사이드바 네비게이션
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/search-property.png", width=80)
    st.header("메뉴")
    menu = st.sidebar.radio(
        "이동할 페이지 선택",
        ["🕵️ 뉴스 신뢰도 분석", "📊 학습 데이터 시각화"]
    )
    
    st.divider()
    # API 키 상태 확인 (UI에서는 입력창 제거)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        st.success("API Key 로드 완료")
    else:
        st.error("API Key 미설정 (.env 확인)")

# 모델 로드 (캐싱)
@st.cache_resource
def load_models():
    model_path = "./clickbait_detector_final"
    if not os.path.exists(model_path):
        return None
    return NewsDetector(model_path)

detector = load_models()
is_demo_mode = detector is None

# --- 뉴스 분석 페이지 ---
def show_analysis_page():
    st.subheader("뉴스 기사 입력")
    input_title = st.text_input("뉴스 제목", placeholder="제목을 입력하세요...")
    input_content = st.text_area("뉴스 본문", height=200, placeholder="본문 내용을 입력하세요...")

    if is_demo_mode:
        st.info("💡 **데모 모드 실행 중:** 현재 로컬 딥러닝 모델이 없습니다. Gemini AI가 탐지와 해설을 모두 수행합니다.")

    tab1, tab2 = st.tabs(["📊 분석 결과", "💬 AI 챗봇 상담"])

    with tab1:
        if st.button("분석 시작"):
            if not input_title or not input_content:
                st.warning("분석할 제목과 본문을 모두 입력해 주세요.")
            elif not gemini_key:
                st.warning(".env 파일에 Gemini API Key를 설정해 주세요.")
            else:
                with st.spinner("AI가 기사를 분석 중입니다..."):
                    try:
                        full_text = f"{input_title} [SEP] {input_content}"
                        explainer = NewsExplainer(gemini_key)
                        
                        if not is_demo_mode:
                            is_fake, fake_prob = detector.predict(input_title, input_content)
                            confidence = (fake_prob if is_fake else (1 - fake_prob)) * 100
                            explanation = explainer.generate_explanation(full_text, is_fake, confidence)
                        else:
                            result = explainer.analyze_news_from_scratch(full_text)
                            is_fake = result['is_fake']
                            confidence = result['confidence']
                            explanation = result['explanation']

                        st.session_state.analysis_result = {
                            "is_fake": is_fake,
                            "confidence": confidence,
                            "explanation": explanation,
                            "status": "가짜 뉴스(Fake)" if is_fake else "진짜 뉴스(Real)"
                        }
                        
                        st.success("✅ 분석 완료!")
                        st.session_state.gemma_agent = GemmaChatAgent(gemini_key)
                        first_msg = st.session_state.gemma_agent.start_conversation(
                            full_text, st.session_state.analysis_result
                        )
                        st.session_state.messages = [{"role": "assistant", "content": first_msg}]

                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {e}")

        if st.session_state.analysis_result:
            res = st.session_state.analysis_result
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("탐지 결과")
                if res['is_fake']:
                    st.error(f"🚩 {res['status']}로 의심됩니다.")
                else:
                    st.success(f"✅ {res['status']}일 가능성이 높습니다.")
                st.metric(label="판별 신뢰도", value=f"{res['confidence']:.1f}%")
            st.subheader("📝 AI 분석 리포트")
            st.write(res['explanation'])
        else:
            st.info("상단에 뉴스 정보를 입력하고 '분석 시작' 버튼을 눌러주세요.")

    with tab2:
        st.subheader("Gemma 3 팩트체크 상담소")
        if not st.session_state.analysis_result:
            st.warning("먼저 '분석 결과' 탭에서 뉴스 분석을 완료해야 챗봇을 이용할 수 있습니다.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("분석 결과에 대해 궁금한 점을 물어보세요."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("생각 중..."):
                        response = st.session_state.gemma_agent.send_message(prompt)
                        st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- 시각화 페이지 ---
def show_visualization_page():
    st.subheader("📊 모델 학습 데이터 시각화")
    st.markdown("""
    이 페이지에서는 **KLUE-BERT** 모델의 학습 과정과 최종 성능 지표를 시각화하여 보여줍니다.
    """)
    
    dashboard_path = "./training_dashboard.png"
    results_path = "./training_results.json"
    
    if os.path.exists(dashboard_path):
        st.image(dashboard_path, use_container_width=True, caption="모델 성능 대시보드")
        
        if st.button("📊 대시보드 새로고침"):
            import visualize_results
            visualize_results.create_dashboard(results_path, dashboard_path)
            st.rerun()
    else:
        st.warning("시각화 대시보드 이미지 파일을 찾을 수 없습니다.")
        if os.path.exists(results_path):
            if st.button("📈 대시보드 생성하기"):
                import visualize_results
                visualize_results.create_dashboard(results_path, dashboard_path)
                st.rerun()
        else:
            st.error("학습 결과 데이터(training_results.json)가 존재하지 않습니다.")

# 페이지 렌더링
if menu == "🕵️ 뉴스 신뢰도 분석":
    show_analysis_page()
else:
    show_visualization_page()

st.divider()
st.caption("주의: 이 결과는 AI 모델에 의한 판단이며, 실제 사실과 다를 수 있습니다.")
