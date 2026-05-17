import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- [1단계] API 키 설정 (사용자님의 키 적용) ---
# st.secrets는 나중에 Streamlit 웹 설정창에서 입력할 값을 불러옵니다.
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

def get_ai_response(prompt):
    # 2026년 기준 실제 호출 가능한 모델 ID 리스트 (순서대로 시도)
    model_list = [
        'gemini-2.5-flash',      # 1순위: 메인 모델
        'gemini-2.5-pro',        # 2순위: 더 똑똑하지만 한도가 타이트함
        'gemini-2.5-flash-lite', # 3순위: 가장 빠르고 한도가 넉넉함
        'gemini-1.5-flash'       # 4순위: 구버전 안정 모델
    ]
    
    for model_id in model_list:
        try:
            # 모델 설정 및 생성
            model = genai.GenerativeModel(model_id)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # 터미널에 어떤 모델이 왜 실패했는지 표시 (디버깅용)
            print(f"로그: {model_id} 사용 불가 ({e})")
            continue  # 다음 모델로 넘어감
            
    return "현재 모든 AI 모델의 사용 한도가 초과되었습니다. 잠시 후 다시 시도해주세요."

# --- [2단계] 실험 데이터 ---
knowledge = {
    "경사면에서의 운동": "물체가 경사를 타고 내려올 때 가속도를 계산합니다.",
    "자유낙하": "공기 저항이 없는 상태에서 물체가 떨어지는 운동입니다."
}

# --- [3단계] 메인 앱 ---
def main():
    st.set_page_config(page_title="AI 가상 실험실")
    st.title("🧪 AI 가상 과학 실험실")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 레이아웃 나누기
    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        st.subheader("🤖 AI 조교와 대화")
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])

        if user_input := st.chat_input("질문을 입력하세요"):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                # 로딩 표시와 함께 AI 답변 가져오기
                with st.spinner("생각 중..."):
                    answer = get_ai_response(user_input)
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with col2:
        st.subheader("🔬 실험 구역")
        selected = st.selectbox("실험 선택", list(knowledge.keys()))
        st.info(knowledge[selected])
        
        # 간단한 그래프 예시
        x = np.linspace(0, 10, 100)
        y = x**2
        fig, ax = plt.subplots()
        ax.plot(x, y)
        st.pyplot(fig)

if __name__ == "__main__":
    main()