import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
import streamlit.components.v1 as components
import json
import math
from typing import Dict, Any

# ==========================================
# 1. 환경 설정 및 상수 정의
# ==========================================
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

# 최신 3 시리즈 모델로 업데이트된 리스트 적용
MODEL_LIST = [
    'gemini-3.1-flash-lite',      # 가장 빠르고 가벼운 3.1 최신 모델
    'gemini-3-flash-preview',     # 추론 성능이 더 뛰어난 3 프리뷰 모델
    'gemini-2.5-flash'            # 혹시 모를 서버 오류를 대비한 2.5 안정화 버전 (백업용)
]

# 💡 실시간 스트리밍과 3중 시각화(물리엔진/그래프/대시보드) 통합용 프롬프트
SYSTEM_INSTRUCTION = """
당신은 최고 수준의 AI 과학 실험실 시뮬레이터 설계자입니다.
응답은 반드시 아래의 [1부분]과 [2부분]으로 나누어 순서대로 작성하세요.

[1부분: 대화형 텍스트]
사용자에게 건넬 친절한 인사, 실험의 원리, 변수 조작 가이드를 마크다운 텍스트로 자유롭게 작성하세요.

[2부분: JSON 실험 설계 데이터]
텍스트 설명이 모두 끝난 후, 반드시 ---JSON_START--- 와 ---JSON_END--- 기호 사이에 화면을 렌더링할 설정값들을 순수 JSON으로만 작성하세요. 

---JSON_START---
{
    "has_experiment": true,
    "experiment_title": "실험 주제",
    "experiment_description": "실험 원리와 관찰 포인트 요약 (우측 상단 표시용)",
    "controls": [
        {"type": "slider", "label": "저항 (Ω)", "key": "resistance", "min": 1, "max": 100, "default": 10}
    ],
    "dashboard": {
        "metrics": [{"label": "전체 전류", "formula": "voltage / resistance", "unit": "A"}],
        "critical_events": []
    },
    "render_mode": "graph", 
    
    "graph_data": {
        "x_label": "전압(V)", "y_label": "전류(A)",
        "x_list": [0, 2, 4, 6, 8, 10],
        "y_formula": "x / resistance",
        "plot_type": "plot"
    },
    
    "html_code": "<style>body{margin:0; overflow:hidden; background:#f0f2f6;}</style><canvas id='simCanvas'></canvas><script>const canvas=document.getElementById('simCanvas'); const ctx=canvas.getContext('2d'); function resize(){canvas.width=window.innerWidth; canvas.height=window.innerHeight;} window.addEventListener('resize', resize); resize(); /* 이후 __key__ 변수들을 활용한 라이브 애니메이션 로직 작성 */</script>"
}
---JSON_END---

[🚨 중요한 작성 가이드]
1. 1부분(대화 텍스트)에는 절대 JSON 문법이나 백틱(```)을 넣지 마세요.
2. ---JSON_START--- 내부에는 ```json 같은 마크다운 기호를 절대 쓰지 마세요. 순수 괄호 { } 로만 시작하고 끝나야 합니다.
3. 수식(formula) 작성 시 파이썬 문법을 엄수하세요 (거듭제곱은 **, math 모듈 사용, 세미콜론 불가).
4. 🚨 [모드 선택 필수] 전기 회로(직렬/병렬), 화학 농도, 수학 함수 등 '데이터의 변화 추이나 통계'를 보는 실험은 반드시 "render_mode": "graph"로 설정하여 통계 자료(graph_data)를 렌더링하세요. 물체의 물리적 움직임(낙하, 진자 등)을 시각적으로 볼 필요가 있을 때만 "simulation"으로 설정하세요.
5. [매우 중요] 'html_code' 작성 시 절대 줄바꿈(Enter)을 하지 마세요. 반드시 한 줄(Single Line)로 길게 이어서 작성해야 파싱 에러가 나지 않습니다.
6. [simulation 모드 한정] 'html_code'는 정지된 그림이 아니라, `requestAnimationFrame`을 사용하여 시간에 따라 사물이 계속 움직이는 라이브 애니메이션 코드로 작성하고, 매 프레임마다 `ctx.clearRect`로 화면을 지우세요.
7. [simulation 모드 한정] 물체가 이동하는 시뮬레이션의 경우, 물체가 화면 밖으로 영원히 사라지지 않도록 바닥에 닿으면 탄성 충돌(Bounce)하거나 처음 위치로 리셋(Reset)되도록 코딩하세요.
"""

# ==========================================
# 2. 로직 함수들
# ==========================================

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_experiment' not in st.session_state:
        st.session_state.current_experiment = {
            "has_experiment": False,
            "experiment_title": "실험 대기 중",
            "experiment_description": "AI 조교에게 '진자 운동 실험 시뮬레이터 만들어줘' 혹은 '산염기 적정 실험 할래' 라고 요청해 보세요."
        }

def extract_json_data(full_text: str) -> Dict[str, Any]:
    if "---JSON_START---" in full_text:
        try:
            json_str = full_text.split("---JSON_START---")[1]
            if "---JSON_END---" in json_str:
                json_str = json_str.split("---JSON_END---")[0]
            
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON 파싱 실패 (안전 무시됨): {e}")
            return {"has_experiment": False}
    return {"has_experiment": False}

# 💡 [버그 3 패치] AI가 math.sin 대신 sin 이라고 적어도 계산되게끔 안전한 수학 환경 생성
SAFE_MATH_ENV = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}
SAFE_MATH_ENV["math"] = math

def render_virtual_lab(exp: Dict[str, Any]):
    if not exp.get("has_experiment"):
        st.info(exp.get("experiment_description", "실험을 요청해 주세요."))
        return

    # 1️⃣ 조작 패널 생성
    control_values = {}
    if exp.get('controls'):
        st.markdown("#### 🎛️ 조작 패널")
        cols = st.columns(len(exp['controls']))
        for i, ctrl in enumerate(exp['controls']):
            with cols[i % len(cols)]:
                unique_key = f"ui_{exp.get('experiment_title', 'lab').replace(' ', '_')}_{ctrl['key']}_{i}"
                
                if ctrl['type'] == 'toggle':
                    control_values[ctrl['key']] = st.toggle(ctrl['label'], value=ctrl.get('default', True), key=unique_key)
                elif ctrl['type'] == 'slider':
                    # 💡 [버그 2 패치] AI가 기본값(default)을 최솟값과 최댓값 밖으로 설정해서 뻗는 에러 방지 (Clamp)
                    min_v = float(ctrl['min'])
                    max_v = float(ctrl['max'])
                    def_v = float(ctrl.get('default', min_v))
                    def_v = max(min_v, min(max_v, def_v))
                    
                    control_values[ctrl['key']] = st.slider(
                        ctrl['label'], min_value=min_v, max_value=max_v, value=def_v, key=unique_key
                    )
    st.divider()

    # 계산용 통합 환경 구성 (버그 3 패치 적용)
    eval_env = {**SAFE_MATH_ENV, **control_values}

    # 2️⃣ 대시보드 (위험/폭발 임계점 경고)
    dashboard = exp.get('dashboard', {})
    for event in dashboard.get('critical_events', []):
        try:
            is_triggered = eval(event['condition'], {"__builtins__": {}}, eval_env)
            if is_triggered:
                st.error(event['message'])
        except Exception:
            pass

    # 3️⃣ 대시보드 (실시간 디지털 계기판)
    if dashboard.get('metrics'):
        metric_cols = st.columns(len(dashboard['metrics']))
        for i, metric in enumerate(dashboard['metrics']):
            try:
                val = eval(metric['formula'], {"__builtins__": {}}, eval_env)
                metric_cols[i % len(metric_cols)].metric(label=metric['label'], value=f"{val:.2f} {metric.get('unit', '')}")
            except Exception:
                metric_cols[i % len(metric_cols)].metric(label=metric['label'], value="계산 오류")
    
    st.markdown("---")
    
    mode = exp.get('render_mode', 'graph')
    
    # [대안 3] HTML5 Canvas 기반 물리 애니메이션 시뮬레이터
    if mode == 'simulation' and exp.get('html_code'):
        st.markdown("#### 🎬 라이브 시뮬레이터")
        sim_height = st.slider("↕️ 시뮬레이터 화면 크기 조절", min_value=300, max_value=1000, value=500, step=50, help="화면이 너무 작으면 바를 오른쪽으로 당겨보세요.")
        
        final_html = exp['html_code']
        for key, val in control_values.items():
            # 💡 [버그 1 패치] 파이썬의 True/False가 자바스크립트로 들어가며 Syntax Error를 유발하는 문제 해결
            js_val = str(val).lower() if isinstance(val, bool) else str(val)
            final_html = final_html.replace(f"__{key}__", js_val)
            
        components.html(final_html, height=sim_height)
        
    # [대안 1] 공간 좌표 시각화 또는 통계 그래프 모드
    elif mode == 'graph' and exp.get('graph_data'):
        st.markdown("#### 📊 공간/데이터 시각화")
        g_data = exp['graph_data']
        x_data = [float(i) for i in g_data.get('x_list', [])]
        y_data = []
        try:
            for x in x_data:
                local_env = {"x": x, **eval_env}
                y_data.append(eval(g_data['y_formula'], {"__builtins__": {}}, local_env))
                
            fig, ax = plt.subplots()
            
            try:
                if g_data.get('plot_type') == 'scatter':
                    ax.scatter(x_data, y_data, color='#FF4B4B', s=100)
                else:
                    ax.plot(x_data, y_data, marker='o', linestyle='-', color='#FF4B4B', linewidth=2)
                    
                ax.set_xlabel(g_data.get('x_label', 'X'))
                ax.set_ylabel(g_data.get('y_label', 'Y'))
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
            finally:
                # 💡 [버그 4 패치] try-finally 블록을 사용하여 그리는 도중 에러가 나도 메모리는 100% 비우도록 강화
                plt.close(fig) 
                
        except Exception as e:
            st.warning(f"수식 렌더링 중 오류가 발생했습니다: {e}")

# ==========================================
# 3. 메인 앱 UI 렌더링
# ==========================================

def main():
    st.set_page_config(page_title="궁극의 AI 가상 실험실", layout="wide")
    st.title("🧪 궁극의 AI 가상 과학 실험실")
    init_session_state()

    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        st.subheader("🤖 AI 조교와 대화")
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.write(chat["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(chat.get("display_content", chat.get("content", "")))

        if user_input := st.chat_input("예: 단진자 운동 시뮬레이션 만들어줘"):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                display_text = "" 
                
                try:
                    model = genai.GenerativeModel(
                        model_name=MODEL_LIST[0], 
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                    
                    response_stream = model.generate_content(user_input, stream=True)
                    
                    for chunk in response_stream:
                        try:
                            if chunk.text:
                                full_response += chunk.text
                                display_text = full_response.split("---JSON_START---")[0].strip()
                                message_placeholder.markdown(display_text + " ▌")
                        except ValueError:
                            pass 
                    
                    message_placeholder.markdown(display_text)
                    
                    ai_result = extract_json_data(full_response)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": full_response,
                        "display_content": display_text 
                    })
                    
                    if ai_result.get("has_experiment"):
                        st.session_state.current_experiment = ai_result
                        st.rerun() 
                        
                except Exception as e:
                    st.error(f"AI 통신 중 오류가 발생했습니다: {e}")

    with col2:
        st.subheader("🔬 가상 실험 구역")
        exp = st.session_state.current_experiment
        st.markdown(f"### ✨ {exp.get('experiment_title', '대기 중')}")
        st.info(exp.get('experiment_description', '좌측에서 실험을 요청해 주세요.'))
        
        render_virtual_lab(exp)

if __name__ == "__main__":
    main()
