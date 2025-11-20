import streamlit as st
from openai import OpenAI
from typing import List, Dict

st.set_page_config(page_title="Dog Breed Recommender Chatbot", page_icon="🐶")

st.title("🐶 강아지 품종 추천 챗봇")
st.write(
    "강아지의 생활 방식과 선호도를 입력하면 적합한 품종을 추천해주고, 추가 질문으로 대화를 이어갑니다."
)


# --- Conversation state (initialize early so system prompt editor can show current value)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specialized in recommending dog breeds. "
                "When a user gives preferences (size, activity level, living situation, "
                "allergies, grooming willingness, experience with dogs, children or pets), "
                "suggest 2-4 suitable breeds with short reasons and follow up with one "
                "clarifying question to better tailor recommendations. "
                "Keep answers friendly and concise in Korean unless the user asks otherwise."
            ),
        }
    ]

# --- System prompt editor (placed directly under the main title)
st.subheader("시스템 프롬프트 편집")
current_system_prompt = st.session_state.messages[0]["content"] if st.session_state.messages else ""
system_prompt_input = st.text_area(
    label="시스템 프롬프트 (챗봇 전체 동작 지침)",
    value="",
    placeholder=current_system_prompt,
    help="여기에 장문의 시스템 프롬프트를 입력한 후 '적용' 버튼을 누르세요.",
    height=180,
)
if st.button("적용 (Apply System Prompt)"):
    if system_prompt_input and system_prompt_input.strip():
        st.session_state.messages[0]["content"] = system_prompt_input.strip()
        st.success("시스템 프롬프트가 적용되었습니다.")
    else:
        st.warning("빈 입력은 적용할 수 없습니다. 기존 프롬프트를 보려면 placeholder를 확인하세요.")


# --- API Key (no user input field) -------------------------------------------------
# Use Streamlit secrets: .streamlit/secrets.toml should contain OPENAI_API_KEY
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.error(
        "OpenAI API 키가 설정되어 있지 않습니다. `.streamlit/secrets.toml`에 `OPENAI_API_KEY`를 추가하세요."
    )
    st.stop()


# --- OpenAI client ---------------------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)


# (Conversation state initialized above so editor can access it)


# --- Sidebar controls -----------------------------------------------------------
with st.sidebar:
    st.header("설정")
    if st.button("대화 초기화 (New Chat)"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.success("대화가 초기화되었습니다.")


# --- Display chat history ------------------------------------------------------
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    # Don't show system message in the chat stream; it's used only for behavior.
    if role == "system":
        continue
    with st.chat_message(role):
        st.markdown(content)


# --- User input ---------------------------------------------------------------
user_input = st.chat_input("강아지에 대해 어떤 점을 중요하게 생각하시나요? (예: 활동량, 크기, 알레르기 등)")
if user_input:
    # Append user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare messages for the API (convert to the format expected)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Call OpenAI Chat Completions (gpt-4o-mini)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            temperature=0.7,
            max_tokens=600,
        )

        # Extract assistant text robustly (support multiple client return shapes)
        assistant_text = ""
        if hasattr(completion, "choices") and len(completion.choices) > 0:
            choice = completion.choices[0]
            # choice.message might be dict-like or an object
            if hasattr(choice, "message"):
                msg = choice.message
                if isinstance(msg, dict):
                    assistant_text = msg.get("content", "")
                else:
                    # object with attributes
                    assistant_text = getattr(msg, "content", "") or getattr(msg, "text", "")
            else:
                # older shape: choice.text
                assistant_text = getattr(choice, "text", "")
        else:
            # fallback for responses API shape
            assistant_text = getattr(completion, "text", None) or str(completion)

    except Exception as e:
        st.error(f"OpenAI API 호출 중 오류 발생: {e}")
        assistant_text = "죄송합니다. 응답을 생성하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."

    # Show assistant response and append to session state
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

