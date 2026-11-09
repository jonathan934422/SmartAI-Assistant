import hashlib
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from config.settings import GROQ_API_KEY
from core.groq_client import generate_response
from agents.assistant_graph import assistant_graph
from core.memory import (
    authenticate_user,
    create_conversation,
    create_user,
    delete_conversation,
    email_exists,
    get_conversations,
    load_memory,
    rename_conversation,
    save_message,
    username_exists,
)
from rag.document_loader import (
    extract_text_from_pdf,
    split_text_into_chunks,
)
from rag.embeddings import create_embeddings
from rag.retriever import find_relevant_chunks
from search.web_search import web_search_response
from voice.speech import transcribe_audio, text_to_speech


# ============================================================
# CUSTOM CHATGPT-STYLE COMPOSER COMPONENT
# ============================================================

smartai_composer = components.declare_component(
    "smartai_composer",
    path=str(
        Path(__file__).parent
        / "components"
        / "smartai_composer"
    ),
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartAI Assistant",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# CHATGPT-INSPIRED CHAT DESIGN
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background: var(--background-color); }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.45rem;
    }
    [data-testid="stChatInput"] {
        border-radius: 24px !important;
        border: 1px solid rgba(128,128,128,.38) !important;
        box-shadow: 0 4px 18px rgba(0,0,0,.06);
    }
    [data-testid="stChatInput"] textarea { font-size: 1rem !important; }
    [data-testid="stFileUploader"] section { border-radius: 14px; }
    .smartai-subtitle { color: var(--text-color); opacity: .72; margin-top: -12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "pdf_embeddings" not in st.session_state:
    st.session_state.pdf_embeddings = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

if "last_composer_submission_id" not in st.session_state:
    st.session_state.last_composer_submission_id = None

if "pending_voice_text" not in st.session_state:
    st.session_state.pending_voice_text = ""


# ============================================================
# SYSTEM MESSAGE
# ============================================================

SYSTEM_MESSAGE = {
    "role": "system",
    "content": """
You are SmartAI Assistant, a helpful multilingual AI assistant.

IMPORTANT LANGUAGE RULES:

1. Understand English, Tamil, Thanglish, and mixed Tamil-English.

2. If the user writes in English:
   - Answer in clear English.

3. If the user writes in Tamil:
   - Answer in natural Tamil.
   - Use Tamil script.

4. If the user writes in Thanglish:
   - Answer in natural Thanglish.
   - Keep Tamil words written using English letters.
   - Do not unnecessarily convert the response into Tamil script.

5. If the user mixes Tamil and English:
   - Respond naturally using a similar Tamil-English mixed style.

6. Do not translate the user's language into another language
   unless the user specifically asks for translation.

7. Keep answers beginner-friendly and easy to understand.

8. If the user asks a technical question, explain difficult
   concepts using simple language and examples.

9. If the user asks for code, provide correct and complete code
   when appropriate.

10. Always prioritize accuracy and directly answer the question.
""",
}


# ============================================================
# AUTHENTICATION FUNCTIONS
# ============================================================

def login_screen():
    st.title("🤖 SmartAI Assistant")

    st.subheader("Welcome back")

    login_tab, register_tab = st.tabs(
        ["🔐 Login", "📝 Register"]
    )

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    with login_tab:
        st.markdown("### Login to your account")

        username = st.text_input(
            "Username",
            key="login_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True,
        ):
            if not username or not password:
                st.error(
                    "Please enter username and password."
                )
            else:
                user = authenticate_user(
                    username.strip(),
                    password,
                )

                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user["id"]
                    st.session_state.username = user["username"]
                    st.session_state.user_email = user.get(
                        "email"
                    )

                    st.session_state.conversations = load_memory(
                        user["id"]
                    )

                    conversations = get_conversations(
                        st.session_state.conversations
                    )

                    if conversations:
                        st.session_state.current_conversation_id = (
                            conversations[0][0]
                        )
                    else:
                        conversation_id = create_conversation(
                            st.session_state.conversations,
                            user["id"],
                            "New Chat",
                            SYSTEM_MESSAGE,
                        )

                        st.session_state.current_conversation_id = (
                            conversation_id
                        )

                    st.success("Login successful!")

                    st.rerun()

                else:
                    st.error(
                        "Invalid username or password."
                    )

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    with register_tab:
        st.markdown("### Create a new account")

        new_username = st.text_input(
            "Username",
            key="register_username",
        )

        new_email = st.text_input(
            "Email",
            key="register_email",
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="register_password",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="register_confirm_password",
        )

        if st.button(
            "Create Account",
            type="primary",
            use_container_width=True,
        ):
            username = new_username.strip()
            email = new_email.strip()

            if not username or not email or not new_password:
                st.error(
                    "Please fill in all fields."
                )

            elif new_password != confirm_password:
                st.error(
                    "Passwords do not match."
                )

            elif len(new_password) < 6:
                st.error(
                    "Password must contain at least 6 characters."
                )

            elif username_exists(username):
                st.error(
                    "Username already exists."
                )

            elif email_exists(email):
                st.error(
                    "Email already exists."
                )

            else:
                try:
                    create_user(
                        username,
                        email,
                        new_password,
                    )

                    st.success(
                        "Account created successfully! "
                        "You can now login."
                    )

                except Exception as error:
                    st.error(
                        f"Could not create account: {error}"
                    )


# ============================================================
# STOP HERE IF NOT LOGGED IN
# ============================================================

if not st.session_state.authenticated:
    login_screen()
    st.stop()


# ============================================================
# LOAD CURRENT USER CONVERSATIONS
# ============================================================

if not st.session_state.conversations:
    st.session_state.conversations = load_memory(
        st.session_state.user_id
    )


if not st.session_state.current_conversation_id:
    conversations = get_conversations(
        st.session_state.conversations
    )

    if conversations:
        st.session_state.current_conversation_id = (
            conversations[0][0]
        )

    else:
        conversation_id = create_conversation(
            st.session_state.conversations,
            st.session_state.user_id,
            "New Chat",
            SYSTEM_MESSAGE,
        )

        st.session_state.current_conversation_id = (
            conversation_id
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💬 SmartAI")

    st.markdown(
        f"👤 **{st.session_state.username}**"
    )

    if st.session_state.user_email:
        st.caption(
            st.session_state.user_email
        )

    if st.button(
        "🚪 Logout",
        use_container_width=True,
    ):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.user_email = None
        st.session_state.conversations = {}
        st.session_state.current_conversation_id = None
        st.session_state.pdf_chunks = []
        st.session_state.pdf_embeddings = None
        st.session_state.pdf_name = None
        st.session_state.last_audio_hash = None

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):
        conversation_id = create_conversation(
            st.session_state.conversations,
            st.session_state.user_id,
            "New Chat",
            SYSTEM_MESSAGE,
        )

        st.session_state.current_conversation_id = (
            conversation_id
        )

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # CONVERSATIONS
    # --------------------------------------------------------

    st.subheader("Your Conversations")

    conversations = get_conversations(
        st.session_state.conversations
    )

    for conversation_id, conversation in conversations:

        title = conversation.get(
            "title",
            "New Chat",
        )

        is_current = (
            conversation_id
            == st.session_state.current_conversation_id
        )

        button_label = (
            f"🟣 {title}"
            if is_current
            else f"💬 {title}"
        )

        if st.button(
            button_label,
            key=f"chat_{conversation_id}",
            use_container_width=True,
        ):
            st.session_state.current_conversation_id = (
                conversation_id
            )

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # RENAME
    # --------------------------------------------------------

    st.subheader("Conversation Settings")

    rename_title = st.text_input(
        "Rename current chat",
        key="rename_title",
    )

    if st.button(
        "✏️ Rename",
        use_container_width=True,
    ):
        if rename_title.strip():
            rename_conversation(
                st.session_state.conversations,
                st.session_state.current_conversation_id,
                rename_title,
                st.session_state.user_id,
            )

            st.rerun()

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    if st.button(
        "🗑️ Delete Current Chat",
        use_container_width=True,
    ):
        current_id = (
            st.session_state.current_conversation_id
        )

        delete_conversation(
            st.session_state.conversations,
            current_id,
            st.session_state.user_id,
        )

        remaining = get_conversations(
            st.session_state.conversations
        )

        if remaining:
            st.session_state.current_conversation_id = (
                remaining[0][0]
            )

        else:
            conversation_id = create_conversation(
                st.session_state.conversations,
                st.session_state.user_id,
                "New Chat",
                SYSTEM_MESSAGE,
            )

            st.session_state.current_conversation_id = (
                conversation_id
            )

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    web_search_enabled = st.toggle(
        "🌐 Web Search",
        value=False,
    )

    # --------------------------------------------------------
    # VOICE SETTINGS
    # --------------------------------------------------------

    st.subheader("🎙️ Voice")

    voice_language = st.selectbox(
        "Voice Input Language",
        [
            "Auto",
            "English",
            "Tamil",
        ],
    )

    voice_output_enabled = st.toggle(
        "🔊 Voice Output",
        value=False,
    )

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.subheader("📄 PDF Knowledge")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_pdf:

        if st.session_state.pdf_name != uploaded_pdf.name:

            with st.spinner(
                "Reading PDF..."
            ):
                pdf_text = extract_text_from_pdf(
                    uploaded_pdf
                )

                chunks = split_text_into_chunks(
                    pdf_text
                )

                embeddings = create_embeddings(
                    chunks
                )

                st.session_state.pdf_chunks = chunks
                st.session_state.pdf_embeddings = embeddings
                st.session_state.pdf_name = (
                    uploaded_pdf.name
                )

            st.success(
                f"Loaded: {uploaded_pdf.name}"
            )

        else:
            st.info(
                f"Active PDF: {uploaded_pdf.name}"
            )

    if st.button(
        "🗑️ Remove PDF",
        use_container_width=True,
    ):
        st.session_state.pdf_chunks = []
        st.session_state.pdf_embeddings = None
        st.session_state.pdf_name = None

        st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================

st.title("🤖 SmartAI Assistant")

st.markdown("<div class='smartai-subtitle'>Your AI assistant for chat, documents, web search, and voice.</div>", unsafe_allow_html=True)
st.caption(f"Logged in as **{st.session_state.username}**")

current_id = st.session_state.current_conversation_id


if current_id not in st.session_state.conversations:

    conversation_id = create_conversation(
        st.session_state.conversations,
        st.session_state.user_id,
        "New Chat",
        SYSTEM_MESSAGE,
    )

    st.session_state.current_conversation_id = (
        conversation_id
    )

    current_id = conversation_id


conversation = st.session_state.conversations[
    current_id
]

messages = conversation["messages"]


# Reserve the chat-history position above the composer. We fill this
# container after processing a new submission, so the newest response
# appears above the composer without forcing an extra Streamlit rerun.
chat_history_area = st.container()


# ============================================================
# CHATGPT-STYLE SINGLE MESSAGE COMPOSER
# ============================================================

# A voice transcript is returned to the composer as editable text.
# Pop it once so it is not reinserted on later Streamlit reruns.
pending_voice_text = st.session_state.pop("pending_voice_text", "")

composer_value = smartai_composer(
    key="smartai_composer",
    placeholder="Message SmartAI...",
    accepted_file_types=["application/pdf"],
    voice_language=voice_language,
    max_file_size_mb=25,
    transcribed_text=pending_voice_text,
    default=None,
)

user_prompt = None

# The component sends a new unique submission ID only when the
# user presses Send. This prevents reruns from resubmitting it.
if isinstance(composer_value, dict):
    submission_id = composer_value.get("id")

    if submission_id and submission_id != st.session_state.last_composer_submission_id:
        st.session_state.last_composer_submission_id = submission_id

        # Load attached PDFs into the existing RAG pipeline.
        attached_files = composer_value.get("files") or []

        for attached_file in attached_files:
            file_name = attached_file.get("name", "uploaded.pdf")
            file_type = attached_file.get("type", "application/pdf")
            encoded_data = attached_file.get("data", "")

            if not file_name.lower().endswith(".pdf") and file_type != "application/pdf":
                st.warning(f"Unsupported file: {file_name}. Please attach a PDF.")
                continue

            try:
                file_bytes = base64.b64decode(encoded_data)
                from io import BytesIO
                pdf_file = BytesIO(file_bytes)
                pdf_file.name = file_name

                with st.spinner(f"Reading {file_name}..."):
                    pdf_text = extract_text_from_pdf(pdf_file)
                    chunks = split_text_into_chunks(pdf_text)
                    embeddings = create_embeddings(chunks)

                st.session_state.pdf_chunks = chunks
                st.session_state.pdf_embeddings = embeddings
                st.session_state.pdf_name = file_name
                st.success(f"Attached and indexed: {file_name}")

            except Exception as error:
                st.error(f"Could not read {file_name}: {error}")

        typed_prompt = (composer_value.get("text") or "").strip()
        audio_payload = composer_value.get("audio")

        # If a voice recording was included, transcribe it with the
        # project's existing voice transcription function.
        voice_prompt = None
        if audio_payload and audio_payload.get("data"):
            try:
                audio_bytes = base64.b64decode(audio_payload["data"])
                audio_hash = hashlib.md5(audio_bytes).hexdigest()

                if audio_hash != st.session_state.last_audio_hash:
                    st.session_state.last_audio_hash = audio_hash

                    language_code = None
                    if voice_language == "English":
                        language_code = "en"
                    elif voice_language == "Tamil":
                        language_code = "ta"

                    # Most speech modules accept raw audio bytes. Some implementations
                    # expect a file-like object instead, so try bytes first and then
                    # retry with a named buffer if the function rejects that input.
                    with st.spinner("Transcribing your voice..."):
                        mime_type = audio_payload.get("type", "audio/webm")
                        voice_prompt = transcribe_audio(
                            audio_bytes,
                            language=language_code,
                            audio_mime_type=mime_type,
                        )

                    if isinstance(voice_prompt, str):
                        voice_prompt = voice_prompt.strip()

                    if not voice_prompt:
                        st.warning(
                            "I couldn't detect speech in that recording. Please speak clearly, "
                            "record a little longer, and try again."
                        )
                    else:
                        st.session_state.last_audio_hash = audio_hash

                        # Voice-only input should fill the message box first.
                        # The user can review/edit the transcript and press Send.
                        if not typed_prompt:
                            st.session_state.pending_voice_text = voice_prompt
                            st.rerun()
            except Exception as error:
                st.error(f"Voice input error: {error}")

        user_prompt = typed_prompt or voice_prompt

        # If the user only attached a PDF, index it without sending an
        # empty prompt to the model.
        # PDF-only submissions are indexed without sending an empty
        # prompt. Continue rendering the existing conversation below.


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_prompt:

    user_prompt = user_prompt.strip()

    if not user_prompt:
        st.stop()

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    save_message(
        current_id,
        st.session_state.user_id,
        "user",
        user_prompt,
    )

    # --------------------------------------------------------
    # AUTOMATIC CHAT TITLE
    # --------------------------------------------------------

    current_title = conversation.get(
        "title",
        "New Chat",
    )

    if current_title == "New Chat":

        new_title = user_prompt[:60]

        if len(user_prompt) > 60:
            new_title += "..."

        rename_conversation(
            st.session_state.conversations,
            current_id,
            new_title,
            st.session_state.user_id,
        )

    # --------------------------------------------------------
    # BUILD AI MESSAGES AND RUN LANGGRAPH
    # --------------------------------------------------------

    ai_messages = list(messages)

    try:
        with st.spinner("SmartAI is thinking..."):
            graph_result = assistant_graph.invoke(
                {
                    "user_message": user_prompt,
                    "messages": ai_messages,
                    "web_search_enabled": web_search_enabled,
                    "pdf_chunks": st.session_state.pdf_chunks,
                    "pdf_embeddings": st.session_state.pdf_embeddings,
                    "route": "",
                    "response": "",
                }
            )

        response_text = (graph_result.get("response") or "").strip()
        selected_route = graph_result.get("route", "general")

        if not response_text:
            response_text = "I couldn't generate a response. Please try again."

        messages.append({"role": "assistant", "content": response_text})
        save_message(
            current_id,
            st.session_state.user_id,
            "assistant",
            response_text,
        )

        if voice_output_enabled:
            try:
                audio_bytes = text_to_speech(response_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
            except Exception as error:
                st.warning(f"Voice output error: {error}")

    except Exception as error:
        st.error(f"AI response error: {error}")


# ============================================================
# DISPLAY UPDATED CHAT HISTORY ABOVE THE COMPOSER
# ============================================================

with chat_history_area:
    for message in messages:
        role = message.get("role")

        if role == "system":
            continue

        with st.chat_message(role):
            st.markdown(message.get("content", ""))
