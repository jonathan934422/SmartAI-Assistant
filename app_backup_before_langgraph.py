import hashlib

import streamlit as st

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartAI Assistant",
    page_icon="🤖",
    layout="wide",
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

st.caption(
    f"Logged in as **{st.session_state.username}**"
)

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


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in messages:

    role = message.get("role")

    if role == "system":
        continue

    with st.chat_message(role):
        st.markdown(
            message.get("content", "")
        )


# ============================================================
# VOICE INPUT
# ============================================================

audio_input = st.audio_input(
    "🎙️ Record your question"
)


if audio_input:

    audio_bytes = audio_input.getvalue()

    audio_hash = hashlib.md5(
        audio_bytes
    ).hexdigest()

    if (
        audio_hash
        != st.session_state.last_audio_hash
    ):

        st.session_state.last_audio_hash = (
            audio_hash
        )

        language_code = None

        if voice_language == "English":
            language_code = "en"

        elif voice_language == "Tamil":
            language_code = "ta"

        try:

            with st.spinner(
                "Transcribing..."
            ):
                transcribed_text = transcribe_audio(
                    audio_bytes,
                    language=language_code,
                )

            if transcribed_text:

                st.session_state.voice_prompt = (
                    transcribed_text
                )

                st.success(
                    f"Transcribed: {transcribed_text}"
                )

                st.rerun()

        except Exception as error:

            st.error(
                f"Voice input error: {error}"
            )


# ============================================================
# CHAT INPUT
# ============================================================

typed_prompt = st.chat_input(
    "Ask SmartAI Assistant..."
)

voice_prompt = st.session_state.pop(
    "voice_prompt",
    None,
)

user_prompt = (
    typed_prompt
    if typed_prompt
    else voice_prompt
)


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
    # BUILD AI MESSAGES
    # --------------------------------------------------------

    ai_messages = list(messages)

    # --------------------------------------------------------
    # PDF RAG
    # --------------------------------------------------------

    if (
        st.session_state.pdf_chunks
        and st.session_state.pdf_embeddings is not None
    ):

        try:

            query_embedding = create_embeddings(
                [user_prompt]
            )[0]

            relevant_chunks = find_relevant_chunks(
                user_prompt,
                st.session_state.pdf_chunks,
                st.session_state.pdf_embeddings,
                query_embedding,
                top_k=4,
            )

            if relevant_chunks:

                context = "\n\n".join(
                    item["chunk"]
                    for item in relevant_chunks
                )

                ai_messages = [
                    messages[0],
                    {
                        "role": "system",
                        "content": (
                            "The user uploaded a PDF. "
                            "Use the following PDF context "
                            "to answer the question. "
                            "If the answer is not present "
                            "in the PDF, clearly say so.\n\n"
                            "PDF CONTEXT:\n"
                            f"{context}"
                        ),
                    },
                ]

                ai_messages.append(
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                )

        except Exception as error:

            st.warning(
                f"PDF search could not be completed: {error}"
            )

    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    if web_search_enabled:

        try:

            with st.spinner(
                "Searching the web..."
            ):
                response_text = web_search_response(
                    user_prompt
                )

            st.chat_message(
                "assistant"
            ).markdown(
                response_text
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

            save_message(
                current_id,
                st.session_state.user_id,
                "assistant",
                response_text,
            )

            if voice_output_enabled:

                try:

                    audio_bytes = text_to_speech(
                        response_text
                    )

                    if audio_bytes:
                        st.audio(
                            audio_bytes,
                            format="audio/wav",
                        )

                except Exception as error:

                    st.warning(
                        f"Voice output error: {error}"
                    )

            st.rerun()

        except Exception as error:

            st.error(
                f"Web search error: {error}"
            )

    # --------------------------------------------------------
    # NORMAL GROQ RESPONSE
    # --------------------------------------------------------

    else:

        st.chat_message(
            "assistant"
        )

        response_placeholder = st.empty()

        full_response = ""

        try:

            for chunk in generate_response(
                ai_messages
            ):

                full_response += chunk

                response_placeholder.markdown(
                    full_response
                )

            if full_response.strip():

                messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                    }
                )

                save_message(
                    current_id,
                    st.session_state.user_id,
                    "assistant",
                    full_response,
                )

                if voice_output_enabled:

                    try:

                        audio_bytes = text_to_speech(
                            full_response
                        )

                        if audio_bytes:

                            st.audio(
                                audio_bytes,
                                format="audio/wav",
                            )

                    except Exception as error:

                        st.warning(
                            f"Voice output error: {error}"
                        )

        except Exception as error:

            st.error(
                f"AI response error: {error}"
            )