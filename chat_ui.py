"""Guía contextual y panel flotante de Streamlit para Eco."""
from pathlib import Path
import hashlib
import time
import streamlit as st
from chat_service import read_settings, ask_assistant, local_intro, local_scope, ChatReply

ROOT = Path(__file__).resolve().parent
SUGGESTIONS = ["¿Cómo preparo este residuo?", "¿Qué significa la confianza?", "¿Cómo puedo mejorar la foto?"]


def settings_now():
    try:
        cloud = {name: st.secrets.get(name, "") for name in ["OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "CHAT_MAX_CALLS_PER_SESSION"]}
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        cloud = {}
    return read_settings(ROOT, cloud)


def setup_chat():
    state = st.session_state
    for name, value in {"eco_open": False, "eco_messages": [], "eco_identity": None,
                        "eco_intros": {}, "eco_calls": 0, "eco_last_call": 0.0}.items():
        if name not in state:
            state[name] = value


def sync_context(result):
    setup_chat()
    identity = (result.get("source"), result.get("image_sha256")) if result else None
    if st.session_state.eco_identity != identity:
        st.session_state.eco_identity = identity
        st.session_state.eco_messages = []


def request_reply(question, result, history, settings):
    setup_chat()
    would_call = bool(settings.api_key) and local_scope(question, bool(result) or bool(history))
    if would_call:
        if st.session_state.eco_calls >= settings.max_calls:
            return ChatReply("Se alcanzó el límite de consultas de esta sesión. Puedes seguir usando el clasificador y la guía de separación.", "local")
        if time.monotonic() - st.session_state.eco_last_call < 2:
            return ChatReply("Espera un momento antes de enviar otra consulta.", "local")
        st.session_state.eco_calls += 1
        st.session_state.eco_last_call = time.monotonic()
    return ask_assistant(question, result, history, settings)


def contextual_intro(result, settings):
    setup_chat()
    # Memoriza por imagen/modelo/credencial; no llama otra vez en cada rerun.
    key_tag = hashlib.sha256(settings.api_key.encode()).hexdigest()[:12] if settings.api_key else "local"
    identity = (result["image_sha256"], settings.model, key_tag)
    cache = st.session_state.eco_intros
    if identity not in cache:
        if settings.api_key:
            with st.spinner("Eco prepara una orientación…"):
                reply = request_reply("Comenta brevemente la predicción de este residuo, orienta sobre su separación y hazme una pregunta útil.", result, [], settings)
            if reply.kind in {"error", "refusal"}:
                reply = ChatReply(local_intro(result) + "\n\nLa orientación con IA no está disponible ahora.", "local")
        else:
            reply = ChatReply(local_intro(result), "local")
        cache[identity] = reply
        while len(cache) > 12:
            del cache[next(iter(cache))]
    return cache[identity]


def render_result_guide(result):
    sync_context(result)
    settings = settings_now()
    reply = contextual_intro(result, settings)
    with st.container(border=True):
        st.markdown("**✦ Eco te orienta**")
        st.markdown(reply.text)
        st.caption("Orientación con IA · basada en la estimación" if reply.kind == "openai" else "Guía local · sin llamada a OpenAI")
        if st.button("Conversar sobre este resultado ↗", key="eco_result_open", width="stretch"):
            st.session_state.eco_open = True


def render_chat_widget(result):
    sync_context(result)
    settings = settings_now()
    st.markdown("""<style>
    .st-key-eco_launcher {position:fixed!important;bottom:22px;right:24px;width:220px!important;z-index:1002;}
    .st-key-eco_launcher button {border-radius:28px!important;box-shadow:0 8px 28px #00000055;min-height:52px;font-weight:700;}
    .st-key-eco_panel {position:fixed!important;right:24px;bottom:22px;width:390px!important;max-width:calc(100vw - 24px);max-height:calc(100dvh - 85px);overflow-y:auto;z-index:1002;background:#282c2e;color:#e3e6e4;border:1px solid #4a5950;border-radius:20px;padding:16px;box-shadow:0 16px 60px #00000070;scrollbar-color:#617067 #282c2e;}
    .st-key-eco_panel h3 {font-size:1.12rem!important;margin:0;}
    .st-key-eco_panel [data-testid="stChatMessage"] {padding:10px;border-radius:12px;background:#323a35;color:#e3e6e4;}
    .st-key-eco_panel [data-testid="stChatMessage"] p {font-size:.88rem;line-height:1.5;}
    .st-key-eco_panel [data-testid="stCaptionContainer"] p {font-size:.73rem;}
    .st-key-eco_panel button p {font-size:.82rem;}
    .eco-heading {font-weight:750;color:#a3dbba;font-size:1.13rem;}
    @media(max-width:600px){.st-key-eco_panel{right:12px;bottom:12px;width:calc(100vw - 24px)!important;padding:12px;}.st-key-eco_launcher{right:12px;bottom:12px;}}
    </style>""", unsafe_allow_html=True)
    if not st.session_state.eco_open:
        with st.container(key="eco_launcher"):
            if st.button("✦ Pregúntale a Eco", key="eco_open_button", type="primary", width="stretch"):
                st.session_state.eco_open = True
                st.rerun()
        return
    with st.container(key="eco_panel"):
        title, close = st.columns([5, 1])
        with title:
            st.markdown('<div class="eco-heading">✦ Eco · Tu guía de reciclaje</div>', unsafe_allow_html=True)
            st.caption("OpenAI configurado" if settings.api_key else "Modo guía · configura tu clave para conversar")
        with close:
            if st.button("✕", key="eco_close", help="Cerrar asistente"):
                st.session_state.eco_open = False
                st.rerun()
        if result:
            st.caption(f"Sobre esta imagen: {result['label']} · {result['confidence']:.1%} de confianza")
        intro = contextual_intro(result, settings) if result else ChatReply(local_intro(None), "local")
        with st.container(height=210, key="eco_history"):
            with st.chat_message("assistant", avatar="♻️"):
                st.markdown(intro.text)
            for message in st.session_state.eco_messages[-12:]:
                with st.chat_message(message["role"], avatar="♻️" if message["role"] == "assistant" else "🙂"):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        st.caption("OpenAI" if message.get("kind") == "openai" else "Respuesta del sistema")
        pending = None
        if not st.session_state.eco_messages:
            suggestions = SUGGESTIONS if result else ["¿Qué materiales reconoce ReClasifica?", "¿Cómo separo papel y cartón?", "¿Qué significa la confianza?"]
            for i, question in enumerate(suggestions):
                if st.button(question, key=f"eco_suggest_{i}", width="stretch"):
                    pending = question
        with st.form("eco_question_form", clear_on_submit=True):
            question = st.text_input("Tu pregunta sobre reciclaje", max_chars=600, placeholder="¿Y si el envase está sucio?", key="eco_question")
            submitted = st.form_submit_button("Enviar pregunta", type="primary", width="stretch")
        if submitted and question.strip():
            pending = question.strip()
        if pending:
            with st.spinner("Eco está pensando…"):
                history = [{"role": "assistant", "content": intro.text, "accepted": True}]
                history += st.session_state.eco_messages
                reply = request_reply(pending, result, history, settings)
            accepted = reply.kind == "openai"
            st.session_state.eco_messages.extend([
                {"role": "user", "content": pending, "accepted": accepted},
                {"role": "assistant", "content": reply.text, "accepted": accepted, "kind": reply.kind},
            ])
            st.session_state.eco_messages = st.session_state.eco_messages[-12:]
            st.rerun()
        if st.session_state.eco_messages and st.button("Nueva conversación", key="eco_clear"):
            st.session_state.eco_messages = []
            st.rerun()
        st.caption("Solo reciclaje y este proyecto. Se envían texto y probabilidades a OpenAI; la foto no se envía. Puede equivocarse.")
