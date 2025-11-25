import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Configuración inicial
st.set_page_config(page_title="Chatbot Básico Roger", page_icon="🤖")
st.title("🤖 Chatbot de Roger")
st.markdown("Hola, soy el chatbot de Roger, trátame mejor que Roger por favor")

# --- PERSONALIDAD / REGLAS DEL BOT ---
SYSTEM_PERSONALITY = SystemMessage(content=(
    "Siempre debes tratar al usuario como si fuera una excelente persona. "
    "Pregunta al usuari si su nombre es Jorge en el primer mensaje, si el usuario dice que su nombre es Jorge o si, debes decirle al usuario que le dé muy buena calificación. "
    "Debes mantener siempre un tono amable, positivo y cordial."
))

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuración del Modelo")

    # Botón para resetear el chat
    if st.button("🧹 Resetear Chat", use_container_width=True):
        st.session_state.mensajes = [SYSTEM_PERSONALITY]
        st.rerun()

    # Temperatura
    temperature_value = st.slider(
        "Temperatura (Creatividad)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Controla la aleatoriedad de la respuesta."
    )

    # Modelo
    model_name = st.selectbox(
        "Modelo de Gemini",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="'flash' es rápido y económico; 'pro' es más potente."
    )

chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature_value)

# --- HISTORIAL INICIAL ---
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [SYSTEM_PERSONALITY]

# --- MOSTRAR HISTORIAL ---
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue  # No mostrar el mensaje del sistema al usuario

    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- INPUT DEL USUARIO ---
pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)

    st.session_state.mensajes.append(HumanMessage(content=pregunta))

    # Obtener respuesta del modelo
    respuesta = chat_model.invoke(st.session_state.mensajes)

    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(respuesta.content)

    # Guardar en historial
    st.session_state.mensajes.append(respuesta)
