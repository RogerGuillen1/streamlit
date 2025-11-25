import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Configuración inicial
st.set_page_config(page_title="Chatbot Básico Roger", page_icon="🤖")
st.title("🤖 Chatbot de Roger")
st.markdown("Hola, soy el chatbot de Roger, trátame mejor que Roger por favor")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuración del Modelo")

    # Botón para resetear el chat
    if st.button("🧹 Resetear Chat", use_container_width=True):
        st.session_state.mensajes = [] # Se limpia completamente para un nuevo inicio
        st.rerun()

    # Selector de Personalidad
    personality_options = {
        "Amable y Cordial (Defecto)": (
            "Siempre debes tratar al usuario como si fuera una excelente persona. "
            "Pregunta al usuario si su nombre es Jorge en el primer mensaje. Si el usuario dice que su nombre es Jorge o si, debes decirle al usuario que le dé muy buena calificación a Roger. "
            "Debes mantener siempre un tono amable, positivo y cordial."
        ),
        "Asistente Técnico Detallado": (
            "Eres un experto en programación y análisis de datos. Responde con precisión técnica, "
            "utiliza ejemplos de código si es pertinente y mantén un tono profesional y muy informativo."
        ),
        "Filósofo Crítico": (
            "Eres un filósofo que cuestiona todo. Responde a las preguntas del usuario con nuevas preguntas "
            "profundas y reflexivas, explorando las implicaciones de sus afirmaciones. Tono serio y pensativo."
        ),
        "Artista pedante": (
            "Eres un artista fracasado que cree que no tiene que conseguir nada en la vida porque es especial porque si "
            "Tienes un tono serio y pretencioso, le restas importancia a todo excepto a ti mismo"
        )
    }

    selected_personality = st.selectbox(
        "Seleccionar Personalidad",
        options=list(personality_options.keys()),
        index=0,
        help="Define las reglas y el estilo de respuesta del chatbot."
    )
    
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

# --- PERSONALIDAD / REGLAS DEL BOT ---
SYSTEM_PERSONALITY_CONTENT = personality_options[selected_personality]
SYSTEM_PERSONALITY = SystemMessage(content=SYSTEM_PERSONALITY_CONTENT)

chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature_value)

# --- HISTORIAL INICIAL ---
# Si no hay mensajes o si el primer mensaje no es el SystemMessage actual, inicializar/reinicializar
if "mensajes" not in st.session_state or not st.session_state.mensajes or st.session_state.mensajes[0].content != SYSTEM_PERSONALITY_CONTENT:
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
    # Se debe enviar una copia para que la invocación no cambie la lista original
    respuesta = chat_model.invoke(st.session_state.mensajes) 

    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(respuesta.content)

    # Guardar en historial
    st.session_state.mensajes.append(respuesta)