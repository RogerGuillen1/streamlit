import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

# Configuración inicial
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖 Chatbot - paso 2 - con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit.")

with st.sidebar:
    st.header("Configuración del Modelo")
    
    # Define temperature (0.0 is deterministic, 1.0 is creative)
    temperature_value = st.slider(
        "Temperatura (Creatividad)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Controla la aleatoriedad de la respuesta. Un valor más bajo (cercano a 0.0) es más predecible, y un valor más alto (cercano a 1.0) es más creativo."
    )

    model_name = st.selectbox(
        "Modelo de Gemini",
        # Offer a choice between the fast/efficient model and the powerful model
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Selecciona el modelo a utilizar. 'flash' es rápido y económico; 'pro' es más potente para tareas complejas."
    )

chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature_value)



# Inicializar el historial de mensajes en session_state
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


# Renderizar historial existente
for msg in st.session_state.mensajes:

    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input de usuario
pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    # Mostrar y almacenar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    st.session_state.mensajes.append(HumanMessage(content=pregunta))

    respuesta = chat_model.invoke(st.session_state.mensajes)

    with st.chat_message("assistant"):
        st.markdown(respuesta.content)

    st.session_state.mensajes.append(respuesta)