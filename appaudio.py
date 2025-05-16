import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langfuse.callback import CallbackHandler

# --------------------------------------------------
# Nuevas importaciones
import io
import openai
from streamlit_audio_recorder import audio_recorder
from gtts import gTTS
# --------------------------------------------------

# Carga variables de entorno
load_dotenv()
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]

# Configuro OpenAI Audio (Whisper) para Together.xyz
openai.api_key = TOGETHER_API_KEY
openai.api_base = "https://api.together.xyz/"

handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
)

model = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=1024,
    openai_api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/",
    callbacks=[handler],
)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

load_vector_store = Chroma(persist_directory="data/stores", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})

template = """Utiliza la siguiente información para responder a la pregunta del usuario.
Si no sabes la respuesta, di simplemente que no la sabes, no intentes inventarte una respuesta.

Contexto: {context}
Pregunta: {question}

Todos los datos sacados con los que respondas al usuario debes sacar los datos reales de la base de datos.

Solo si el usuario te pregunta especificando sobre algo de su nómina en concreto, debe darte su id_empleado y debes revisar su fila en la base de datos para poder explicarle el motivo de la pregunta que te haga. Para asegurarte que el usuario es quien dice ser, debes pedirle su id_empleado y comprobar que existe en la base de datos. Si no existe, debes decirle que no existe y que no puedes ayudarle. Si existe, debes responderle a su pregunta con los datos que has sacado de la base de datos.

No te inventes los datos que usas ya que podría haber muchos conflictos en la empresa. No uses la palabra "contexto" en tu respuesta. Usa un tono profesional y directo. No uses palabras como "creo" o "pienso". Usa siempre el mismo formato de respuesta.

Devuelve sólo la respuesta útil que aparece a continuación y nada más.
Responde siempre en castellano.
Respuesta útil:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

def get_response(query: str) -> str:
    return chain.invoke(query)

st.title("basketQuery")
st.markdown("Este es un chatbot RAG con entrada y salida de audio.")

# Estado para el chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte hoy?"}]

# Mostrar historial
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# --- NUEVA SECCIÓN: grabación de audio ---
st.markdown("### 🎙️ Hablar en lugar de escribir")
audio_bytes = audio_recorder()  # componente de streamlit_audio_recorder

user_input: str = ""
if audio_bytes:
    # Reproducimos lo que grabamos
    st.audio(audio_bytes, format="audio/wav")
    # Enviar a Whisper API
    transcription = openai.Audio.create_transcription(
        file=io.BytesIO(audio_bytes),
        model="whisper-1",
        response_format="text",
        language="es"
    )
    user_input = transcription.strip()
    st.write(f"**Transcripción:** {user_input}")

# --- FIN sección audio ---

# Entrada de texto (fallback / si no hay audio)
if not user_input:
    if prompt_txt := st.chat_input("Escribe tu mensaje aquí…"):
        user_input = prompt_txt

# Si tenemos alguna entrada (texto o audio)
if user_input:
    # Añadimos al historial
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Obtenemos respuesta del modelo
    response_text = get_response(user_input)

    # Añadimos al historial
    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)

    # --- NUEVA SECCIÓN: Text-to-Speech ---
    tts = gTTS(response_text, lang="es")
    tts_buffer = io.BytesIO()
    tts.write_to_fp(tts_buffer)
    tts_buffer.seek(0)
    st.audio(tts_buffer, format="audio/mp3")
    # --- FIN Text-to-Speech ---