import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse.callback import CallbackHandler

# Cargar variables de entorno
load_dotenv()
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]

# Callback de Langfuse
handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY)

# Modelo
model = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=1024,
    openai_api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/",
    callbacks=[handler],
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# Cargar vector stores
policy_store = Chroma(persist_directory="data/stores/nominas", embedding_function=embeddings)
employee_store = Chroma(persist_directory="data/stores/empleados", embedding_function=embeddings)

# Prompt template con historial de conversación
template_with_history = """
Eres un asistente virtual de Recursos Humanos especializado en consultas sobre nóminas.

Reglas importantes:
- Si la pregunta se refiere a la nómina específica de un empleado (ej. "¿cuánto he cobrado?", "¿por qué mi nómina es así?"), solicita SIEMPRE el ID y el nombre del empleado antes de intentar responder. Indica claramente que necesitas esta información para poder consultar sus datos específicos.
- Si la pregunta contiene explícitamente un ID de empleado Y un nombre de empleado, busca en la siguiente información del empleado y responde a la pregunta utilizando los valores exactos que encuentres allí.
- Si la pregunta NO contiene un ID de empleado ni se refiere explícitamente a una nómina personal, asume que la consulta es sobre la política general de nóminas y busca la respuesta en la base de datos de 'nominas'.
- Nunca inventes respuestas. Si no sabes algo o no está en el contexto, di que no puedes ayudar con eso.
- Siempre responde en castellano.
- Sé directo y útil.
- Si el usuario pregunta por cantidades en números, asegúrate de que sean los números exactos que encuentras en la información del empleado, no inventes ni approximes valores.
- Si el usuario pregunta por un documento específico, asegúrate de que el documento exista en la base de datos y proporciona información relevante.
- Si el usuario proporciona un ID y un nombre pero alguno de los dos no coincide con un registro de la base de datos, indica que no puedes ayudar y sugiere reformular la pregunta.
- NO pongas la fuente de la información en la respuesta. Si el usuario pregunta por la fuente, indícale que no puedes proporcionar esa información.


Historial de conversación:
{chat_history}

Información del empleado:
{context}

Fuente de la información:
{source}

Pregunta del usuario:
{question}

Respuesta útil:
"""

prompt_with_history = ChatPromptTemplate.from_template(template_with_history)

def debug_print(data):
    print("DEBUG INPUT:", data.keys())
    return data

def format_prompt_input(data):
    print("INPUT TO FORMAT_PROMPT:", data)
    return {
        "context": data["context_data"]["context"],
        "source": data["context_data"]["source"],
        "question": data["question"],
        "chat_history": format_chat_history(data["chat_history"])
    }

chain_with_history = (
    RunnablePassthrough.assign(context_data=lambda x: get_context(x["question"]))
    | RunnableLambda(lambda x: (print("AFTER GET_CONTEXT:", x) or x))
    | RunnableLambda(format_prompt_input)
    | prompt_with_history
    | model
    | StrOutputParser()
    | (lambda output: f"{output} (Fuente: {output.get('source')})" if isinstance(output, dict) and output.get('source') else output)
).with_config({"run_name": "chain_with_history"})

# --- FUNCIONES AUXILIARES ---

def extract_employee_id(text):
    match = re.search(r"id[_ ]?empleado\s*[:=]?\s*(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_employee_name(text):
    match = re.search(r"nombre\s*[:=]?\s*([a-zA-ZÁÉÍÓÚÑñáéíóúüÜ\s]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def get_context(query):
    id_empleado = extract_employee_id(query)
    nombre = extract_employee_name(query)

    if id_empleado and nombre:
        try:
            # Filtrar por metadatos usando Chroma
            docs = employee_store.get(
                where={"id_empleado": int(id_empleado)},
                include=["metadatas", "documents"]
            )

            if not docs["metadatas"]:
                return {
                    "context": "No se encontraron datos con ese ID de empleado.",
                    "source": None
                }

            for metadata in docs["metadatas"]:
                if metadata["nombre"].strip().lower() == nombre.strip().lower():
                    context = "\n".join(
                        f"{k.replace('_', ' ').title()}: {v}" for k, v in metadata.items()
                    )
                    return {"context": context, "source": "empleados"}

            return {
                "context": "El nombre no coincide con el ID de empleado proporcionado. Por favor, asegúrate de que ambos datos sean correctos.",
                "source": None
            }

        except Exception as e:
            return {
                "context": f"Error accediendo a la base de datos de empleados: {e}",
                "source": None
            }

    elif id_empleado and not nombre:
        return {"context": "Por favor, proporciona también tu nombre para consultar tu información específica.", "source": None}
    elif not id_empleado and nombre:
        return {"context": "Por favor, proporciona también tu ID de empleado para consultar tu información específica.", "source": None}
    else:
        # Consulta general: ir a la política
        policy_docs = policy_store.similarity_search(query, k=2)
        context_value = policy_docs[0].page_content if policy_docs else "No se encontró información sobre la política de nóminas."
        source_value = "nominas" if policy_docs else None
        return {"context": context_value, "source": source_value}

def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        if message["role"] == "user":
            formatted_history += f"Usuario: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_history += f"Asistente: {message['content']}\n"
    return formatted_history.strip()

def get_response_with_history(query, chat_history):
    print("QUERY RECIBIDA:", query)
    return chain_with_history.invoke({"question": query, "chat_history": chat_history})

# --- STREAMLIT INTERFAZ ---

st.cache_data.clear()
st.cache_resource.clear()
st.title("Chatbot de Nóminas")
st.subheader("Asistente de Recursos Humanos")
st.write("Este asistente está diseñado para responder preguntas relacionadas con nóminas y empleados.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte hoy?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    chat_history = st.session_state["messages"][:-1]  # esto está bien

    response_data = get_response_with_history(prompt, chat_history)  # también está bien
    response_content = response_data

    st.session_state["messages"].append({"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)