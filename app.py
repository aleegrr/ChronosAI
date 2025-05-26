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
# Eliminada la sección ## SOURCE y simplificado el formato de respuesta
template_with_history = """
## SYSTEM ROLE
Eres un asistente virtual de Recursos Humanos especializado en consultas sobre nóminas. Tu objetivo es proporcionar información precisa y directa sobre las nóminas de los empleados o sobre la política general de nóminas de la empresa.

## USER QUESTION
El usuario ha preguntado:
"{question}"

## CHAT HISTORY
Aquí tienes el historial de la conversación:
'''
{chat_history}
'''

## CONTEXT
Aquí tienes la información relevante para responder a la pregunta:
'''
{context}
'''

## GUIDELINES
1.  **Prioridad y Extracción Exhaustiva del Contexto de Empleado**:
    * Si el `CONTEXT` proporcionado contiene **"Información del empleado:"** (es decir, datos detallados) y los datos específicos del empleado (ID y nombre) fueron detectados en la `USER QUESTION` o ya fueron establecidos y validados previamente en la conversación, **UTILIZA SIEMPRE LOS VALORES EXACTOS Y TAL CUAL APARECEN** en ese `CONTEXT` para responder a la pregunta actual.
    * Si el usuario pregunta por varios datos de su nómina (ej. salario base y antigüedad), busca y proporciona **TODOS** los datos relevantes que estén disponibles en la sección "Información del empleado" del `CONTEXT`.
    * No inventes, no aproximes, no asumas cálculos (ej. anual/mensual) si no se especifica explícitamente en el contexto.

2.  **Empleado Identificado, Esperando Consulta**:
    * Si el `CONTEXT` indica **"Empleado identificado: ID..."** (y **no** contiene "Información del empleado:"), significa que el empleado ha sido reconocido, pero el usuario no ha formulado una pregunta específica. En este caso, **pregunta al usuario en qué le puedes ayudar** con su nómina.

3.  **Solicitud de Datos (Solo si son realmente necesarios)**:
    * Si la pregunta se refiere a la nómina específica de un empleado, pero **NO se detectaron ID ni nombre en la pregunta actual**, y **NO hay un empleado validado en el estado de la conversación**, o **si el `CONTEXT` no pudo encontrar el empleado con los datos actuales**: solicita el **ID** y el **nombre completo del empleado**.
    * Si el `CONTEXT` ya contiene la "Información del empleado:", no vuelvas a pedir el ID o el nombre.


4.  **Precisión y Veracidad**:
    * Nunca inventes respuestas. Si no sabes algo o no está en el contexto, di que no puedes ayudar con eso.
    * Si el usuario pregunta por **cantidades en números**, asegúrate de que sean los números exactos que encuentras en la `Información del empleado` proporcionada. No inventes ni aproximes valores.
    * Si el usuario pregunta por un **documento específico**, asegúrate de que el documento exista en la base de datos y proporciona información relevante.

5.  **Idioma y Estilo**:
    * Siempre responde en **castellano**.
    * Sé **directo y útil**.

6.  **Transparencia (Limitada)**:
    * NO pongas la fuente de la información en la respuesta. Si el usuario pregunta por la fuente, indícale que no puedes proporcionar esa información.

## TASK
1.  Evalúa la `USER QUESTION` y el `CONTEXT` proporcionado.
2.  Determina si la pregunta es sobre una nómina específica de un empleado o sobre una política general de nóminas.
3.  Si es sobre una nómina específica y el `CONTEXT` ya contiene la "Información del empleado:", extrae y proporciona **TODOS** los detalles solicitados directamente de esa sección.
4.  **Si el `CONTEXT` solo indica que el empleado está identificado pero no contiene detalles específicos**, pregunta al usuario qué información desea.
5.  Responde de manera precisa y concisa, siguiendo todas las `GUIDELINES`.

# La respuesta debe ser solo el texto generado, sin ningún formato adicional ni título.
"""

prompt_with_history = ChatPromptTemplate.from_template(template_with_history)

def debug_print(data):
    print("DEBUG INPUT:", data.keys())
    return data

def format_prompt_input(data):
    print("INPUT TO FORMAT_PROMPT:", data)
    # No es necesario pasar 'source' al prompt si no se va a usar explícitamente o imprimir.
    # El modelo no necesita conocer la fuente de dónde vino el contexto en este punto.
    return {
        "context": data["context_data"]["context"],
        "question": data["question"],
        "chat_history": format_chat_history(data["chat_history"])
    }

chain_with_history = (
    RunnablePassthrough.assign(context_data=lambda x: get_context(x["question"]))
    | RunnableLambda(lambda x: (print("AFTER GET_CONTEXT:", x) or x)) # Para depuración
    | RunnableLambda(format_prompt_input)
    | prompt_with_history
    | model
    | StrOutputParser()
    # Eliminada la lambda que añadía la fuente al final de la respuesta
).with_config({"run_name": "chain_with_history"})

# --- FUNCIONES AUXILIARES ---

def extract_employee_id(text):
    # Ya tienes esta función, asegúrate de que sea robusta como la última vez que la mejoramos
    match = re.search(r"(?:id[ _-]?empleado\s*(?:es|:|indicado|igual)?\s*(\d+))", text, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"(?:ID|numero de empleado)\D*(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"(\d+)\D{0,20}(?:id|empleado)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def extract_employee_name(text):
    # Tu función extract_employee_name (asegúrate de que sea la última versión robusta)
    match = re.search(r"(?:nombre\s*[:=]?\s*|mi nombre es\s*)([\w\sÁÉÍÓÚÑñáéíóúüÜ'\.-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def get_context(query):
    id_empleado_from_query = extract_employee_id(query)
    nombre_from_query = extract_employee_name(query)

    final_id_empleado = id_empleado_from_query if id_empleado_from_query else st.session_state["current_employee_id"]
    final_nombre = nombre_from_query if nombre_from_query else st.session_state["current_employee_name"]

    print(f"DEBUG: ID de sesión/query usado: '{final_id_empleado}'")
    print(f"DEBUG: Nombre de sesión/query usado: '{final_nombre}'")

    # Flag para saber si la pregunta ya contiene una intención de consulta (ej. "salario base", "antigüedad")
    # Puedes expandir esta lista de palabras clave según las consultas que esperes.
    # La idea es que si la query no es SOLO "mi id es X y mi nombre es Y", se considera una pregunta.
    keywords = ["salario", "antiguedad", "departamento", "cargo", "irpf", "complemento", "plus", "horas", "dietas", "pagas", "convenio", "fecha alta", "cuenta bancaria", "cuanto", "que"]
    is_specific_query = any(keyword in query.lower() for keyword in keywords) or \
                        (id_empleado_from_query and nombre_from_query and len(query.split()) > 5) # Si tiene ID/Nombre y más de 5 palabras, asumimos que hay pregunta.


    if final_id_empleado and final_nombre:
        try:
            docs = employee_store.get(
                where={"id_empleado": final_id_empleado},
                include=["metadatas"]
            )
            
            print(f"DEBUG: Metadatos recuperados de Chroma: {docs.get('metadatas', 'Ninguno')}")

            if not docs["metadatas"]:
                st.session_state["current_employee_id"] = None
                st.session_state["current_employee_name"] = None
                return {
                    "context": f"No se encontraron datos con el ID de empleado '{final_id_empleado}'. Por favor, verifica el ID y nombre.",
                    "source": None
                }

            matched_metadata = None
            for metadata_item in docs["metadatas"]:
                if metadata_item.get("nombre", "").strip().lower() == final_nombre.strip().lower():
                    matched_metadata = metadata_item
                    break

            if matched_metadata:
                # ¡Éxito! Guardar en el estado para futuras consultas
                st.session_state["current_employee_id"] = final_id_empleado
                st.session_state["current_employee_name"] = final_nombre

                # --- LÓGICA CLAVE PARA CONTROLAR LA SALIDA ---
                if is_specific_query:
                    # Si hay una pregunta específica, formatear el contexto con los datos
                    context_lines = ["Información del empleado:"]
                    for k, v in matched_metadata.items():
                        context_lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
                    return {"context": "\n".join(context_lines), "source": "empleados"}
                else:
                    # Si solo se proporcionó ID/nombre, indicar que el empleado está validado
                    # y pedir al modelo que pregunte qué necesita.
                    return {
                        "context": f"Empleado identificado: ID {final_id_empleado}, Nombre: {final_nombre}. ¿En qué puedo ayudarte con tu nómina?",
                        "source": "empleados_identificado" # Nueva fuente para diferenciar
                    }
            else:
                st.session_state["current_employee_id"] = None
                st.session_state["current_employee_name"] = None
                return {
                    "context": "El nombre proporcionado no coincide con el ID de empleado. Por favor, asegúrate de que ambos datos sean correctos.",
                    "source": None
                }

        except Exception as e:
            st.session_state["current_employee_id"] = None
            st.session_state["current_employee_name"] = None
            return {
                "context": f"Ocurrió un error al intentar consultar los datos del empleado: {e}. Por favor, inténtalo de nuevo.",
                "source": None
            }

    # Si no hay ID ni nombre (ni en query ni en sesión), pedir ambos
    elif not final_id_empleado and not final_nombre:
        return {"context": "Para consultar la nómina específica de un empleado, necesito el **ID** y el **nombre completo** del empleado.", "source": None}
    
    # Si falta solo el ID o solo el nombre (pero no ambos)
    elif final_id_empleado and not final_nombre:
        return {"context": "Para consultar la nómina específica de un empleado, necesito también el **nombre completo** del empleado, además del ID.", "source": None}
    elif not final_id_empleado and final_nombre:
        return {"context": "Para consultar la nómina específica de un empleado, necesito también el **ID** del empleado, además del nombre.", "source": None}
    
    else:
        # Fallback a consulta general de políticas de nómina
        policy_docs = policy_store.similarity_search(query, k=2)
        context_value = policy_docs[0].page_content if policy_docs else "No se encontró información relevante sobre la política general de nóminas."
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
    print("QUERY RECIBIDA:", query) # Para depuración
    return chain_with_history.invoke({"question": query, "chat_history": chat_history})

# --- STREAMLIT INTERFAZ ---

st.cache_data.clear()
st.cache_resource.clear()
st.title("Chatbot de Nóminas")
st.subheader("Asistente de Recursos Humanos")
st.write("Este asistente está diseñado para responder preguntas relacionadas con nóminas y empleados.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte hoy?"}]

# Inicializar variables para almacenar el ID y nombre del empleado actual
if "current_employee_id" not in st.session_state:
    st.session_state["current_employee_id"] = None
if "current_employee_name" not in st.session_state:
    st.session_state["current_employee_name"] = None

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Excluimos el último mensaje (el actual del usuario) del historial que se pasa al modelo
    chat_history_for_model = st.session_state["messages"][:-1]

    response_content = get_response_with_history(prompt, chat_history_for_model)

    st.session_state["messages"].append({"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)