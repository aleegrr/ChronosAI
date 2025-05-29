import os
# MODIFICACIÓN PARA STREAMLIT
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import re
import unicodedata

# INICIO DEL PARCHE PARA SQLITE3 COMPATIBLE CON CHROMADB
# Las líneas de print y st.info son para depuración, puedes quitarlas una vez funcione
print("Intentando parchear pysqlite3...")
# st.info("Intentando parchear pysqlite3...") # Si no se muestra, no importa mucho
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Parche pysqlite3 aplicado con éxito.")
    # st.info("Parche pysqlite3 aplicado con éxito.")
except ImportError as e:
    print(f"ERROR: Fallo al importar pysqlite3: {e}. Asegúrate de que 'pysqlite3-binary' está en requirements.txt")
    # st.error(f"ERROR: Fallo al importar pysqlite3: {e}. Asegúrate de que 'pysqlite3-binary' está en requirements.txt")
    st.stop() # Detenemos la ejecución si el parche falla
# FIN DEL PARCHE

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from langchain.embeddings import FastEmbedEmbeddings
from langfuse.callback import CallbackHandler

# Cargar variables de entorno
load_dotenv()
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']
LANGFUSE_PUBLIC_KEY = os.environ['LANGFUSE_PUBLIC_KEY']
LANGFUSE_SECRET_KEY = os.environ['LANGFUSE_SECRET_KEY']

# Callback de Langfuse
handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY)

# Modelo
model = ChatOpenAI(
    model='mistralai/Mixtral-8x7B-Instruct-v0.1',
    temperature=0,
    max_tokens=1024,
    openai_api_key=TOGETHER_API_KEY,
    base_url='https://api.together.xyz/',
    callbacks=[handler],
)

# Embeddings
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Cargar vector stores
try:
    policy_store = Chroma(persist_directory='data/stores/nominas', embedding_function=embeddings)
    employee_store = Chroma(persist_directory='data/stores/empleados', embedding_function=embeddings)
except Exception as e:
    st.error(f'Error al cargar las bases de datos vectoriales. Asegúrate de que los directorios existan y contengan datos válidos: {e}')
    st.stop()


# Prompt template con historial de conversación
template_with_history = '''
## SYSTEM ROLE
Eres un asistente virtual de Recursos Humanos especializado en consultas sobre nóminas. Tu objetivo es proporcionar información precisa y directa sobre las nóminas de los empleados o sobre la política general de nóminas de la empresa.
**IMPERATIVO: RESPONDE SIEMPRE Y ÚNICAMENTE EN CASTELLANO.**

## USER QUESTION
El usuario ha preguntado:
'{question}'

## CHAT HISTORY
Aquí tienes el historial de la conversación:
"""
{chat_history}
"""

## CONTEXT
Aquí tienes la información relevante para responder a la pregunta:
"""
{context}
"""

## GUIDELINES
1.  **Prioridad y Extracción Exhaustiva del Contexto de Empleado**:
    * Si el `CONTEXT` proporcionado contiene **'Información del empleado:'** (es decir, datos detallados del empleado) y los datos específicos del empleado (ID y nombre) fueron detectados en la `USER QUESTION` o ya fueron establecidos y validados previamente en la conversación, **UTILIZA SIEMPRE LOS VALORES EXACTOS Y TAL CUAL APARECEN** en ese `CONTEXT` para responder a la pregunta actual.
    * Si el usuario pregunta por varios datos de su nómina (ej. salario base y antigüedad), busca y proporciona **TODOS** los datos relevantes que estén disponibles en la sección 'Información del empleado' del `CONTEXT`.
    * No inventes, no aproximes, no asumas cálculos (ej. anual/mensual) si no se especifica explícitamente en el contexto.

2.  **Manejo de Solicitud y Validación de Datos del Empleado**:
    * Si el `CONTEXT` contiene el mensaje **'Para poder consultar su nómina, necesito...'** (falta un dato) o **'Los datos proporcionados no coinciden.'** o **'No se encontraron datos con el ID de empleado...'** (datos incorrectos), tu única respuesta debe ser **indicar directamente que los datos son incorrectos y solicitar la información correcta o faltante, especificando cuál es el problema (ej. 'el nombre no corresponde al ID'). No pidas al usuario que 'verifique' su propio dato, simplemente indícale que los datos son incorrectos y cuáles son.** Es crucial que persistas en solicitar o corregir los datos hasta que sean válidos.
    * Si el `CONTEXT` indica **'Empleado identificado: ID...'** (y **no** contiene 'Información del empleado:' ni un mensaje de error de validación), significa que el empleado ha sido reconocido y validado correctamente y no hay una pregunta específica sobre su nómina en la consulta actual. En este caso, tu respuesta debe ser **preguntar al usuario específicamente en qué le puedes ayudar** con su nómina, sin proporcionar ningún dato del empleado todavía.

3.  **Preguntas de Política General**:
    * Si la pregunta es sobre la política de nóminas en general y el `CONTEXT` contiene información relevante (que no sea 'Información del empleado:' o mensajes de solicitud/error de datos de empleado), responde directamente basándote en esa información.

4.  **Precisión y Veracidad**:
    * Nunca inventes respuestas. Si no sabes algo o no está en el contexto, di que no puedes ayudar con eso.
    * Si el usuario pregunta por **cantidades en números**, asegúrate de que sean los números exactos que encuentras en la `Información del empleado` proporcionada. No inventes ni aproximes valores.

5.  **Prioridad de Respuesta de Datos Específicos Tras Validación (¡IMPORTANTE!)**:
    * Si el `CONTEXT` contiene 'Información del empleado:' Y la `USER QUESTION` es una pregunta clara sobre esa información, **DEBES responder directamente utilizando la 'Información del empleado' del CONTEXT. La respuesta debe ser la información solicitada de forma concisa y directa, como 'Tu salario base es de [cantidad].' o 'Su salario base es de [cantidad].'.** Opcionalmente, al final de la respuesta, *puedes* añadir una pregunta concisa como '¿Hay algo más en lo que pueda ayudarte con tu nómina?' o '¿Deseas saber algo más?'.

6.  **Idioma y Estilo**:
    * **RESPONDE SIEMPRE EN CASTELLANO.**
    * Sé **directo y útil**.

7.  **Restricciones de Contenido**:
    * **Nunca incluyas la fuente de la información en tu respuesta.**
    * **No incluyas ninguna nota o comentario sobre cómo estás siguiendo las directrices o por qué no proporcionas cierta información.** Simplemente sigue las reglas de forma silenciosa.
    * **Nunca incluyas información personal sensible** (como números de identificación, direcciones, etc.) en tus respuestas.
    * **No saludes en cada respuesta al empleado, solo contesta a lo que te pida**

## TASK
1.  Evalúa la `USER QUESTION` y el `CONTEXT` proporcionado.
2.  Determina el estado de la conversación y la intencionalidad del usuario basándote en el `CONTEXT`.
3.  **Prioriza estrictamente:** Si el `CONTEXT` indica que se necesitan datos del empleado (ID/nombre) o que los proporcionados son incorrectos, tu respuesta debe centrarse *únicamente* en solicitar o corregir esos datos. Persiste hasta que sean válidos.
4.  Si el empleado ya ha sido identificado y validado, y el `CONTEXT` NO contiene 'Información del empleado:', **pregunta al usuario en qué le puedes ayudar con su nómina**.
5.  Si el empleado ya ha sido identificado y validado, Y la `USER QUESTION` es una consulta específica sobre su nómina, Y el `CONTEXT` proporciona 'Información del empleado:', extrae y proporciona **TODOS** los detalles solicitados directamente de esa sección. **La respuesta debe ser la información solicitada de forma concisa y directa, sin preámbulos ni re-preguntas. Opcionalmente, puedes finalizar con una pregunta muy breve para ofrecer más ayuda.**
6.  Si la pregunta es sobre política general, responde usando el `CONTEXT` de políticas.
7.  Responde de manera precisa y concisa, siguiendo todas las `GUIDELINES`. **No añadas comentarios sobre las directrices que estás siguiendo.**
8.  **Responde siempre en CASTELLANO** y evita cualquier formato adicional o etiquetas innecesarias.
9. La salida debe ser solamente la respuesta textual directa al usuario, sin añadir notas, justificaciones, explicaciones entre paréntesis, aclaraciones del modelo, ni ningún otro contenido adicional. La respuesta debe parecer que fue escrita por un humano experto, sin mencionar que es un modelo ni aludir a las instrucciones seguidas.

# La respuesta debe ser solo el texto generado, sin ningún formato adicional ni título.
'''

prompt_with_history = ChatPromptTemplate.from_template(template_with_history)

def format_prompt_input(data):
    return {
        'context': data['context_data']['context'],
        'question': data['question'],
        'chat_history': format_chat_history(data['chat_history'])
    }

chain_with_history = (
    RunnablePassthrough.assign(context_data=lambda x: get_context(x['question']))
    | RunnableLambda(format_prompt_input)
    | prompt_with_history
    | model
    | StrOutputParser()
).with_config({'run_name': 'chain_with_history'})

# --- FUNCIONES AUXILIARES ---

# Helper para normalizar texto (quitar tildes)
def normalize_text(text):
    if text is None:
        return None
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar tildes y otros diacríticos
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def extract_employee_id(text):
    match = re.search(r'(?:id[ _-]?empleado\s*(?:es|:|indicado|igual)?\s*(\d+))', text, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r'(?:ID|numero de empleado)\D*(\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r'(\d+)\D{0,20}(?:id|empleado)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def extract_employee_name(text):
    match = re.search(r'(?:nombre\s*[:=]?\s*|mi nombre es\s*)([\w\sÁÉÍÓÚÑñáéíóúüÜ"\.-]+?)(?:\s*y mi id|\s*y mi ID|\s*mi id|\s*mi ID|\s*mi identidad|\s*y mi numero de identificacion|$)', text, re.IGNORECASE | re.UNICODE)
    if match:
        return match.group(1).strip()
    return None

def get_context(query):
    # 1. Extraer ID y Nombre de la query actual
    id_from_current_query = extract_employee_id(query)
    name_from_current_query = extract_employee_name(query)

    # 2. Obtener el estado actual de la sesión
    stored_id = st.session_state.get('current_employee_id')
    stored_name = st.session_state.get('current_employee_name')
    employee_validated = st.session_state.get('employee_validated', False)

    # 3. Determinar el ID y Nombre que usaremos para la validación/consulta
    final_id = id_from_current_query if id_from_current_query else stored_id
    final_name = name_from_current_query if name_from_current_query else stored_name

    # 4. Actualizar session_state con los datos extraídos de la query actual
    if id_from_current_query is not None:
        st.session_state['current_employee_id'] = id_from_current_query
    if name_from_current_query is not None:
        st.session_state['current_employee_name'] = name_from_current_query

    # 5. Resetear la validación si se proporciona un nuevo dato en la query
    if (id_from_current_query is not None and id_from_current_query != stored_id) or \
       (name_from_current_query is not None and normalize_text(name_from_current_query) != normalize_text(stored_name)):
        st.session_state['employee_validated'] = False
        employee_validated = False # Actualizar la variable local también

    # 6. Detectar si la intención es sobre un empleado específico
    keywords_employee_data = ['salario', 'antiguedad', 'departamento', 'cargo', 'irpf', 'complemento', 'plus', 'horas', 'dietas', 'pagas', 'convenio', 'fecha alta', 'cuenta bancaria', 'cuanto', 'que me expliques', 'mi nomina', 'mi salario', 'mi sueldo']
    is_employee_specific_query_intent = any(keyword in query.lower() for keyword in keywords_employee_data) or \
                                        (id_from_current_query is not None or name_from_current_query is not None) or \
                                        employee_validated

    # 7. Lógica de flujo principal para preguntas sobre empleados
    if is_employee_specific_query_intent:
        # A. Si faltan datos necesarios para la validación, pedirlos
        if not final_id or not final_name:
            missing_info = []
            if not final_id:
                missing_info.append('el **ID** de empleado')
            if not final_name:
                missing_info.append('el **nombre completo**')
                return {'context': f'Para poder consultar su nómina, necesito {", y ".join(missing_info)}.', 'source': 'request_credentials'}

        # B. Si tenemos ambos datos (final_id y final_name), intentar validar/consultar
        try:
            docs = employee_store.get(
                where={'id_empleado': final_id},
                include=['metadatas']
            )

            if not docs['metadatas']:
                st.session_state['employee_validated'] = False
                return {
                    'context': f'Los datos proporcionados no coinciden. No se encontró ningún empleado con el ID "{final_id}". Por favor, verifica el ID y tu nombre completo.',
                    'source': 'invalid_credentials'
                }

            matched_metadata = None
            for metadata_item in docs['metadatas']:
                if normalize_text(metadata_item.get('nombre', '')) == normalize_text(final_name):
                    matched_metadata = metadata_item
                    break

            if matched_metadata:
                st.session_state['current_employee_id'] = final_id
                st.session_state['current_employee_name'] = final_name
                st.session_state['employee_validated'] = True

                is_current_query_a_data_question = any(keyword in query.lower() for keyword in keywords_employee_data)

                if is_current_query_a_data_question:
                    context_lines = ['Información del empleado:']
                    for k, v in matched_metadata.items():
                        context_lines.append(f'- **{k.replace("_", " ").title()}**: {v}')
                    return {'context': '\n'.join(context_lines), 'source': 'empleados_data'}
                else:
                    return {
                        'context': f'Empleado identificado: ID {final_id}, Nombre: {final_name}. ¿En qué puedo ayudarte con tu nómina?',
                        'source': 'empleados_identified_no_question'
                    }
            else:
                st.session_state['employee_validated'] = False
                return {
                    'context': 'Los datos proporcionados no coinciden. El nombre que has dado no corresponde al ID de empleado. Por favor, asegúrate de que ambos datos sean correctos.',
                    'source': 'invalid_credentials'
                }

        except Exception as e:
            st.session_state['employee_validated'] = False
            return {
                'context': f'Ocurrió un error al intentar consultar los datos del empleado: {e}. Por favor, inténtalo de nuevo.',
                'source': 'error'
            }
    else:
        policy_docs = policy_store.similarity_search(query, k=2)
        context_value = policy_docs[0].page_content if policy_docs else 'No se encontró información relevante sobre la política general de nóminas.'
        source_value = 'nominas_policy' if policy_docs else None
        return {'context': context_value, 'source': source_value}


def format_chat_history(chat_history):
    formatted_history = ''
    for message in chat_history:
        if message['role'] == 'user':
            formatted_history += f'Usuario: {message["content"]}\n'
        elif message['role'] == 'assistant':
            formatted_history += f'Asistente: {message["content"]}\n'
    return formatted_history.strip()

def get_response_with_history(query, chat_history):
    return chain_with_history.invoke({'question': query, 'chat_history': chat_history})

# --- STREAMLIT INTERFAZ ---

st.cache_data.clear()
st.cache_resource.clear()

st.title('Chatbot de Nóminas')
st.subheader('Asistente de Recursos Humanos')
st.write('Este asistente está diseñado para responder preguntas relacionadas con nóminas y empleados.')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': '¡Hola! ¿En qué puedo ayudarte hoy?'}]

if 'current_employee_id' not in st.session_state:
    st.session_state['current_employee_id'] = None
if 'current_employee_name' not in st.session_state:
    st.session_state['current_employee_name'] = None
if 'employee_validated' not in st.session_state:
    st.session_state['employee_validated'] = False


for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input('Escribe tu consulta aquí...'):
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    st.chat_message('user').write(prompt)

    chat_history_for_model = st.session_state['messages'][:-1]

    response_content = get_response_with_history(prompt, chat_history_for_model)

    st.session_state['messages'].append({'role': 'assistant', 'content': response_content})
    st.chat_message('assistant').write(response_content)