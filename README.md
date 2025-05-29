# ChronosAI

**ChronosAI** es un asistente virtual inteligente diseñado para resolver de forma instantánea las dudas de los empleados sobre sus nóminas. Desde conceptos como salario base, horas extra, bonificaciones o vacaciones, hasta interpretaciones específicas de las políticas internas de la empresa, ChronosAI ofrece respuestas claras, precisas y personalizadas.

Este proyecto es un **MVP (Producto Mínimo Viable)** desarrollado como parte de un proyecto de fin de máster. Su objetivo es demostrar el potencial de la idea en un entorno simulado y servir como base para futuras fases de desarrollo.

---

## 🚀 Elevator Pitch

Cuando llega la nómina, lo que debería ser rutina se convierte en confusión para 7 de cada 10 empleados. Lo preocupante es que solo 3 de ellos preguntan. El resto se queda con la duda, lo que genera frustración, rotación y un equipo de Recursos Humanos saturado resolviendo lo mismo una y otra vez.

Ahí es donde entra **ChronosAI**: un asistente virtual inteligente que aclara al instante cualquier duda sobre nómina, horas extra, bonos o vacaciones. Basado en leyes laborales y políticas internas de cada empresa, ofrece respuestas contextualizadas para cada empleado.

Según nuestras simulaciones internas y feedback preliminar, ChronosAI podría reducir hasta un **60% de las consultas repetitivas** al equipo de RRHH, permitiendo que se centren en tareas de mayor valor.

---

## 🧠 ¿Cómo funciona?

El MVP está compuesto por varias etapas y componentes principales:

1. **Generación de datos ficticios (`creardb.ipynb`)**

   - Se genera una base de datos simulada con 150 empleados que incluye información como salario base, antigüedad, y otros datos relevantes.
   - Estos datos se guardan en la carpeta `data/`.

2. **Política de nóminas (`politicasNomina.md`)**

   - Un documento en formato Markdown que contiene las políticas internas de nómina de la empresa.
   - También se encuentra en la carpeta `data/`.

3. **Procesamiento de datos (`ingest.ipynb`)**

   - Utilizando Chroma y técnicas de Recuperación Aumentada por Generación (RAG), se crean dos bases vectoriales:
     - Una para las políticas de nómina.
     - Otra para los datos de empleados.
   - Los vectores generados se almacenan en `data/stores/`.

4. **Aplicación final (`app.py`)**
   - Una interfaz desarrollada con Streamlit que permite al usuario interactuar con el asistente virtual.
   - El modelo de lenguaje, potenciado por OpenAI, responde preguntas tanto generales (sobre las políticas) como específicas (casos individuales de empleados).

---

## 💡 Propósito del proyecto

Este MVP ha sido concebido como una **prueba de concepto**. El objetivo no es ofrecer un producto final, sino validar una idea con alto potencial: automatizar la resolución de dudas relacionadas con la nómina, una necesidad recurrente y poco resuelta en muchas empresas.

ChronosAI representa una visión clara: **ser el copiloto inteligente de referencia para PYMEs y equipos de RRHH en toda España**, facilitando una comunicación transparente y eficiente entre empresa y empleado.

---

## 📁 Estructura del proyecto

```
ChronosAI/
├── app.py                               # Aplicación principal (Streamlit + OpenAI)
├── creardb.ipynb                        # Generación de la base de datos ficticia
├── ingest.ipynb                         # Ingesta de datos y creación de RAGs con Chroma
├── requirements.txt                     # Requisitos del proyecto para entorno virtual
├── ChronosAI_Elevator_Pitch_Formal.pdf  # Pitch formal del proyecto
└── data/
    ├── empleados.csv                    # Base de datos simulada de empleados
    ├── politicasNomina.md               # Políticas internas de nómina
    └── stores/                          # Bases vectoriales generadas (Chroma)
```

---

## 🧪 Estado actual

- ✅ Datos de ejemplo generados.
- ✅ Documento de políticas preparado.
- ✅ Vector stores funcionales.
- ✅ Interfaz de chatbot básica implementada.

**Próximos pasos**: Validación con usuarios reales, integración con sistemas existentes y mejoras en personalización de respuestas.

---

## 📌 Notas finales

ChronosAI no solo responde preguntas: **transforma la relación entre empresa y empleado**, haciendo de la confianza la nueva norma.  
Este proyecto demuestra que es posible mejorar la experiencia del empleado mientras se optimiza el tiempo del equipo de RRHH.
