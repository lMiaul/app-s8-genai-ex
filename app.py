import streamlit as st
import pymongo
from google import genai
from google.genai import types
import numpy as np

# =======================
# CONFIGURACIÓN DE PÁGINA (Actualizado)
# =======================
st.set_page_config(
    page_title="Chat PDF - Filosofía de Aristóteles", 
    page_icon="🏛️",
    layout="wide"
)

# =======================
# SECRETS Y CLIENTES
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["pdf_embeddings_db"]
    return db["pdf_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# FUNCIONES
# =======================

def crear_embedding(texto: str):
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values

def buscar_similares(embedding, k=5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))

def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""Eres un asistente experto. Usa EXCLUSIVAMENTE el siguiente contexto para responder la pregunta del usuario. Si la respuesta no está en el contexto, indícalo claramente.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma concisa y clara en español."""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# =======================
# INTERFAZ STREAMLIT
# =======================

# Título principal actualizado
st.title("🏛️ Chatbot: Aristóteles y su Filosofía Práctica")

# --- NUEVO: BARRA LATERAL CON CONTEXTO Y PREGUNTAS ---
with st.sidebar:
    st.header("📖 Sobre este documento")
    st.write(
        "Este chatbot está entrenado con el fragmento del libro **'Descubrir la Filosofía - Aristóteles'** (páginas 76-91)."
    )
    st.write("**Temas principales:**")
    st.markdown(
        """
        - La Ética y la Política
        - La *Eudaimonia* (Felicidad como fin último)
        - La Virtud (*Areté*) adquirida por el hábito
        - La teoría del Justo Medio
        """
    )
    
    st.divider()
    
    st.subheader("💡 Preguntas de ejemplo")
    st.info("Copia y pega alguna de estas preguntas en el chat:")
    st.markdown(
        """
        1. ¿Qué es la felicidad o eudaimonia para Aristóteles?
        2. ¿En qué consiste la teoría del justo medio? Dame un ejemplo.
        3. ¿Por qué Aristóteles considera que el hombre es un "animal político"?
        4. ¿Cuál es la diferencia entre virtudes éticas y dianoéticas?
        5. ¿Cómo se llega a ser virtuoso según el texto?
        """
    )
# -----------------------------------------------------

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar historial PRIMERO
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])

pregunta = st.chat_input("Escribe tu pregunta sobre Aristóteles...")

if pregunta:
    # Mostrar inmediatamente la pregunta del usuario
    st.chat_message("user").write(pregunta)
    st.session_state.historial.append({"rol": "usuario", "texto": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Consultando los pergaminos..."):
            try:
                emb = crear_embedding(pregunta)
                similares = buscar_similares(emb, k=5)

                if not similares:
                    respuesta = "No encontré información relevante en el documento."
                else:
                    respuesta = generar_respuesta(pregunta, similares)
            except Exception as e:
                respuesta = f"⚠️ Ocurrió un error: {e}"

        st.write(respuesta)

        # Opcional: mostrar fuentes recuperadas
        if 'similares' in locals() and similares:
            with st.expander("🔍 Fragmentos recuperados del PDF"):
                for i, c in enumerate(similares, 1):
                    st.markdown(f"**Fragmento {i}** — score: `{c['score']:.4f}`")
                    st.write(c["texto"][:500] + ("…" if len(c["texto"]) > 500 else ""))
                    st.divider()

    st.session_state.historial.append({"rol": "bot", "texto": respuesta})
