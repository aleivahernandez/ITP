import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io
import html
from gtts import gTTS
from io import BytesIO
from googletrans import Translator ### NUEVO ###

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Brújula Tecnológica Territorial")

# Custom CSS (sin cambios, se omite por brevedad)
st.markdown(
    """
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0f69b4;
        }
        .stApp {
            max-width: 800px;
            margin: 2rem auto;
            background-color: #ffffff;
            border-radius: 1.5rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            padding: 2.5rem;
        }
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > div[data-testid="stTextArea"] {
            border-radius: 9999px !important;
            border: 1px solid #d1d5db !important;
            padding: 0.5rem 1.5rem !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
            font-size: 1.125rem !important;
            margin-bottom: 1rem;
            resize: none !important;
        }
        button[data-testid="stFormSubmitButton"] {
            background-color: #20c997 !important;
            color: white !important;
            border-radius: 0.75rem !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            transition: background-color 0.2s ease !important;
            display: block !important;
            margin: 0 auto 2rem auto !important;
            width: fit-content;
        }
        button[data-testid="stFormSubmitButton"]:hover {
            background-color: #1aae89 !important;
        }
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > label[data-testid="stWidgetLabel"] {
            display: none !important;
        }
        .google-patent-result-container {
            background-color: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        .result-header {
            display: flex;
            align-items: flex-start;
            margin-bottom: 0.5rem;
            gap: 1rem;
        }
        .result-image-wrapper {
            flex-shrink: 0;
            width: 80px;
            height: 80px;
            border-radius: 4px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f0f0f0;
        }
        .result-image {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 4px;
        }
        .result-text-content {
            flex-grow: 1;
        }
        .result-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #1a0dab;
            line-height: 1.3;
            margin-bottom: 0.4rem;
        }
        .result-summary {
            font-size: 0.9rem;
            color: #4d5156;
            margin-bottom: 0.5rem;
            line-height: 1.5;
        }
        .result-meta {
            font-size: 0.8rem;
            color: #70757a;
        }
        .similarity-score-display {
            font-size: 0.8rem;
            font-weight: 600;
            color: #ffffff; 
            margin-left: auto;
            background-color: #eb3c46; /* <-- COLOR CAMBIADO */
            padding: 0.15rem 0.4rem;
            border-radius: 0.4rem;
        }
        
        /* --- ESTILOS PARA VISTA DE DETALLE --- */
        .full-patent-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1a0dab;
            margin-bottom: 0;
        }
        .detail-subtitle {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
            padding-top: 8px; /* Alineación vertical con el botón */
        }
        .full-patent-abstract {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #333333;
            text-align: justify;
        }
        .full-patent-meta {
            font-size: 0.9rem;
            color: #70757a;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Functions for loading and processing data/models ---

@st.cache_resource
def load_embedding_model():
    """Carga el modelo pre-entrenado SentenceTransformer."""
    with st.spinner("Cargando modelo de semántica..."):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

### NUEVO ###
@st.cache_data
def translate_text_to_spanish(text, translator):
    """Traduce texto a español, detectando el idioma de origen."""
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        # Detecta el idioma, si ya es español, lo devuelve directamente
        if translator.detect(text).lang == 'es':
            return text
        # Si no, lo traduce
        return translator.translate(text, dest='es').text
    except Exception as e:
        # En caso de error (ej. texto muy corto, error de API), devuelve el original
        print(f"Error de traducción: {e}")
        return text

@st.cache_data
def process_patent_data(file_path):
    """Procesa el archivo de patentes, traduce y genera los embeddings."""
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                st.error("Formato de archivo no soportado. Sube un .csv o .xlsx.")
                return None, None

            df.columns = df.columns.str.strip().str.lower()
            required_cols = ['title (original language)', 'abstract (original language)', 'publication number']
            if not all(col in df.columns for col in required_cols):
                st.error(f"El archivo debe contener las columnas: {', '.join(required_cols)}")
                return None, None

            for col in required_cols:
                df[col] = df[col].fillna('')

            ### --- NUEVA SECCIÓN DE TRADUCCIÓN --- ###
            with st.spinner("Traduciendo textos al español (esto puede tardar unos minutos)..."):
                translator = Translator()
                progress_bar = st.progress(0, text="Traduciendo títulos...")
                total_rows = len(df)

                # Usamos una función lambda para actualizar la barra de progreso
                df['title_es'] = df['title (original language)'].apply(
                    lambda x: translate_text_to_spanish(x, translator)
                )
                progress_bar.progress(50, text="Traduciendo resúmenes...")
                df['abstract_es'] = df['abstract (original language)'].apply(
                    lambda x: translate_text_to_spanish(x, translator)
                )
                progress_bar.progress(100, text="Traducción completada.")
                progress_bar.empty() # Limpia la barra de progreso
            ### --- FIN SECCIÓN DE TRADUCCIÓN --- ###

            github_image_base_url = "https://raw.githubusercontent.com/aleivahernandez/ITP_v2/main/images/"
            df['image_url_processed'] = df['publication number'].apply(
                lambda x: f"{github_image_base_url}{x}.png" if x else ""
            )

            ### MODIFICADO ###: Usar las columnas traducidas para el embedding
            df['Descripción Completa'] = df['title_es'] + ". " + df['abstract_es']
            model = load_embedding_model()
            corpus_embeddings = model.encode(df['Descripción Completa'].tolist(), convert_to_tensor=True)
            return df, corpus_embeddings
        except FileNotFoundError:
            st.error(f"Error: No se encontró el archivo '{file_path}'.")
            return None, None
        except Exception as e:
            st.error(f"Error al procesar el archivo '{file_path}': {e}")
            return None, None
    return None, None

# --- Automatic local Excel file loading section ---
excel_file_name = "patentes.xlsx"
df_patents, patent_embeddings = process_patent_data(excel_file_name)

if df_patents is None:
    st.error("La aplicación no puede continuar sin la base de datos de patentes.")
    st.stop()

# --- Session State Initialization ---
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'search'
if 'selected_patent' not in st.session_state:
    st.session_state.selected_patent = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'query_description' not in st.session_state:
    st.session_state.query_description = "Certificación calidad de miel."

# --- Functions for view management ---
def show_search_view():
    st.session_state.current_view = 'search'
    st.session_state.selected_patent = None

def show_patent_detail(patent_data):
    st.session_state.current_view = 'detail'
    st.session_state.selected_patent = patent_data

# --- Main Application Logic ---
if st.session_state.current_view == 'search':
    st.markdown("<h1 style='font-size: 1.75rem; font-weight: 700; margin-bottom: 1rem;'>Brújula Tecnológica Territorial</h1>", unsafe_allow_html=True)

    with st.form(key='search_form'):
        problem_description = st.text_area(
            "Describe tu problema técnico o necesidad funcional:",
            value=st.session_state.query_description,
            height=68,
            key="problem_description_input_area",
            placeholder="Escribe aquí tu necesidad apícola..."
