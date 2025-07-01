import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io
import html

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Brújula Tecnológica Territorial")

# Custom CSS for a better visual match to Google Patents style
st.markdown(
    """
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #e0f2f7; /* Light blue background for the page */
        }
        /* El contenedor principal de la aplicación Streamlit. */
        .stApp {
            max-width: 800px; /* Constrain the app width */
            margin: 2rem auto; /* Center the app on the page */
            background-color: #ffffff; /* White background for the app container */
            border-radius: 1.5rem; /* Rounded Corners */
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1); /* Soft shadow */
            padding: 2.5rem; /* Padding inside the app container */
        }
        /* Style for the main search input container */
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > div[data-testid="stTextArea"] {
            border-radius: 9999px !important; /* Fully rounded */
            border: 1px solid #d1d5db !important; /* Light gray border */
            padding: 0.5rem 1.5rem !important; /* Adjust padding */
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
            font-size: 1.125rem !important; /* text-lg */
            margin-bottom: 1rem; /* Space below the input */
            resize: none !important; /* Prevent manual resizing */
        }
        /* Style for the submit button */
        button[data-testid="stFormSubmitButton"] {
            background-color: #20c997 !important;
            color: white !important;
            border-radius: 0.75rem !important; /* Rounded corners */
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            transition: background-color 0.2s ease !important;
            display: block !important; /* Make it a block element */
            margin: 0 auto 2rem auto !important; /* Center the button and add margin below */
            width: fit-content; /* Adjust width to content */
        }
        button[data-testid="stFormSubmitButton"]:hover {
            background-color: #1aae89 !important;
        }
        /* Hide the default label for st.text_area */
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > label[data-testid="stWidgetLabel"] {
            display: none !important;
        }

        /* --- Google Patents style for patent results (Search View) --- */
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
            color: #20c997;
            margin-left: auto;
            background-color: #e0f2f7;
            padding: 0.15rem 0.4rem;
            border-radius: 0.4rem;
        }

        /* --- Styles for the full patent view (Detail View) --- */
        .full-patent-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1a0dab;
            margin-bottom: 1rem;
        }
        .detail-subtitle {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
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
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Functions for loading and processing data/models ---

@st.cache_resource
def load_embedding_model():
    """Carga el modelo pre-entrenado SentenceTransformer."""
    with st.spinner("Cargando el modelo de embeddings..."):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

@st.cache_data
def process_patent_data(file_path):
    """Procesa el archivo de patentes y genera los embeddings."""
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

            github_image_base_url = "https://raw.githubusercontent.com/aleivahernandez/ITP_v2/main/images/"
            df['image_url_processed'] = df['publication number'].apply(
                lambda x: f"{github_image_base_url}{x}.png" if x else ""
            )

            df['Descripción Completa'] = df['title (original language)'] + ". " + df['abstract (original language)']
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
        )
        submitted = st.form_submit_button("Buscar Soluciones")

        if submitted:
            st.session_state.query_description = problem_description.strip()
            if not st.session_state.query_description:
                st.warning("Por favor, ingresa una descripción del problema.")
                st.session_state.search_results = []
            else:
                with st.spinner("Buscando patentes relevantes..."):
                    model = load_embedding_model()
                    query_embedding = model.encode(st.session_state.query_description, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(query_embedding, patent_embeddings)[0]
                    top_indices = np.argsort(-cosine_scores.cpu().numpy())[:3]

                    results = []
                    for idx in top_indices:
                        patent_data = df_patents.iloc[idx]
                        results.append({
                            'title': patent_data['title (original language)'],
                            'abstract': patent_data['abstract (original language)'],
                            'publication_number': patent_data['publication number'],
                            'image_url': patent_data['image_url_processed'],
                            'score': cosine_scores[idx].item()
                        })
                    st.session_state.search_results = results

    if st.session_state.search_results:
        st.markdown("---")
        st.subheader("Resultados de la Búsqueda:")
        for i, patent in enumerate(st.session_state.search_results):
            escaped_title = html.escape(patent['title'])
            escaped_summary = html.escape(patent['abstract'][:120]) + "..."
            default_image = "https://placehold.co/120x120/cccccc/000000?text=No+Image"

            st.markdown(f"""
            <div class="google-patent-result-container">
                <div class="result-header">
                    <div class="result-image-wrapper">
                        <img src="{patent['image_url'] or default_image}" class="result-image" onerror="this.onerror=null;this.src='{default_image}';">
                    </div>
                    <div class="result-text-content">
                        <h3 class="result-title">{escaped_title}</h3>
                        <p class="result-summary">{escaped_summary}</p>
                        <p class="result-meta">Patente: {patent['publication_number']} <span class="similarity-score-display">Similitud: {patent['score']:.2%}</span></p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.button("Ver Detalles Completos", key=f"view_{i}", on_click=show_patent_detail, args=(patent,), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)

    elif submitted and not st.session_state.search_results:
        st.info("No se encontraron patentes relevantes con la descripción proporcionada.")

elif st.session_state.current_view == 'detail':
    patent = st.session_state.selected_patent
    if patent:
        # 1. Título de ancho completo
        st.markdown(f"<h1 class='full-patent-title'>{html.escape(patent['title'])}</h1>", unsafe_allow_html=True)

        # 2. Crear dos columnas para la imagen y el resumen
        col1, col2 = st.columns([0.5, 0.5], gap="large")

        with col1:
            # Columna de la izquierda para la imagen
            st.markdown("<h2 class='detail-subtitle'>Imagen</h2>", unsafe_allow_html=True)
            default_image = "https://placehold.co/400x400/cccccc/000000?text=No+Disponible"
            st.image(patent.get('image_url') or default_image, use_container_width=True)

        with col2:
            # Columna de la derecha para el resumen (abstract)
            st.markdown("<h2 class='detail-subtitle'>Resumen</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='full-patent-abstract'>{html.escape(patent['abstract'])}</p>", unsafe_allow_html=True)

        # 3. Metadatos y botón de volver
        st.markdown(f"<p class='full-patent-meta'>Número de Publicación: {patent['publication_number']}</p>", unsafe_allow_html=True)
        st.button("Volver a la Búsqueda", on_click=show_search_view, use_container_width=True)

    else:
        st.warning("No se ha seleccionado ninguna patente para ver los detalles.")
        show_search_view()
