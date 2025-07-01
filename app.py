import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io
import html

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Explorador de Soluciones Técnicas (Patentes)")

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
        /* El contenedor principal de la aplicación Streamlit.
           Estos estilos se aplicarán consistentemente a toda la app,
           incluyendo las vistas de búsqueda y detalle. */
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
        /* Adjust default paragraph font size for st.markdown */
        .st-emotion-cache-16idsys p, 
        .st-emotion-cache-1s2a8v p { 
            font-size: 1rem;
        }
        /* Hide the default label for st.text_area */
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > label[data-testid="stWidgetLabel"] {
            display: none !important;
        }

        /* --- Google Patents style for patent results (Search View) --- */
        .google-patent-result-container { /* New container for each result block (individual search results) */
            background-color: #ffffff; /* White background */
            border: 1px solid #dadce0; /* Light gray border */
            border-radius: 8px; /* Slightly rounded corners */
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            position: relative; /* For similarity score */
        }
        .result-header {
            display: flex;
            align-items: flex-start; /* Align image and text to the top */
            margin-bottom: 0.5rem;
            gap: 1rem; /* Space between image and text */
        }
        .result-image-wrapper { /* Wrapper for the image to control its size and flex behavior */
            flex-shrink: 0;
            width: 80px; /* Fixed width for the image container */
            height: 80px; /* Fixed height for the image container */
            border-radius: 4px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f0f0f0; /* Placeholder background */
        }
        .result-image {
            width: 100%;
            height: 100%;
            object-fit: contain; /* Ensure image fits without cropping, maintaining aspect ratio */
            border-radius: 4px;
        }
        .result-text-content { /* Wrapper for title, summary, meta */
            flex-grow: 1; /* Allows text content to take remaining space */
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
        .similarity-score-display { /* For displaying score without absolute positioning */
            font-size: 0.8rem;
            font-weight: 600;
            color: #20c997; /* Teal color */
            margin-left: auto; /* Push to the right */
            background-color: #e0f2f7; /* Light blue background to contrast */
            padding: 0.15rem 0.4rem;
            border-radius: 0.4rem;
        }

        /* --- Styles for the full patent view (Detail View) --- */
        .detail-content-wrapper {
            padding-top: 0.5rem; 
            padding-bottom: 0.5rem;
        }

        .full-patent-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1a0dab;
            margin-bottom: 1rem;
        }
        .full-patent-abstract {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #333333;
            margin-bottom: 1.5rem;
        }
        .full-patent-meta {
            font-size: 0.9rem;
            color: #70757a;
            margin-top: 1rem;
        }
        .back-button {
            background-color: #6c757d !important; /* Grey color for back button */
            color: white !important;
            border-radius: 0.75rem !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            transition: background-color 0.2s ease !important;
            display: block !important;
            margin: 2rem auto 0 auto !important;
            width: fit-content;
        }
        .back-button:hover {
            background-color: #5a6268 !important;
        }

        /* --- INICIO: NUEVOS ESTILOS PARA VISTA DETALLADA (2 COLUMNAS) --- */
        .detail-body-container {
            display: flex;
            flex-wrap: wrap; /* Permite que las columnas se reorganicen en pantallas pequeñas */
            gap: 1.5rem;     /* Espacio entre la imagen y el texto */
            align-items: flex-start; /* Alinea los items en la parte superior */
            margin-bottom: 1.5rem; /* Espacio debajo de la sección de dos columnas */
        }

        .detail-image-column {
            flex: 1 1 30%;      /* Define la flexibilidad de la columna */
            max-width: 30%;     /* Límite máximo del 30% del ancho */
        }

        .detail-image-column img {
            width: 100%;        /* La imagen ocupa todo el ancho de su columna */
            height: auto;
            border-radius: 8px; /* Opcional: suaviza las esquinas de la imagen */
        }

        .detail-abstract-column {
            flex: 1 1 65%;      /* Ocupa el espacio restante */
        }

        /* Media Query para mejorar la responsividad en pantallas pequeñas */
        @media (max-width: 640px) {
            .detail-body-container {
                flex-direction: column; /* Apila la imagen y el texto verticalmente */
            }
            .detail-image-column, .detail-abstract-column {
                max-width: 100%; /* Ambas columnas ocupan el ancho completo */
            }
        }
        /* --- FIN: NUEVOS ESTILOS PARA VISTA DETALLADA --- */
    </style>
    """,
    unsafe_allow_html=True
)

# --- Functions for loading and processing data/models ---

@st.cache_resource
def load_embedding_model():
    """
    Carga el modelo pre-entrenado SentenceTransformer.
    `st.cache_resource` se usa para cargar el modelo una sola vez y reutilizarlo,
    mejorando el rendimiento de la aplicación.
    """
    with st.spinner("Cargando el modelo de embeddings (esto puede tardar un momento)..."):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model

@st.cache_data
def process_patent_data(file_path):
    """
    Procesa el archivo de patentes desde una ruta local (CSV o Excel).
    Lee el archivo, combina título y resumen, obtiene las URLs de las imágenes (desde GitHub)
    y genera los embeddings.
    `st.cache_data` se usa para almacenar en caché los datos procesados
    y los embeddings generados, evitando reprocesamientos innecesarios.
    """
    if file_path:
        try:
            # Determina el tipo de archivo y lo lee en consecuencia
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                st.error("Formato de archivo no soportado. Por favor, sube un archivo .csv o .xlsx.")
                return None, None

            # Normaliza los nombres de las columnas: elimina espacios y convierte a minúsculas
            df.columns = df.columns.str.strip().str.lower()

            # Define las columnas requeridas después de la normalización
            required_columns_normalized = [
                'title (original language)',
                'abstract (original language)',
                'publication number',
            ]
            
            # Verifica si todas las columnas requeridas existen después de la normalización
            for col in required_columns_normalized:
                if col not in df.columns:
                    st.error(f"El archivo Excel debe contener la columna requerida: '{col}'. "
                             "Por favor, revisa que los nombres de las columnas sean exactos (ignorando mayúsculas/minúsculas y espacios extra).")
                    return None, None

            # Accede a las columnas usando sus nombres normalizados
            original_title_col = 'title (original language)'
            original_abstract_col = 'abstract (original language)'
            publication_number_col = 'publication number'

            # Rellena los valores nulos con cadenas vacías
            df[original_title_col] = df[original_title_col].fillna('')
            df[original_abstract_col] = df[original_abstract_col].fillna('')
            df[publication_number_col] = df[publication_number_col].fillna('')

            # --- Configure GitHub Image Base URL ---
            github_image_base_url = "https://raw.githubusercontent.com/aleivahernandez/ITP_v2/main/images/"
            # --- End GitHub Image Base URL Configuration ---

            # Construct image URLs using Publication Number
            df['image_url_processed'] = df[publication_number_col].apply(
                lambda x: f"{github_image_base_url}{x}.png" if x else ""
            )

            # Combines the original title and summary to create a complete patent description
            df['Descripción Completa'] = df[original_title_col] + ". " + df[original_abstract_col]

            # Load the embedding model INSIDE this cached function
            model_instance = load_embedding_model()

            # Generates embeddings for all patent descriptions using the loaded model instance
            corpus_embeddings = model_instance.encode(df['Descripción Completa'].tolist(), convert_to_tensor=True)
            return df, corpus_embeddings
        except FileNotFoundError:
            st.error(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de que está en la misma carpeta que 'app.py' en tu repositorio de GitHub.")
            return None, None
        except Exception as e:
            st.error(f"Error al procesar el archivo '{file_path}': {e}")
            return None, None
    return None, None

# --- Automatic local Excel file loading section ---

# The name of the local patent file in the same repository
excel_file_name = "patentes.xlsx" 

# Initialize df_patents and patent_embeddings
df_patents = None
patent_embeddings = None

# Processes the data automatically upon application startup
with st.spinner(f"Inicializando base de datos de patentes..."):
    df_patents, patent_embeddings = process_patent_data(excel_file_name)

if df_patents is None or patent_embeddings is None:
    st.error(f"No se pudo cargar o procesar la base de datos de patentes desde '{excel_file_name}'. "
             "Por favor, verifica que el archivo exista en el mismo directorio de 'app.py' en tu repositorio de GitHub "
             "y que contenga las columnas requeridas (ver mensaje de error anterior).")
    st.stop() # Stop the app if data can't be loaded

# --- Session State Initialization ---
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'search' # 'search' or 'detail'
if 'selected_patent' not in st.session_state:
    st.session_state.selected_patent = None
if 'search_results' not in st.session_state: # To store results after a search
    st.session_state.search_results = []
if 'query_description' not in st.session_state: # To persist search query
    st.session_state.query_description = "Certificación calidad de miel."


# --- Functions for view management ---
def show_search_view():
    st.session_state.current_view = 'search'
    st.session_state.selected_patent = None
    # No borramos los resultados para que el usuario pueda volver a ellos
    # st.session_state.search_results = [] 

def show_patent_detail(patent_data):
    st.session_state.current_view = 'detail'
    st.session_state.selected_patent = patent_data

# --- Main Application Logic ---

if st.session_state.current_view == 'search':
    st.markdown("<h2 class='text-2xl font-bold mb-4'>Explorar soluciones técnicas</h2>", unsafe_allow_html=True)

    # Fixed number of results, no slider
    MAX_RESULTS = 3

    # Use a form to capture the text input and button press together for better UX
    with st.form(key='search_form', clear_on_submit=False):
        # This is the Streamlit text_area, now visible and primary for input
        problem_description = st.text_area(
            "Describe tu problema técnico o necesidad funcional:",
            value=st.session_state.query_description, # Use persisted query
            height=68, # Required minimum height
            label_visibility="visible", # Keep label visible
            key="problem_description_input_area",
            placeholder="Escribe aquí tu necesidad apícola..."
        )
        
        # This is the Streamlit form submit button.
        submitted = st.form_submit_button("Buscar Soluciones", type="primary")

        # If the form is submitted, perform the search and store results in session_state
        if submitted:
            current_problem_description = problem_description.strip() # Direct access to text_area value
            st.session_state.query_description = current_problem_description # Persist the query

            if not current_problem_description:
                st.warning("Por favor, ingresa una descripción del problema.")
                st.session_state.search_results = [] # Clear results if query is empty
            else:
                with st.spinner("Buscando patentes relevantes..."):
                    try: 
                        current_model = load_embedding_model()
                        query_embedding = current_model.encode(current_problem_description, convert_to_tensor=True)

                        cosine_scores = util.cos_sim(query_embedding, patent_embeddings)[0]
                        top_results_indices = np.argsort(-cosine_scores.cpu().numpy())[:MAX_RESULTS]

                        # Store the search results in session_state
                        results_to_display = []
                        for i, idx in enumerate(top_results_indices):
                            score = cosine_scores[idx].item()
                            patent_title = df_patents.iloc[idx]['title (original language)']
                            patent_summary = df_patents.iloc[idx]['abstract (original language)']
                            patent_image_url = df_patents.iloc[idx]['image_url_processed'] 
                            patent_number_found = df_patents.iloc[idx]['publication number']
                            
                            results_to_display.append({
                                'title': patent_title,
                                'abstract': patent_summary,
                                'publication_number': patent_number_found,
                                'image_url': patent_image_url,
                                'score': score
                            })
                        st.session_state.search_results = results_to_display
                        
                    except Exception as e: 
                        st.error(f"Ocurrió un error durante la búsqueda: {e}")
                        st.session_state.search_results = [] # Clear results on error

    # Display search results OUTSIDE the form
    if st.session_state.search_results:
        st.subheader("Resultados de la búsqueda:") 
        for i, patent_data in enumerate(st.session_state.search_results):
            escaped_patent_title = html.escape(patent_data['title'])
            escaped_patent_summary_short = html.escape(patent_data['abstract'][:100]) + "..."
            default_image_url = "https://placehold.co/120x120/cccccc/000000?text=No+Image" 
            
            with st.container(border=False):
                st.markdown(f"""
<div class="google-patent-result-container">
    <div class="result-header">
        <div class="result-image-wrapper">
            <img src="{patent_data['image_url'] if patent_data['image_url'] else default_image_url}" 
                 alt="" class="result-image" 
                 onerror="this.onerror=null;this.src='{default_image_url}';">
        </div>
        <div class="result-text-content">
            <h3 class="result-title">{escaped_patent_title}</h3>
            <p class="result-summary">{escaped_patent_summary_short}</p>
            <p class="result-meta">Patente: {patent_data['publication_number']} <span class="similarity-score-display">Similitud: {patent_data['score']:.2%}</span></p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
                # Add a button to view full details
                st.button(
                    "Ver Detalles Completos", 
                    key=f"view_patent_{i}", # Unique key for each button
                    on_click=show_patent_detail, 
                    args=(patent_data,), # Pass the full patent_data to the callback
                    use_container_width=True
                )
                st.markdown("---") # Separator between results
    elif st.session_state.query_description and submitted and not st.session_state.search_results:
        # Only show this message if a search was actually submitted and yielded no results
        st.info("No se encontraron patentes relevantes con la descripción proporcionada.")


elif st.session_state.current_view == 'detail':
    selected_patent = st.session_state.selected_patent
    if selected_patent:
        # Envoltorio para todo el contenido del detalle
        st.markdown("<div class='detail-content-wrapper'>", unsafe_allow_html=True)
        
        # 1. Título de ancho completo (sin cambios)
        st.markdown(f"<h1 class='full-patent-title'>{html.escape(selected_patent['title'])}</h1>", unsafe_allow_html=True)
        
        # --- INICIO: NUEVA ESTRUCTURA DE 2 COLUMNAS ---
        
        # Prepara el HTML de la columna de la imagen (solo si existe una URL)
        image_html_content = ""
        if selected_patent.get('image_url'):
            image_html_content = f"""
            <div class='detail-image-column'>
                <img src="{selected_patent['image_url']}" alt="Imagen de la patente">
            </div>
            """
        
        # Prepara el HTML de la columna del resumen/descripción
        abstract_html_content = f"""
        <div class='detail-abstract-column'>
            <p class='full-patent-abstract'>{html.escape(selected_patent['abstract'])}</p>
        </div>
        """

        # Combina la imagen y el resumen en el contenedor flex
        st.markdown(f"""
        <div class='detail-body-container'>
            {image_html_content}
            {abstract_html_content}
        </div>
        """, unsafe_allow_html=True)
        # --- FIN: NUEVA ESTRUCTURA DE 2 COLUMNAS ---

        # 3. Metadatos y botón de volver (sin cambios en su lógica)
        st.markdown(f"<p class='full-patent-meta'>Número de Publicación: {selected_patent['publication_number']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) # Cierra detail-content-wrapper
        
        st.button(
            "Volver a la Búsqueda", 
            on_click=show_search_view, 
            key="back_to_search_btn", 
            help="Regresar a la página de resultados de búsqueda.", 
            type="secondary", 
            use_container_width=True
        )
    else:
        st.warning("No se ha seleccionado ninguna patente para ver los detalles.")
        show_search_view() # Redirige a la búsqueda si no hay patente seleccionada
