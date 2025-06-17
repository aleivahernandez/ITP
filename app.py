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

        /* --- Google Patents style for patent results --- */
        .google-patent-card {
            background-color: #ffffff; /* White background */
            border: 1px solid #dadce0; /* Light gray border */
            border-radius: 8px; /* Slightly rounded corners */
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            transition: box-shadow 0.2s ease;
            position: relative; /* For similarity score */
        }
        .google-patent-card:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* More prominent shadow on hover */
        }
        .google-patent-title {
            font-size: 1.15rem;
            font-weight: 600; /* Semi-bold */
            color: #1a0dab; /* Google blue link color */
            margin-bottom: 0.4rem;
            line-height: 1.3;
        }
        .google-patent-summary {
            font-size: 0.9rem;
            color: #4d5156; /* Darker gray for text */
            margin-bottom: 0.5rem;
            line-height: 1.5;
        }
        .google-patent-meta {
            font-size: 0.8rem;
            color: #70757a; /* Lighter gray for metadata */
            margin-top: 0.5rem;
        }
        /* Adjusting similarity score position for Google Patents style */
        .similarity-score {
            position: absolute;
            top: 0.75rem;
            right: 0.75rem;
            background-color: #e0f2f7; /* Light blue background for contrast */
            color: #20c997; /* Teal color */
            padding: 0.15rem 0.4rem;
            border-radius: 0.4rem;
            font-size: 0.75rem;
            font-weight: 600;
            z-index: 10;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# No Magnifying Glass SVG needed outside CSS
# MAGNIFYING_GLASS_SVG = """..."""

# --- Functions for loading and processing data/models ---

@st.cache_resource
def load_embedding_model():
    """
    Loads the pre-trained SentenceTransformer model.
    `st.cache_resource` is used to load the model only once and reuse it,
    improving application performance.
    """
    with st.spinner("Cargando el modelo de embeddings (esto puede tardar un momento)..."):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        st.success("Modelo de embeddings cargado correctamente.")
    return model

@st.cache_data
def process_patent_data(file_path):
    """
    Processes the patent file from a local path (CSV or Excel).
    Reads the file, combines title and summary, and generates the embeddings.
    `st.cache_data` is used to cache processed data
    and generated embeddings, avoiding unnecessary reprocessing.
    """
    if file_path:
        try:
            # Determine file type and read accordingly
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                st.error("Formato de archivo no soportado. Por favor, sube un archivo .csv o .xlsx.")
                return None, None

            # Normalize column names: strip spaces and convert to lowercase
            df.columns = df.columns.str.strip().str.lower()

            # Define required columns after normalization
            required_columns_normalized = ['title (original language)', 'abstract (original language)']
            
            # Check if all required columns exist after normalization
            if not all(col in df.columns for col in required_columns_normalized):
                st.error(f"El archivo debe contener las columnas: '{required_columns_normalized[0]}' y '{required_columns_normalized[1]}'. "
                         "Por favor, revisa que los nombres de las columnas sean exactos (ignorando mayúsculas/minúsculas y espacios extra).")
                return None, None

            # Access columns using their normalized names
            original_title_col = 'title (original language)'
            original_abstract_col = 'abstract (original language)'

            # Fill null values with empty strings
            df[original_title_col] = df[original_title_col].fillna('')
            df[original_abstract_col] = df[original_abstract_col].fillna('')

            # Combines the original title and summary to create a complete patent description
            df['Descripción Completa'] = df[original_title_col] + ". " + df[original_abstract_col]

            # Load the embedding model INSIDE this cached function
            # Since load_embedding_model is also cached, it will only run once
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
             "y que contenga las columnas 'Title (Original language)' y 'Abstract (Original language)'. "
             "Se ignora mayúsculas/minúsculas y espacios extra en los nombres de las columnas.")
    st.stop() # Stop the app if data can't be loaded

# --- Section for problem input and search ---
st.markdown("<h2 class='text-2xl font-bold mb-4'>Explorar soluciones técnicas</h2>", unsafe_allow_html=True)
st.markdown("<p class='text-gray-600 mb-6'>Describe tu problema técnico o necesidad funcional</p>", unsafe_allow_html=True)


# Fixed number of results, no slider
MAX_RESULTS = 3

# Use a form to capture the text input and button press together for better UX
with st.form(key='search_form', clear_on_submit=False):
    # This is the Streamlit text_area, now visible and primary for input
    problem_description = st.text_area(
        "Describe tu problema técnico o necesidad funcional:",
        value="Necesito soluciones para la gestión eficiente de la producción de miel.",
        height=68, # Required minimum height
        label_visibility="visible", # We will hide the label with CSS later
        key="problem_description_input_area",
        placeholder="Escribe aquí tu necesidad apícola..."
    )
    
    # This is the Streamlit form submit button.
    submitted = st.form_submit_button("Buscar Soluciones", type="primary")

    # If the form is submitted
    if submitted:
        current_problem_description = problem_description.strip() # Direct access to text_area value

        if not current_problem_description:
            st.warning("Por favor, ingresa una descripción del problema.")
        else:
            with st.spinner("Buscando patentes relevantes..."):
                try: # Start of the try block
                    current_model = load_embedding_model()
                    query_embedding = current_model.encode(current_problem_description, convert_to_tensor=True)

                    cosine_scores = util.cos_sim(query_embedding, patent_embeddings)[0]
                    top_results_indices = np.argsort(-cosine_scores.cpu().numpy())[:MAX_RESULTS]

                    if len(top_results_indices) == 0:
                        st.info("No se encontraron patentes relevantes con la descripción proporcionada.")
                    else:
                        for i, idx in enumerate(top_results_indices):
                            score = cosine_scores[idx].item()
                            # Use the original (untranslated) title and abstract for display
                            patent_title = df_patents.iloc[idx]['title (original language)']
                            patent_summary = df_patents.iloc[idx]['abstract (original language)']
                            
                            patent_number_found = 'N/A'
                            if 'numero de patente' in df_patents.columns:
                                patent_number_found = df_patents.iloc[idx]['numero de patente']
                            elif 'publication number' in df_patents.columns: 
                                patent_number_found = df_patents.iloc[idx]['publication number']
                            elif 'patent number' in df_patents.columns: 
                                patent_number_found = df_patents.iloc[idx]['patent number']
                            
                            # Escape HTML-breaking characters in the content
                            escaped_patent_title = html.escape(patent_title)
                            escaped_patent_summary_short = html.escape(patent_summary[:100]) + "..."

                            # Display using new Google Patents-like HTML structure
                            card_html = f"""
<div class="google-patent-card">
    <div class="similarity-score">Similitud: {score:.2%}</div>
    <p class="google-patent-title">{escaped_patent_title}</p>
    <p class="google-patent-summary">{escaped_patent_summary_short}</p>
    <p class="google-patent-meta">Patente: {patent_number_found}</p>
</div>
"""
                            st.markdown(card_html, unsafe_allow_html=True)
                            
                except Exception as e: # End of the try block, start of the except block
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")

# No custom JavaScript for syncing is needed now.
