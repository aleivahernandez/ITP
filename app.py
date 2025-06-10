import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io
import html

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Explorador de Soluciones Técnicas (Patentes)")

# Custom CSS for a better visual match to the provided image
# Using Tailwind CSS classes for styling
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
            border-radius: 1.5rem; /* Rounded corners */
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1); /* Soft shadow */
            padding: 2.5rem; /* Padding inside the app container */
        }
        /* Removed custom search input container and button styles as they are no longer used */
        .patent-card {
            display: flex;
            align-items: flex-start;
            background-color: #f0fdf4; /* Light green background */
            border-left: 5px solid #20c997; /* Teal left border */
            border-radius: 0.75rem; /* Rounded corners */
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .patent-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .patent-icon {
            flex-shrink: 0;
            width: 48px; /* w-12 */
            height: 48px; /* h-12 */
            margin-right: 1.5rem; /* mr-6 */
            color: #20c997; /* Teal color */
        }
        .patent-details {
            flex-grow: 1;
        }
        .patent-title {
            font-size: 1.25rem; /* text-xl */
            font-weight: 600; /* font-semibold */
            color: #1f2937; /* Gray-900 */
            margin-bottom: 0.5rem;
        }
        .patent-summary {
            font-size: 0.95rem; /* text-base */
            color: #4b5563; /* Gray-700 */
        }
        .patent-tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 0.5rem;
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 0.75rem;
            color: #ffffff;
        }
        .tag-available {
            background-color: #10b981; /* Green-500 */
        }
        .tag-protected {
            background-color: #f59e0b; /* Amber-500 */
        }
        .stSpinner > div {
            border-top-color: #20c997 !important;
        }
        /* --- CSS para estilizar componentes nativos de Streamlit --- */
        /* Estiliza el st.text_area para parecerse al input de la imagen */
        textarea[aria-label="Describe tu problema técnico o necesidad funcional:"] {
            border-radius: 9999px !important; /* Fully rounded */
            border: 2px solid #20c997 !important; /* Teal border */
            padding: 0.5rem 1.5rem !important; /* Adjust padding */
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
            font-size: 1.125rem !important; /* text-lg */
            margin-bottom: 1rem; /* Space below the input */
            resize: none !important; /* Prevent manual resizing */
        }
        /* Estiliza el botón de envío del formulario */
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
        /* Ajustes de estilos para los elementos de texto estándar de Streamlit, si aparecen */
        .st-emotion-cache-16idsys p, /* Ajusta el tamaño de fuente predeterminado para st.markdown */
        .st-emotion-cache-1s2a8v p { /* Ajusta el tamaño de fuente para `p` en `st.markdown` para versiones anteriores */
            font-size: 1rem;
        }
        /* Ocultar el label del text_area si no queremos que aparezca */
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > label[data-testid="stWidgetLabel"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Laptop icon SVG (used in patent cards) - still using the original SVG for the cards
LAPTOP_ICON_SVG = """
<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path fill-rule="evenodd" d="M3 5a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V5zm1.454 11.085a.75.75 0 00-.735.434l-.626 1.706a.75.75 0 00.923 1.054l.583-.178h13.375l.583.178a.75.75 0 00.923-1.054l-.626-1.706a.75.75 0 00-.735-.434H4.454z" clip-rule="evenodd" />
</svg>
"""

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
    Processes the Excel patent file from a local path.
    Reads the file, combines title and summary, and generates the embeddings.
    `st.cache_data` is used to cache processed data
    and generated embeddings, avoiding unnecessary reprocessing.
    """
    if file_path:
        try:
            # Reads the Excel file from the local path
            df = pd.read_excel(file_path)

            # Validate that the necessary columns exist (now in lowercase)
            required_columns = ['titulo', 'resumen']
            if not all(col in df.columns for col in required_columns):
                st.error(f"El archivo Excel debe contener las columnas: {', '.join(required_columns)}")
                return None, None

            # Fill null values in 'titulo' and 'resumen' with empty strings to avoid errors
            df['titulo'] = df['titulo'].fillna('')
            df['resumen'] = df['resumen'].fillna('')

            # Combines title and summary to create a complete patent description
            df['Descripción Completa'] = df['titulo'] + ". " + df['resumen']

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
            st.error(f"Error al procesar el archivo Excel desde '{file_path}': {e}")
            return None, None
    return None, None

# --- Automatic local Excel file loading section ---

# The name of the local Excel file in the same repository
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
             "y que contenga las columnas 'titulo' y 'resumen'.")
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
        value="Necesito un sistema de cierre hermético de envases sin usar calor.",
        height=68, # Required minimum height
        label_visibility="visible", # Ensure label is visible
        key="problem_description_input_area", # Renamed key for clarity
        placeholder="Escribe aquí tu problema técnico..." # Added placeholder
    )
    
    # This is the Streamlit form submit button, now visible and primary for submission
    submitted = st.form_submit_button("Buscar Soluciones", type="primary")

    # If the form is submitted
    if submitted:
        current_problem_description = problem_description.strip() # Direct access to text_area value

        if not current_problem_description:
            st.warning("Por favor, ingresa una descripción del problema.")
        else:
            with st.spinner("Buscando patentes relevantes..."):
                try:
                    current_model = load_embedding_model()
                    query_embedding = current_model.encode(current_problem_description, convert_to_tensor=True)

                    cosine_scores = util.cos_sim(query_embedding, patent_embeddings)[0]
                    top_results_indices = np.argsort(-cosine_scores.cpu().numpy())[:MAX_RESULTS]

                    if len(top_results_indices) == 0:
                        st.info("No se encontraron patentes relevantes con la descripción proporcionada.")
                    else:
                        for i, idx in enumerate(top_results_indices):
                            score = cosine_scores[idx].item()
                            patent_title = df_patents.iloc[idx]['titulo']
                            patent_summary = df_patents.iloc[idx]['resumen']
                            
                            patent_number = df_patents.iloc[idx]['numero de patente'] if 'numero de patente' in df_patents.columns else 'N/A'

                            tag_text = "Disponible para uso" if i % 2 == 0 else "Protección vigente"
                            tag_class = "tag-available" if i % 2 == 0 else "tag-protected"

                            # Escape HTML-breaking characters in the content
                            escaped_patent_title = html.escape(patent_title)
                            escaped_patent_summary_short = html.escape(patent_summary[:100]) + "..."

                            card_html = f"""
<div class="patent-card">
    <div class="patent-icon">{LAPTOP_ICON_SVG}</div>
    <div class="patent-details">
        <p class="patent-title">{escaped_patent_title}</p>
        <p class="patent-summary text-sm">{escaped_patent_summary_short}</p>
        <span class="patent-tag {tag_class}">{tag_text}</span>
    </div>
</div>
"""
                            st.markdown(card_html, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")

# Removed the JavaScript section as it's no longer needed for syncing custom HTML input

