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
            border-radius: 1.5rem; /* Rounded Corners */
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1); /* Soft shadow */
            padding: 2.5rem; /* Padding inside the app container */
        }
        .search-input-container {
            display: flex;
            align-items: center;
            border: 2px solid #20c997; /* Teal border */
            border-radius: 9999px; /* Fully rounded */
            padding: 0.5rem 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .search-input-container input {
            flex-grow: 1;
            border: none;
            outline: none;
            font-size: 1.125rem; /* text-lg */
            padding: 0.5rem 0.75rem;
            background: transparent;
        }
        .search-button-div { /* Changed from .search-button to .search-button-div */
            background-color: #20c997; /* Teal background */
            color: white;
            border-radius: 9999px; /* Fully rounded */
            padding: 0.75rem 1rem;
            cursor: pointer;
            transition: background-color 0.2s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex; /* Make it flex to center content */
            justify-content: center;
            align-items: center;
        }
        .search-button-div:hover { /* Changed from .search-button to .search-button-div */
            background-color: #1aae89; /* Darker teal on hover */
        }
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
        /* Removed patent-tag specific styles as the element is removed */
        .stSpinner > div {
            border-top-color: #20c997 !important;
        }
        /* --- CSS para ocultar completamente componentes nativos de Streamlit --- */
        /* Oculta los contenedores principales de st.text_area y st.form_submit_button */
        div[data-testid="stForm"] div[data-testid^="stVerticalBlock"] > div > label[data-testid="stWidgetLabel"][for^="textarea"],
        div[data-testid="stForm"] div[data-testid^="stVerticalBlock"] > div > label[data-testid="stWidgetLabel"][for^="textarea"] + div[data-testid="stTextArea"],
        div[data-testid="stForm"] div[data-testid^="stVerticalBlock"] > div > label[data-testid="stWidgetLabel"][for^="textarea"] + div[data-testid="stTextArea"] * {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
        }
        /* Oculta el st.form_submit_button y sus descendientes */
        button[data-testid="stFormSubmitButton"],
        button[data-testid="stFormSubmitButton"] * {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            overflow: hidden !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
        }

        /* Ajustes de estilos para los elementos de texto estándar de Streamlit, si aparecen */
        .st-emotion-cache-16idsys p, /* Ajusta el tamaño de fuente predeterminado para st.markdown */
        .st-emotion-cache-1s2a8v p { /* Ajusta el tamaño de fuente para `p` en `st.markdown` para versiones anteriores */
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Magnifying glass SVG (used in search button and patent cards)
MAGNIFYING_GLASS_SVG = """
<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path fill-rule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.327 3.328a.75.75 0 11-1.06 1.06l-3.328-3.327A7 7 0 012 9z" clip-rule="evenodd" />
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
    # Initial value for the custom input.
    initial_search_value = "Necesito soluciones para la gestión eficiente de la producción de miel."

    # This creates the visual search bar with an HTML input and a custom SVG-based clickable div
    st.markdown(f"""
        <div class="search-input-container">
            <input type="text" id="problem_description_input" name="problem_description"
                   value="{initial_search_value}" placeholder="Escribe aquí tu necesidad apícola...">
            <div id="custom_search_button" class="search-button-div">
                {MAGNIFYING_GLASS_SVG}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # These Streamlit widgets are present only for functionality.
    # They are completely hidden by CSS rules.
    # The label has a data-testid assigned by Streamlit and for^="textarea"
    problem_description_from_form = st.text_area(
        "Hidden input for problem description", # Label, though hidden
        value=initial_search_value, # Initial value
        height=68, # Required minimum height
        label_visibility="hidden", # Hide the label
        key="form_problem_description", # Key for internal Streamlit tracking
        placeholder="Este campo está oculto.", # Placeholder, won't be seen
        help="Este campo es solo para la lógica interna y está oculto visualmente." # Help text, also hidden
    )
    
    submitted = st.form_submit_button("Buscar Soluciones", type="primary")

    # If the form is submitted via the custom HTML button (which triggers st.form_submit_button)
    if submitted:
        current_problem_description = problem_description_from_form.strip() # Get value from hidden text_area

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

                            # Escape HTML-breaking characters in the content
                            escaped_patent_title = html.escape(patent_title)
                            escaped_patent_summary_short = html.escape(patent_summary[:100]) + "..."

                            card_html = f"""
<div class="patent-card">
    <div class="patent-icon">{MAGNIFYING_GLASS_SVG}</div>
    <div class="patent-details">
        <p class="patent-title">{escaped_patent_title}</p>
        <p class="patent-summary text-sm">{escaped_patent_summary_short}</p>
    </div>
</div>
"""
                            st.markdown(card_html, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")

# This JavaScript ensures that when the custom HTML input changes, the Streamlit text_area also updates.
# This makes the form_submit_button work correctly with the custom input's value.
st.markdown("""
<script>
    const customInput = document.getElementById('problem_description_input');
    const streamlitTextArea = document.querySelector('textarea[aria-label="Hidden input for problem description"]');
    const submitButton = document.querySelector('button[data-testid="stFormSubmitButton"]'); // Native submit button
    const customSearchButton = document.getElementById('custom_search_button'); // The custom div button

    if (customInput && streamlitTextArea && submitButton && customSearchButton) {
        // Set initial value for Streamlit text_area from custom input
        streamlitTextArea.value = customInput.value;
        streamlitTextArea.dispatchEvent(new Event('input', { bubbles: true }));

        customInput.addEventListener('input', (event) => {
            streamlitTextArea.value = event.target.value;
            streamlitTextArea.dispatchEvent(new Event('input', { bubbles: true }));
        });

        // Trigger form submission if Enter key is pressed in the custom input
        customInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault(); // Prevent default form submission
                submitButton.click(); // Programmatically click the hidden Streamlit submit button
            }
        });

        // Ensure the custom div button also triggers the hidden Streamlit submit button
        customSearchButton.addEventListener('click', (event) => {
            event.preventDefault(); // Prevent default behavior of custom div
            submitButton.click(); // Programmatically click the hidden Streamlit submit button
        });
    }
</script>
""", unsafe_allow_html=True)
