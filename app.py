import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io

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
        .search-button {
            background-color: #20c997; /* Teal background */
            color: white;
            border-radius: 9999px; /* Fully rounded */
            padding: 0.75rem 1rem;
            cursor: pointer;
            border: none;
            transition: background-color 0.2s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .search-button:hover {
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
        .st-emotion-cache-1gysd4a { /* Adjust text area styling */
             border-radius: 0.75rem;
             border: 1px solid #d1d5db;
             padding: 0.75rem;
             box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .st-emotion-cache-10o5u3h { /* Adjust button styling */
            background-color: #20c997;
            color: white;
            border-radius: 0.75rem;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.2s ease;
        }
        .st-emotion-cache-10o5u3h:hover {
            background-color: #1aae89;
        }
        .st-emotion-cache-16idsys p { /* Adjust default paragraph font size */
            font-size: 1rem;
        }
        .st-emotion-cache-1s2a8v p { /* Adjust `st.markdown` `p` tag font size */
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Laptop icon SVG (used in patent cards)
LAPTOP_ICON_SVG = """
<svg class="patent-icon" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
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
    st.write("Cargando el modelo de embeddings (esto puede tardar un momento)...")
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

            # Loads the embeddings model
            model = load_embedding_model()

            st.write("Generando embeddings para las patentes (esto puede tardar un momento)...")
            # Generates embeddings for all patent descriptions
            corpus_embeddings = model.encode(df['Descripción Completa'].tolist(), convert_to_tensor=True)
            st.success(f"Embeddings generados para {len(df)} patentes.")
            return df, corpus_embeddings
        except FileNotFoundError:
            st.error(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de que está en la misma carpeta que 'app.py' en tu repositorio de GitHub.")
            return None, None
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel desde '{file_path}': {e}")
            return None, None
    return None, None

# --- Automatic local Excel file loading section (no visible header) ---

# The name of the local Excel file in the same repository
excel_file_name = "patentes.xlsx"

# Initialize model and corpus_embeddings
model = None
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


# Custom search input with magnifying glass icon
search_query = st.text_input(
    label="search_input",
    label_visibility="hidden",
    value="Necesito un sistema de cierre hermético de envases sin usar calor.",
    placeholder="Describe tu problema técnico o necesidad funcional",
    key="search_input"
)

# Fixed number of results, no slider
MAX_RESULTS = 3

# Use a form to capture the text input and button press together for better UX
with st.form(key='search_form', clear_on_submit=False):
    # This creates the visual search bar
    st.markdown(f"""
        <div class="search-input-container">
            <input type="text" id="problem_description_input" name="problem_description" 
                   value="{search_query}" placeholder="Describe tu problema técnico o necesidad funcional">
            <button type="submit" class="search-button">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-6 h-6">
                    <path fill-rule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.327 3.328a.75.75 0 11-1.06 1.06l-3.328-3.327A7 7 0 012 9z" clip-rule="evenodd" />
                </svg>
            </button>
        </div>
        """, unsafe_allow_html=True)
    
    # Hidden Streamlit text_area to capture the value from the HTML input
    # This is a workaround because Streamlit widgets don't directly integrate with custom HTML inputs for value updates easily.
    # We will use JavaScript to update this hidden text_area when the custom HTML input changes.
    # However, for simplicity with the 'submit' button, we'll assume the text_area holds the main value.
    # A more robust solution for live updating would require custom JS event listeners.
    problem_description_from_form = st.text_area(
        "Ingresa la descripción de tu problema técnico o necesidad funcional:",
        value=search_query, # Use the initial value from the custom search_query
        height=68, # Corrected minimum height as per Streamlit's requirement
        label_visibility="hidden",
        key="form_problem_description"
    )
    
    # Submit button for the form (hidden, as the custom HTML button is used)
    submitted = st.form_submit_button("Buscar Soluciones", help="Presiona Enter o haz click en la lupa para buscar", type="primary")

    # If the form is submitted via the custom button (which triggers form_submit_button)
    if submitted:
        # Use the value from the hidden text area which should have been populated by the user's input
        current_problem_description = problem_description_from_form.strip()

        if not current_problem_description:
            st.warning("Por favor, ingresa una descripción del problema.")
        else:
            with st.spinner("Buscando patentes relevantes..."):
                try:
                    # Generates the embedding of the problem description
                    query_embedding = model.encode(current_problem_description, convert_to_tensor=True)

                    # Calculates cosine similarity between the problem and all patents
                    # It's important that both tensors are on the same device (CPU/GPU)
                    cosine_scores = util.cos_sim(query_embedding, patent_embeddings)[0]

                    # Gets the indices of the most similar patents
                    top_results_indices = np.argsort(-cosine_scores.cpu().numpy())[:MAX_RESULTS] # Use MAX_RESULTS

                    if len(top_results_indices) == 0:
                        st.info("No se encontraron patentes relevantes con la descripción proporcionada.")
                    else:
                        for i, idx in enumerate(top_results_indices):
                            score = cosine_scores[idx].item()
                            patent_title = df_patents.iloc[idx]['titulo'] # Using 'titulo' in lowercase
                            patent_summary = df_patents.iloc[idx]['resumen'] # Using 'resumen' in lowercase
                            
                            # Tries to get the patent number if the column exists
                            patent_number = df_patents.iloc[idx]['numero de patente'] if 'numero de patente' in df_patents.columns else 'N/A'

                            # Randomly assign a tag for demonstration
                            tag_text = "Disponible para uso" if i % 2 == 0 else "Protección vigente"
                            tag_class = "tag-available" if i % 2 == 0 else "tag-protected"

                            st.markdown(f"""
                                <div class="patent-card">
                                    {LAPTOP_ICON_SVG}
                                    <div class="patent-details">
                                        <p class="patent-title">{patent_title}</p>
                                        <p class="patent-summary text-sm">{patent_summary[:100]}...</p>
                                        <span class="patent-tag {tag_class}">{tag_text}</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # For the full summary, use an expander (optional, as the image doesn't show it)
                            # with st.expander(f"Ver Resumen Completo de '{patent_title}'"):
                            #     st.write(patent_summary)

                except Exception as e:
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")

# This script ensures that when the custom HTML input changes, the Streamlit text_area also updates.
# This makes the form_submit_button work correctly with the custom input's value.
st.markdown("""
<script>
    const customInput = document.getElementById('problem_description_input');
    const streamlitTextArea = document.querySelector('textarea[aria-label="Ingresa la descripción de tu problema técnico o necesidad funcional:"]');

    if (customInput && streamlitTextArea) {
        customInput.addEventListener('input', (event) => {
            streamlitTextArea.value = event.target.value;
            // Dispatch input event for Streamlit to recognize the change
            streamlitTextArea.dispatchEvent(new Event('input', { bubbles: true }));
        });
        // Set initial value
        streamlitTextArea.value = customInput.value;
        streamlitTextArea.dispatchEvent(new Event('input', { bubbles: true }));
    }
</script>
""", unsafe_allow_html=True)
