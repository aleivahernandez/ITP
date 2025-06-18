import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io
import html

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Explorador de Soluciones Técnicas (Patentes)")

# Initialize session state for selected patent for detail view
if 'selected_patent_idx' not in st.session_state:
    st.session_state.selected_patent_idx = None

# Custom CSS for a better visual match to Google Patents style and detail view
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
            padding: 1.25rem; /* Re-added padding directly to the card */
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            transition: box-shadow 0.2s ease;
            position: relative; /* For similarity score and click overlay */
            display: flex; /* Use flexbox for image and content layout */
            align-items: flex-start; /* Align items to the top */
        }
        .google-patent-card:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); /* More prominent shadow on hover */
        }
        .patent-image-container {
            flex-shrink: 0; /* Prevent image container from shrinking */
            width: 120px; /* Fixed width for the image container */
            height: 120px; /* Fixed height for the image container */
            margin-right: 1rem; /* Space between image and text */
            border-radius: 4px; /* Slightly rounded corners for the image box */
            overflow: hidden; /* Hide overflowing parts of the image */
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f0f0f0; /* Placeholder background */
        }
        .patent-thumbnail {
            width: 100%;
            height: 100%;
            object-fit: contain; /* Ensure image fits without cropping, maintaining aspect ratio */
            border-radius: 4px;
        }
        .google-patent-content-details { /* New class to wrap text content */
            flex-grow: 1; /* Allow content to take remaining space */
            padding: 1.25rem; /* Re-add padding inside the text content area */
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
            background-color: #e0f2f7; /* Light blue background to contrast */
            color: #20c997; /* Teal color */
            padding: 0.15rem 0.4rem;
            border-radius: 0.4rem;
            font-size: 0.75rem;
            font-weight: 600;
            z-index: 10;
        }
        /* --- Detail View Styles (simplified) --- */
        .detail-view-container {
            background-color: #ffffff;
            border-radius: 1.5rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            padding: 2.5rem;
            margin-top: 2rem;
        }
        .detail-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 1.5rem;
        }
        .detail-content {
            font-size: 1rem;
            color: #3c4043;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        .detail-content h4 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .back-button {
            background-color: #607d8b !important; /* Grey-blue color */
            color: white !important;
            border-radius: 0.5rem !important;
            padding: 0.75rem 1.25rem !important;
            font-weight: 500 !important;
            margin-top: 1.5rem !important;
        }
        .back-button:hover {
            background-color: #455a64 !important; /* Darker grey-blue on hover */
        }
        /* Style to make the hidden st.form_submit_button cover the entire custom card */
        /* This targets the div that Streamlit creates for the button widget */
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > div > button[data-testid^="stFormSubmitButton"] {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 5; /* Ensure it's clickable above card content, but below similarity score if present */
            background-color: transparent !important;
            color: transparent !important;
            border: none !important;
            cursor: pointer !important;
            padding: 0 !important;
            margin: 0 !important;
            /* Hide the text label of the button */
            font-size: 0 !important; /* Hide text */
            line-height: 0 !important; /* Collapse line height */
            overflow: hidden !important; /* Hide overflow */
        }
        /* Ensure the actual button element inside is also hidden */
        div[data-testid="stForm"] div[data-testid^="stBlock"] > div > div > button[data-testid^="stFormSubmitButton"] > * {
            display: none !important;
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
    Processes the patent file from a local path (CSV or Excel).
    Reads the file, combines title and summary, gets image URLs (from GitHub), and generates the embeddings.
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
            required_columns_normalized = [
                'title (original language)',
                'abstract (original language)',
                'publication number',
                'assignee - dwpi',        # New required column
                'publication country',    # New required column
            ]
            
            # Check if all required columns exist after normalization
            for col in required_columns_normalized:
                if col not in df.columns:
                    st.error(f"El archivo Excel debe contener la columna requerida: '{col}'. "
                             "Por favor, revisa que los nombres de las columnas sean exactos (ignorando mayúsculas/minúsculas y espacios extra).")
                    return None, None

            # Access columns using their normalized names
            original_title_col = 'title (original language)'
            original_abstract_col = 'abstract (original language)'
            publication_number_col = 'publication number'
            assignee_dwpi_col = 'assignee - dwpi'
            publication_country_col = 'publication country'

            # Fill null values with empty strings
            df[original_title_col] = df[original_title_col].fillna('')
            df[original_abstract_col] = df[original_abstract_col].fillna('')
            df[publication_number_col] = df[publication_number_col].fillna('')
            df[assignee_dwpi_col] = df[assignee_dwpi_col].fillna('')
            df[publication_country_col] = df[publication_country_col].fillna('')

            # --- Configure GitHub Image Base URL ---
            github_image_base_url = "https://raw.githubusercontent.com/aleivahernandez/ITP/main/images/" 
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

# --- Detail View Logic ---
if st.session_state.selected_patent_idx is not None:
    selected_idx = st.session_state.selected_patent_idx
    patent = df_patents.iloc[selected_idx]

    # Extract patent details for display
    title = patent['title (original language)']
    abstract = patent['abstract (original language)']
    publication_number = patent['publication number']
    assignee = patent['assignee - dwpi']
    publication_country = patent['publication country']
    image_url = patent['image_url_processed'] # Get processed image URL
    
    # Default image for onerror, if needed
    default_image_url = "https://placehold.co/250x250/cccccc/000000?text=No+Image"

    st.markdown("<div class='detail-view-container'>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='detail-header'>{html.escape(title)}</h1>", unsafe_allow_html=True)
    
    # Simplified detail view content (Title and Abstract only for now, as requested)
    st.markdown("<h4>Resumen:</h4>", unsafe_allow_html=True)
    st.markdown(f"<p>{html.escape(abstract)}</p>", unsafe_allow_html=True)

    # Back Button
    if st.button("Volver a la búsqueda", key="back_to_search", help="Regresar a la lista de resultados"):
        st.session_state.selected_patent_idx = None
        st.rerun() # Rerun to refresh the view
        
    st.markdown("</div>", unsafe_allow_html=True) # Close detail-view-container

# --- Main Search View Logic ---
else:
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
            label_visibility="visible", # Keep label visible
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
                            st.subheader("Resultados de la búsqueda:") # Added a subheader for clarity
                            for i, idx in enumerate(top_results_indices):
                                score = cosine_scores[idx].item()
                                # Use the original (untranslated) title and abstract for display
                                patent_title = df_patents.iloc[idx]['title (original language)']
                                patent_summary = df_patents.iloc[idx]['abstract (original language)']
                                patent_image_url = df_patents.iloc[idx]['image_url_processed'] # Get processed image URL
                                
                                # Use the processed publication number directly for consistency
                                patent_number_found = df_patents.iloc[idx]['publication number']
                                
                                # Escape HTML-breaking characters in the content
                                escaped_patent_title = html.escape(patent_title)
                                escaped_patent_summary_short = html.escape(patent_summary[:100]) + "..."
                                
                                # Default image for onerror, if needed
                                default_image_url = "https://placehold.co/120x120/cccccc/000000?text=No+Image" 
                                
                                # Wrap each card in a form for clickability
                                # The hidden submit button will now cover the card visually for click detection
                                with st.form(key=f"patent_card_form_{idx}", clear_on_submit=False):
                                    card_html = """
    <div class="google-patent-card">
        <div class="similarity-score">Similitud: {0:.2%}</div>
        <div class="patent-image-container">
            <img src="{1}" 
                 alt="[Image]" class="patent-thumbnail" 
                 onerror="this.onerror=null;this.src='{2}';">
        </div>
        <div class="google-patent-content-details">
            <p class="google-patent-title">{3}</p>
            <p class="google-patent-summary">{4}</p>
            <p class="google-patent-meta">Patente: {5}</p>
        </div>
        <button type="submit" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; border: none; background: transparent;"></button>
    </div>
    """.format(score, patent_image_url if patent_image_url else default_image_url, default_image_url, escaped_patent_title, escaped_patent_summary_short, patent_number_found)
                                    st.markdown(card_html, unsafe_allow_html=True)
                                    
                                    # Hidden button that gets clicked by the overlaying transparent button
                                    clicked_card = st.form_submit_button(
                                        label="Ver Detalles", # This label is hidden by CSS
                                        key=f"hidden_card_button_{idx}",
                                        help="Haz clic para ver los detalles de la patente"
                                    )
                                    # If this hidden button is clicked, store the index and rerun
                                    if clicked_card:
                                        st.session_state.selected_patent_idx = idx
                                        st.rerun() # Rerun to switch to detail view
                                
                except Exception as e: # End of the try block, start of the except block
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")
