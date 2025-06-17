import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io
import html

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Explorador de Soluciones Técnicas (Patentes)")

# No custom CSS for styling in this version
# Removed: st.markdown("""...style...""", unsafe_allow_html=True)

# Removed: Magnifying glass SVG as it was part of custom HTML/CSS
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
# User confirmed it's 'patentes.xlsx' (Excel)
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
                        for i, idx in enumerate(top_results_indices):
                            score = cosine_scores[idx].item()
                            # Use the original (untranslated) title and abstract for display
                            # Access columns using their normalized names
                            patent_title = df_patents.iloc[idx]['title (original language)']
                            patent_summary = df_patents.iloc[idx]['abstract (original language)']
                            
                            # It's possible 'numero de patente' might not exist in the new dataset.
                            # We can also check if a column like 'Publication Number' or 'Patent Number' exists and and use that.
                            # Access columns using their normalized names
                            patent_number_found = 'N/A'
                            if 'numero de patente' in df_patents.columns:
                                patent_number_found = df_patents.iloc[idx]['numero de patente']
                            elif 'publication number' in df_patents.columns: # Check normalized names
                                patent_number_found = df_patents.iloc[idx]['publication number']
                            elif 'patent number' in df_patents.columns: # Check normalized names
                                patent_number_found = df_patents.iloc[idx]['patent number']
                            
                            # Escape HTML-breaking characters in the content
                            escaped_patent_title = html.escape(patent_title)
                            escaped_patent_summary_short = html.escape(patent_summary[:100]) + "..."

                            # Display using standard Streamlit components
                            st.subheader(f"**{i+1}. {escaped_patent_title}**")
                            st.write(f"Similitud: {score:.2%}")
                            st.write(f"Resumen: {escaped_patent_summary_short}")
                            st.write(f"Patente: {patent_number_found}")
                            st.markdown("---") # Separator between patents

                except Exception as e: # End of the try block, start of the except block
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")

# No custom JavaScript for syncing is needed now.

