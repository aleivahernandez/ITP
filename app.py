import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Explorador de Soluciones Técnicas (Patentes)")
st.title("💡 Explorador de Soluciones Técnicas (Patentes)")
st.markdown("Describe tu problema técnico o necesidad funcional y encuentra patentes relevantes.")

# --- Functions for loading and processing data/models ---

@st.cache_resource
def load_embedding_model():
    """
    Loads the pre-trained SentenceTransformer model.
    `st.cache_resource` is used to load the model only once and reuse it,
    improving application performance.
    """
    # Multilingual model that works well for Spanish and semantic similarity
    # Other possible models: 'distiluse-base-multilingu al-cased-v1'
    # Check: https://www.sbert.net/docs/pretrained_models.html
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
st.header("1. Describe tu Problema o Necesidad") # Re-numbered to 1 as previous section is hidden
problem_description = st.text_area(
    "Ingresa la descripción de tu problema técnico o necesidad funcional:",
    "Necesito un sistema de cierre hermético para envases sin usar calor.",
    height=100
)

# Fixed number of results, no slider
MAX_RESULTS = 3

if st.button("Buscar Soluciones"):
    if not problem_description.strip():
        st.warning("Por favor, ingresa una descripción del problema.")
    else:
        with st.spinner("Buscando patentes relevantes..."):
            try:
                # Generates the embedding of the problem description
                query_embedding = model.encode(problem_description, convert_to_tensor=True)

                # Calculates cosine similarity between the problem and all patents
                # It's important that both tensors are on the same device (CPU/GPU)
                cosine_scores = util.cos_sim(query_embedding, patent_embeddings)[0]

                # Gets the indices of the most similar patents
                top_results_indices = np.argsort(-cosine_scores.cpu().numpy())[:MAX_RESULTS] # Use MAX_RESULTS

                st.subheader(f"Patentes más relevantes para: '{problem_description}'")

                if len(top_results_indices) == 0:
                    st.info("No se encontraron patentes relevantes con la descripción proporcionada.")
                else:
                    for i, idx in enumerate(top_results_indices):
                        score = cosine_scores[idx].item()
                        patent_title = df_patents.iloc[idx]['titulo'] # Using 'titulo' in lowercase
                        patent_summary = df_patents.iloc[idx]['resumen'] # Using 'resumen' in lowercase
                        
                        # Tries to get the patent number if the column exists
                        # Also looks for 'numero de patente' in lowercase if the user changed it
                        patent_number = df_patents.iloc[idx]['numero de patente'] if 'numero de patente' in df_patents.columns else 'N/A'

                        st.markdown(f"---")
                        st.markdown(f"**{i+1}. Título:** {patent_title}")
                        st.markdown(f"**Número de Patente:** {patent_number}")
                        st.markdown(f"**Similitud:** {score:.4f}")
                        with st.expander("Ver Resumen Completo"):
                            st.write(patent_summary)

            except Exception as e:
                st.error(f"Ocurrió un error durante la búsqueda: {e}")
