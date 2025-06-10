import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import io

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Explorador de Soluciones Técnicas (Patentes)")
st.title("💡 Explorador de Soluciones Técnicas (Patentes)")
st.markdown("Describe tu problema técnico o necesidad funcional y encuentra patentes relevantes.")

# --- Funciones para cargar y procesar datos/modelos ---

@st.cache_resource
def load_embedding_model():
    """
    Carga el modelo pre-entrenado de SentenceTransformer.
    Se utiliza `st.cache_resource` para cargar el modelo una sola vez y reutilizarlo,
    mejorando el rendimiento de la aplicación.
    """
    # Modelo multilingüe que funciona bien para español y semantic similarity
    # Otros modelos posibles: 'distiluse-base-multilingu al-cased-v1'
    # Consulta: https://www.sbert.net/docs/pretrained_models.html
    st.write("Cargando el modelo de embeddings (esto puede tardar un momento)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    st.success("Modelo de embeddings cargado correctamente.")
    return model

@st.cache_data
def process_patent_data(file_path):
    """
    Procesa el archivo Excel de patentes desde una ruta local.
    Lee el archivo, combina título y resumen, y genera los embeddings.
    Se utiliza `st.cache_data` para almacenar en caché los datos procesados
    y los embeddings generados, evitando reprocesamientos innecesarios.
    """
    if file_path:
        try:
            # Lee el archivo Excel desde la ruta local
            df = pd.read_excel(file_path)

            # Validar que las columnas necesarias existan (ahora en minúsculas)
            required_columns = ['titulo', 'resumen']
            if not all(col in df.columns for col in required_columns):
                st.error(f"El archivo Excel debe contener las columnas: {', '.join(required_columns)}")
                return None, None

            # Rellenar valores nulos en 'titulo' y 'resumen' con cadenas vacías para evitar errores
            df['titulo'] = df['titulo'].fillna('')
            df['resumen'] = df['resumen'].fillna('')

            # Combina el título y el resumen para crear una descripción completa de la patente
            df['Descripción Completa'] = df['titulo'] + ". " + df['resumen']

            # Carga el modelo de embeddings
            model = load_embedding_model()

            st.write("Generando embeddings para las patentes (esto puede tardar un momento)...")
            # Genera embeddings para todas las descripciones de patentes
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

# --- Sección para la carga automática del archivo Excel local ---
st.header("1. Patentes cargadas")

# El nombre del archivo Excel local en el mismo repositorio
excel_file_name = "patentes.xlsx"

# Inicializar model y corpus_embeddings
model = None
df_patents = None
patent_embeddings = None

# Procesa los datos automáticamente al iniciar la aplicación
with st.spinner(f"Cargando y procesando patentes desde '{excel_file_name}'..."):
    df_patents, patent_embeddings = process_patent_data(excel_file_name)

if df_patents is not None and patent_embeddings is not None:
    model = load_embedding_model()
    st.success(f"Archivo '{excel_file_name}' cargado y {len(df_patents)} patentes procesadas.")
    st.dataframe(df_patents[['titulo', 'resumen']].head()) # Muestra las primeras filas para verificación
else:
    st.error(f"No se pudo cargar o procesar el archivo Excel '{excel_file_name}'. "
             "Por favor, verifica que el archivo exista en el mismo directorio de 'app.py' en tu repositorio de GitHub "
             "y que contenga las columnas 'titulo' y 'resumen'.")


# --- Sección para la entrada de la problemática y búsqueda ---
st.header("2. Describe tu Problema o Necesidad")
problem_description = st.text_area(
    "Ingresa la descripción de tu problema técnico o necesidad funcional:",
    "Necesito un sistema de cierre hermético para envases sin usar calor.",
    height=100
)

num_results = st.slider("Número de patentes a mostrar:", min_value=1, max_value=20, value=5)

if st.button("Buscar Soluciones"):
    if df_patents is None or patent_embeddings is None or model is None:
        st.error("Por favor, asegúrate de que el archivo de patentes se haya cargado correctamente.")
    elif not problem_description.strip():
        st.warning("Por favor, ingresa una descripción del problema.")
    else:
        with st.spinner("Buscando patentes relevantes..."):
            try:
                # Genera el embedding de la descripción del problema
                query_embedding = model.encode(problem_description, convert_to_tensor=True)

                # Calcula la similitud coseno entre el problema y todas las patentes
                # Es importante que ambos tensores estén en el mismo dispositivo (CPU/GPU)
                cosine_scores = util.cos_sim(query_embedding, patent_embeddings)[0]

                # Obtiene los índices de las patentes más similares
                top_results_indices = np.argsort(-cosine_scores.cpu().numpy())[:num_results]

                st.subheader(f"Patentes más relevantes para: '{problem_description}'")

                if len(top_results_indices) == 0:
                    st.info("No se encontraron patentes relevantes con la descripción proporcionada.")
                else:
                    for i, idx in enumerate(top_results_indices):
                        score = cosine_scores[idx].item()
                        patent_title = df_patents.iloc[idx]['titulo'] # Usando 'titulo' en minúscula
                        patent_summary = df_patents.iloc[idx]['resumen'] # Usando 'resumen' en minúscula
                        
                        # Intenta obtener el número de patente si existe la columna
                        # Aquí también se buscará 'Número de Patente' en minúscula si el usuario lo cambió
                        patent_number = df_patents.iloc[idx]['numero de patente'] if 'numero de patente' in df_patents.columns else 'N/A'

                        st.markdown(f"---")
                        st.markdown(f"**{i+1}. Título:** {patent_title}")
                        st.markdown(f"**Número de Patente:** {patent_number}")
                        st.markdown(f"**Similitud:** {score:.4f}")
                        with st.expander("Ver Resumen Completo"):
                            st.write(patent_summary)

            except Exception as e:
                st.error(f"Ocurrió un error durante la búsqueda: {e}")

st.markdown("---")
st.markdown("Construido con ❤️ usando Streamlit y Sentence-Transformers.")
