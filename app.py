elif st.session_state.current_view == 'detail':
    selected_patent = st.session_state.selected_patent
    if selected_patent:
        # Envoltorio para todo el contenido del detalle
        st.markdown("<div class='detail-content-wrapper'>", unsafe_allow_html=True)
        
        # 1. Título de ancho completo
        st.markdown(f"<h1 class='full-patent-title'>{html.escape(selected_patent['title'])}</h1>", unsafe_allow_html=True)
        
        # --- INICIO: ESTRUCTURA DE 2 COLUMNAS CORREGIDA ---
        
        # Prepara la columna de la imagen (solo si existe una URL)
        image_column_html = ""
        if selected_patent.get('image_url'):
            image_column_html = f"""
            <div class='detail-image-column'>
                <img src="{selected_patent['image_url']}" alt="Imagen de la patente">
            </div>
            """
        
        # Construye el cuerpo principal que contiene las columnas
        body_html = f"""
        <div class='detail-body-container'>
            {image_column_html}
            <div class='detail-abstract-column'>
                <p class='full-patent-abstract'>{html.escape(selected_patent['abstract'])}</p>
            </div>
        </div>
        """

        # Llama a st.markdown UNA SOLA VEZ para el cuerpo, asegurando que se interprete el HTML
        st.markdown(body_html, unsafe_allow_html=True)
        
        # --- FIN: ESTRUCTURA CORREGIDA ---

        # 3. Metadatos y botón de volver
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
        show_search_view()
