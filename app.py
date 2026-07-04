import streamlit as st
import tempfile
import os
import logging
import numpy as np
# Manejo imagenes y videos
from src.inference import VideoDetector # Asumimos que VideoDetector tambien puede manejar imagenes
from PIL import Image # Importar Pillow para manejar imagenes
import cv2 # Importar OpenCV para leer imagenes si es necesario

# Configurar logging para la aplicación
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Config global
# Asigno confidence a usar
DEFAULT_CONFIDENCE = 0.5
# Nombre del archivo del modelo (coincidir con carpeta model/)
MODEL_PATH = "model/best.pt"

def main():
    ## HEADERS
    st.set_page_config(page_title="DDAO - MVP", page_icon="♿", layout="wide")
    st.title("DDAO - MVP")
    st.markdown("## Detección de Discapacitados para Asistencia Optimizada")

    # Verificar si el modelo existe
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error crítico: No se encontró el modelo en '{MODEL_PATH}'. Revisar carpeta 'model'.")
        st.stop()
    
    ## TOP - Ingreso Multimedia
    # Permitir subir archivos de imagen o video
    st.markdown("---")
    st.subheader("Ingreso de Imágenes/Video")
    uploaded_file = st.file_uploader(
        "Sube una imagen (JPG, PNG) o un video (MP4, MOV, AVI, etc.)",
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'mkv'] # Tipos de archivo permitidos
    )

    # Crear layout de columnas - Pruebo fuera de if None
    col_left, col_right = st.columns([0.33, 0.66], gap="large")

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        """"
        # Verificar si es una imagen o un video
        if file_extension in ['.jpg', '.jpeg', '.png']:
            handle_image_upload(uploaded_file)
        elif file_extension in ['.mp4', '.mov', '.avi', '.mkv']:
            handle_video_upload(uploaded_file)
        else:
            st.error(f"Tipo de archivo no soportado: {file_extension}")
        """
        with col_left:
            st.markdown("### Pre-procesamiento")

            # Controles de pre-procesamiento
            detection_confidence = st.slider(
                "Umbral de confianza",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_CONFIDENCE,
                step=0.05,
                help="Valores más altos = menos falsos positivos pero posible pérdida de detecciones"
            )

            #st.info(f"Confianza configurada: **{detection_confidence:.2f}**")
            # Metadata del archivo
            st.markdown("#### Información del archivo")
            st.write(f"**Nombre:** {uploaded_file.name}")
            st.write(f"**Tamaño:** {uploaded_file.size / 1024:.2f} KB")
            st.write(f"**Tipo:** {file_extension}")
            
            # Fin col_left

        with col_right:
            # Multimedia procesada
            st.markdown("### Multimedia Procesada")
            output_placeholder = st.empty()
            
            # Registro de procesamiento
            st.markdown("### Registro de Procesamiento")
            log_container = st.container()
            
            # Procesar según tipo
            if file_extension in ['.jpg', '.jpeg', '.png']:
                handle_image_upload_layout(uploaded_file, detection_confidence, output_placeholder, log_container)
            elif file_extension in ['.mp4', '.mov', '.avi', '.mkv']:
                handle_video_upload_layout(uploaded_file, detection_confidence, output_placeholder, log_container)
            else:
                st.error(f"Tipo de archivo no soportado: {file_extension}")
            
            # Fin col_right

        # Fin if
    #Fin main

def handle_image_upload_layout(uploaded_file, confidence, output_placeholder, log_container):
    """Maneja imagen con nuevo layout."""
    
    # Mostrar imagen original
    image = Image.open(uploaded_file)
    output_placeholder.image(image, caption='Imagen Original', width="stretch")
    
    if st.button("🔍 Procesar Imagen", type="primary"):
        with log_container:
            st.info("⏳ Iniciando detección...")
        
        try:
            # Inicializar detector
            detector = VideoDetector(MODEL_PATH)
            
            # Convertir a OpenCV
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Inferencia
            # Realizar predicción y anotación usando el metodo de la clase VideoDetector
            detected_image = detector.detect_and_annotate_image(opencv_image, conf_threshold=confidence)
            
            # Extraer detecciones
            boxes = detected_image.boxes
            # dict basado en classes.txt: {0: 'muleta', 1: 'silla_de_ruedas', ...}
            class_names = detected_image.names 
            
            # Carga de objetos detectados
            detected_objects = []
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if conf >= confidence:
                    label = class_names[cls_id]
                    detected_objects.append((label, conf))
            
            # Mostrar resultados en log
            with log_container:
                if detected_objects:
                    count = len(detected_objects)
                    details = ", ".join([f"{lbl} ({conf:.2f})" for lbl, conf in detected_objects])
                    st.warning(f"✅ **Detección confirmada:** {count} objeto(s) detectado(s)\n\n📌 Detalles: {details}")
                else:
                    st.info("ℹ️ No se detectaron objetos ortopédicos con la confianza configurada.")
            
            # Mostrar imagen procesada
            annotated_opencv_image = detected_image.plot()
            annotated_pil_image = Image.fromarray(cv2.cvtColor(annotated_opencv_image, cv2.COLOR_BGR2RGB))
            
            output_placeholder.image(annotated_pil_image, caption='Imagen con Detecciones', width="stretch")
            
            with log_container:
                st.success("✨ Procesamiento completado exitosamente")
        
        except Exception as e:
            with log_container:
                st.error(f"❌ Error: {e}")
            logger.error(f"Error en handle_image_upload: {e}")

def handle_video_upload_layout(uploaded_file, confidence, output_placeholder, log_container):
    """Maneja video con nuevo layout."""
    
    # Guardar video temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    temp_input_path = tfile.name
    tfile.close()
    
    # Mostrar video original
    output_placeholder.video(temp_input_path)
    
    if st.button("🎬 Procesar Video", type="primary"):
        with log_container:
            st.info("⏳ Iniciando procesamiento de video...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        temp_output_path = tempfile.mktemp(suffix='.mp4')
        
        try:
            detector = VideoDetector(MODEL_PATH)
            
            # Procesar video (puedes agregar lógica de progreso aquí si modificas inference.py)
            detector.detect_and_annotate_video(temp_input_path, temp_output_path, conf_threshold=confidence)
            
            progress_bar.progress(100)
            status_text.text("✅ Video procesado")
            
            # Mostrar video procesado
            output_placeholder.video(temp_output_path)
            
            with log_container:
                st.success("✨ Procesamiento de video completado")
                st.info("📹 El video anotado muestra las detecciones frame a frame")
        
        except Exception as e:
            with log_container:
                st.error(f"❌ Error durante el procesamiento: {e}")
            logger.error(f"Error en handle_video_upload: {e}")
        
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

### Funciones legacy!!        
def handle_image_upload(uploaded_file):
    """Maneja la subida y procesamiento de una imagen."""
    st.subheader("Imagen Original")
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Subida', width="stretch")

    if st.button("Procesar Imagen"):
        st.info("Iniciando detección en la imagen...")
        try:
            # Inicializar el detector
            detector = VideoDetector(MODEL_PATH)

            # Convertir PIL Image a array de OpenCV (BGR)
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Realizar predicción y anotación usando el metodo de la clase VideoDetector
            detected_image = detector.detect_and_annotate_image(opencv_image, conf_threshold=DEFAULT_CONFIDENCE)
            # Bajo imagen detectada
            annotated_opencv_image = detected_image.plot()

            # Extraer detecciones
            boxes = detected_image.boxes
            class_names = detected_image.names  # dict basado en classes.txt: {0: 'muleta', 1: 'silla_de_ruedas', ...}

            print(detected_image.boxes)

            # Carga de objetos detectados
            detected_objects = []
            for box in boxes:
                cls_id = int(box.cls.item()) # Class ID
                conf = float(box.conf.item()) # Confidence
                if conf >= DEFAULT_CONFIDENCE:
                    label = class_names[cls_id]
                    detected_objects.append((label, conf))

            # Notifico detecciones
            if detected_objects:
                st.warning(f"Detección confirmada: {', '.join([f'{lbl} ({conf:.2f})' for lbl, conf in detected_objects])}")
            else:
                st.info("No se detectaron objetos ortopédicos con la confianza configurada.")

            # Convertir la imagen anotada de vuelta a PIL para mostrarla en Streamlit
            annotated_pil_image = Image.fromarray(cv2.cvtColor(annotated_opencv_image, cv2.COLOR_BGR2RGB))

            st.success("¡Detección en imagen finalizada!")
            st.subheader("Imagen procesada")
            st.image(annotated_pil_image, caption='Imagen con Detecciones', width="stretch")

        except Exception as e:
            st.error(f"Ocurrió un error durante la detección de la imagen: {e}")
            logger.error(f"Error en handle_image_upload: {e}")

def handle_video_upload(uploaded_file):
    """Maneja la subida y procesamiento de un video."""
    st.subheader("Video Original")
    # Usar un temporal file para manejar el video/imagen subido por Streamlit
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    temp_input_path = tfile.name
    tfile.close()

    # Mostrar video original
    st.video(temp_input_path)

    # Botón para iniciar el procesamiento
    if st.button("Procesar Video"):
        st.info("Iniciando detección en el video... Esto puede tardar unos minutos.")

        # Ruta temporal para el video de salida
        temp_output_path = tempfile.mktemp(suffix='.mp4')

        try:
            # Inicializar el detector
            detector = VideoDetector(MODEL_PATH)
            # Realizar la detección y guardar video anotado
            detector.detect_and_annotate_video(temp_input_path, temp_output_path, conf_threshold=DEFAULT_CONFIDENCE)

            st.success("¡Detección en video finalizada!")

            # Mostrar video anotado
            st.subheader("Video con Detecciones")
            st.video(temp_output_path)

        except Exception as e:
            st.error(f"Ocurrió un error durante la detección del video: {e}")
            logger.error(f"Error en handle_video_upload: {e}")

        finally:
            # Limpiar archivos temporales
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

if __name__ == "__main__":
    main()
