import streamlit as st
import tempfile
import os
import logging
# Manejo imagenes y videos
from src.inference import VideoDetector # Asumimos que VideoDetector tambien puede manejar imagenes
from PIL import Image # Importar Pillow para manejar imagenes
import cv2 # Importar OpenCV para leer imagenes si es necesario

# Configurar logging para la aplicación
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asigno confidence a usar
detection_confidence = 0.5

# Nombre del archivo del modelo (debe coincidir con el que esté en la carpeta model/)
MODEL_PATH = "model/best.pt"

def main():
    st.title("DDAO - MVP")
    st.subheader("Detección de Discapacitados para Asistencia Optimizada")

    # Verificar si el modelo existe
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error crítico: No se encontró el modelo en '{MODEL_PATH}'. Revisar carpeta 'model'.")
        st.stop()

    # Permitir subir archivos de imagen o video
    uploaded_file = st.file_uploader(
        "Sube una imagen (JPG, PNG) o un video (MP4, MOV, AVI, etc.)",
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'mkv'] # Tipos de archivo permitidos
    )

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # Verificar si es una imagen o un video
        if file_extension in ['.jpg', '.jpeg', '.png']:
            handle_image_upload(uploaded_file)
        elif file_extension in ['.mp4', '.mov', '.avi', '.mkv']:
            handle_video_upload(uploaded_file)
        else:
            st.error(f"Tipo de archivo no soportado: {file_extension}")

def handle_image_upload(uploaded_file):
    """Maneja la subida y procesamiento de una imagen."""
    st.subheader("Imagen Original")
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Subida', width="stretch")

    if st.button("Procesar Imagen"):
        st.info("Iniciando detección en la imagen...")
        try:
            import numpy as np # Importar numpy si no está ya importado globalmente
            # Inicializar el detector
            detector = VideoDetector(MODEL_PATH) # <-- Usar VideoDetector en lugar de Detector

            # Convertir PIL Image a array de OpenCV (BGR)
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Realizar predicción y anotación usando el metodo de la clase VideoDetector
            detected_image = detector.detect_and_annotate_image(opencv_image, conf_threshold=detection_confidence)
            # Bajo imagen detectada
            annotated_opencv_image = detected_image.plot()

            # TODO Extraer detecciones
            boxes = detected_image.boxes
            class_names = detected_image.names  # dict basado en classes.txt: {0: 'muleta', 1: 'silla_de_ruedas', ...}

            print(detected_image.boxes)

            # Carga de objetos detectados
            detected_objects = []
            for box in boxes:
                cls_id = int(box.cls.item()) # Class ID
                conf = float(box.conf.item()) # Confidence
                if conf >= detection_confidence:
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
            detector.detect_and_annotate_video(temp_input_path, temp_output_path, conf_threshold=detection_confidence)

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
