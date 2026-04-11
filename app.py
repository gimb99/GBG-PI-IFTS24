import streamlit as st
from src.inference import VideoDetector
import tempfile
import os
import logging

# Configurar logging para la aplicación
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nombre del archivo del modelo (debe coincidir con el que esté en la carpeta model/)
MODEL_PATH = "model/best.pt"

def main():
    st.title("Sistema de Detección de Objetos Ortopédicos - MVP")

    # Verificar si el modelo existe
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error crítico: No se encontró el modelo en '{MODEL_PATH}'. Asegúrate de haberlo colocado en la carpeta 'model'.")
        st.stop()

    uploaded_file = st.file_uploader("Sube un archivo de video (MP4, MOV, AVI, etc.)", type=['mp4', 'mov', 'avi', 'mkv'])

    if uploaded_file is not None:
        # Mostrar el video subido
        st.subheader("Video Original")
        # Usar un temporal file para manejar el video subido por Streamlit
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        temp_input_path = tfile.name
        tfile.close()

        # Mostrar video original
        st.video(temp_input_path)

        # Botón para iniciar el procesamiento
        if st.button("Procesar Video"):
            st.info("Iniciando detección... Esto puede tardar unos minutos.")

            # Ruta temporal para el video de salida
            temp_output_path = tempfile.mktemp(suffix='.mp4')

            try:
                # Inicializar el detector
                detector = VideoDetector(MODEL_PATH)
                # Realizar la detección y guardar video anotado
                detector.detect_and_annotate_video(temp_input_path, temp_output_path, conf_threshold=0.5)

                st.success("¡Detección finalizada!")

                # Mostrar video anotado
                st.subheader("Video con Detecciones")
                st.video(temp_output_path)

            except Exception as e:
                st.error(f"Ocurrió un error durante la detección: {e}")
                logger.error(f"Error en app.py: {e}")

            finally:
                # Limpiar archivos temporales
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

if __name__ == "__main__":
    main()
