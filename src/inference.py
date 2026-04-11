import cv2
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class VideoDetector:
    """
    Clase para cargar el modelo YOLO y realizar detecciones en videos.
    """
    def __init__(self, model_path: str):
        """
        Inicializa el detector con el modelo especificado.

        Args:
            model_path (str): Ruta al archivo .pt del modelo entrenado.
        """
        logger.info(f"Cargando modelo YOLO desde {model_path}")
        try:
            self.model = YOLO(model_path)
            logger.info("Modelo YOLO cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo YOLO: {e}")
            raise

    def detect_and_annotate_video(self, input_video_path: str, output_video_path: str, conf_threshold: float = 0.5):
        """
        Lee un video, realiza detecciones en cada frame y guarda un video anotado.

        Args:
            input_video_path (str): Ruta al video de entrada.
            output_video_path (str): Ruta donde se guardará el video de salida anotado.
            conf_threshold (float): Umbral de confianza para las detecciones.
        """
        logger.info(f"Iniciando detección en video: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video de entrada: {input_video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para MP4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar predicción
            results = self.model(frame, conf=conf_threshold)

            # Anotar el frame con las detecciones
            annotated_frame = results[0].plot()

            # Escribir el frame anotado en el video de salida
            out.write(annotated_frame)

            frame_count += 1
            # Opcional: Loggear progreso cada N frames
            # if frame_count % 30 == 0:
            #     logger.info(f"Procesando frame {frame_count}...")

        cap.release()
        out.release()
        logger.info(f"Detección finalizada. Video anotado guardado en: {output_video_path}")

# Ejemplo de uso (descomentar si se ejecuta este archivo directamente)
# if __name__ == "__main__":
#     detector = VideoDetector("model/best.pt")
#     detector.detect_and_annotate_video("ruta/al/video_entrada.mp4", "ruta/al/video_salida_annotated.mp4")
