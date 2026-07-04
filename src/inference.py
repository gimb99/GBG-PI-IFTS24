import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os

logger = logging.getLogger(__name__)

class VideoDetector:
    # Carga modelo YOLO
    def __init__(self, model_path: str):
        """
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

    def _run_inference(self, input_data, conf_threshold: float = 0.5):
        """
        Función privada para inferir modelo
        Puede manejar arrays de imagen (OpenCV) o frames individuales de video.
        """
        results = self.model(input_data, conf=conf_threshold)
        return results[0]

    def detect_and_annotate_image(self, image_array_or_path, conf_threshold: float = 0.5):
        """
        Toma array OpenCV o ruta de archivo y devuelve array de imagen anotada

        Args:
            image_array_or_path (np.ndarray or str): Imagen como array NumPy (BGR) o ruta del archivo.
            conf_threshold (float): Umbral de confianza para las detecciones.

        Returns:
            np.ndarray: Imagen anotada como array NumPy (BGR).
        """
        logger.info("Iniciando detección en imagen.")
        
        if isinstance(image_array_or_path, str):
            # Si es una ruta, cargar la imagen
            if not os.path.exists(image_array_or_path):
                 logger.error(f"La ruta de imagen no existe: {image_array_or_path}")
                 raise FileNotFoundError(f"La ruta de imagen no existe: {image_array_or_path}")
            image_array = cv2.imread(image_array_or_path)
            if image_array is None:
                logger.error(f"No se pudo leer la imagen desde la ruta: {image_array_or_path}")
                raise ValueError(f"No se pudo leer la imagen desde la ruta: {image_array_or_path}")
        elif isinstance(image_array_or_path, np.ndarray):
            # Si ya es un array, usarlo directamente
            image_array = image_array_or_path
        else:
            logger.error(f"Tipo de entrada no válido para imagen: {type(image_array_or_path)}")
            raise TypeError(f"Tipo de entrada no válido para imagen: {type(image_array_or_path)}. Se espera str (ruta) o np.ndarray.")

        # Ejecutar inferencia (usando la función común)
        results = self._run_inference(image_array, conf_threshold)

        # Anotar la imagen
        annotated_image = results.plot()

        logger.info("Detección en imagen finalizada.")
        return results


    def detect_and_annotate_video(self, input_video_path: str, output_video_path: str, conf_threshold: float = 0.5):
        """
        Lee video, detecta por cada frame y guarda un video anotado

        Args:
            input_video_path (str): Ruta al video de entrada.
            output_video_path (str): Ruta donde se guardará el video de salida anotado.
            conf_threshold (float): Umbral de confianza para las detecciones.
        """
        logger.info(f"Iniciando detección en video: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video de entrada: {input_video_path}")
            raise IOError(f"No se pudo abrir el video de entrada: {input_video_path}")

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

            # Ejecutar inferencia en el frame (usando la función común)
            results = self._run_inference(frame, conf_threshold)

            # Anotar el frame
            annotated_frame = results.plot()

            # Escribir el frame anotado en el video de salida
            out.write(annotated_frame)

            frame_count += 1
            # Opcional: Loggear progreso cada N frames
            # if frame_count % 30 == 0:
            #     logger.info(f"Procesando frame {frame_count}...")

        cap.release()
        out.release()
        logger.info(f"Detección en video finalizada. Video anotado guardado en: {output_video_path}")


# Ejemplo de uso (descomentar si se ejecuta este archivo directamente)
# if __name__ == "__main__":
#     detector = VideoDetector("model/best.pt") # <-- Cambiado a VideoDetector
#     # Ejemplo con imagen
#     # annotated_img = detector.detect_and_annotate_image("ruta/a/imagen.jpg", conf=0.5)
#     # cv2.imshow("Annotated Image", annotated_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # Ejemplo con video
#     # detector.detect_and_annotate_video("ruta/al/video_entrada.mp4", "ruta/al/video_salida_annotated.mp4", conf=0.5)