# IFTS24 - Proyecto Integrador - 2026
# Sistema de Detección de Objetos Ortopédicos

## Descripción
Este proyecto es un MVP (Producto Mínimo Viable) para un sistema de detección de objetos ortopédicos (sillas de ruedas, bastones, etc.) en videos. Su objetivo es identificar personas con discapacidades motoras en espacios públicos a partir de videos de entrada, con vistas a futuras aplicaciones en cámaras de seguridad para mejorar la accesibilidad, como en paradas de transporte público.

## Tecnologías Utilizadas
*   **Python:** Lenguaje de programación principal.
*   **Ultralytics YOLOv8:** Modelo de visión por computadora utilizado para la detección de objetos.
*   **Streamlit:** Framework para la construcción de la interfaz web local.
*   **OpenCV (cv2):** Librería para el manejo de imágenes y video.

## Instalación

1.  **Clona este repositorio:**
    ```bash
    git clone https://github.com/gimb99/GBG-PI-IFTS24.git
    cd GBG-PI-IFTS24
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Coloca tu modelo entrenado:**
    *   Asegúrate de tener el archivo `best.pt` del modelo YOLOv8 entrenado.
    *   Créala carpeta `model` si no existe (`mkdir model`).
    *   Mueve el archivo `best.pt` a la carpeta `./model/`.

## Uso
1.  Desde la raíz del proyecto, ejecuta la aplicación Streamlit:
    ```bash
    streamlit run app.py
    ```
2.  Abre tu navegador en la URL que indique la consola (generalmente `http://localhost:8501`).
3.  Utiliza el botón "Browse files" para subir un archivo de video (MP4, MOV, AVI, etc.).
4.  Haz clic en el botón "Procesar Video" (o similar, dependiendo de cómo lo llames en `app.py`).
5.  Observa el video procesado con las detecciones superpuestas.

## Estructura del Proyecto
*   `app.py`: Punto de entrada de la aplicación Streamlit. Contiene la lógica del frontend y orquesta la inferencia.
*   `requirements.txt`: Lista las dependencias de Python necesarias.
*   `model/best.pt`: Modelo YOLOv8 entrenado para la detección de objetos ortopédicos.
*   `src/inference.py`: Contiene la lógica principal para cargar el modelo y realizar la inferencia sobre frames de video.
*   `src/utils.py`: (Opcional) Funciones auxiliares para tareas como manejo de video.
*   `.gitignore`: Indica al sistema de control de versiones qué archivos o carpetas ignorar (por ejemplo, el modelo `best.pt` si es muy grande, o archivos temporales).

## Notas
*   Este proyecto es un MVP. La funcionalidad principal es analizar videos de entrada y mostrar las detecciones.
*   El modelo `best.pt` debe haber sido entrenado previamente con un dataset de imágenes etiquetadas que contengan los objetos de interés (sillas de ruedas, bastones, etc.).
