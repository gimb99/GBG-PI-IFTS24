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
    # Para Anaconda
    pip install -r requirements.txt
    ```

4.  **Coloca tu modelo entrenado:**
    *   Asegúrate de tener el archivo `best.pt` del modelo YOLOv8 entrenado.
    *   Créala carpeta `model` si no existe (`mkdir model`).
    *   Mueve el archivo `best.pt` a la carpeta `./model/`.

## Configuración de Notificaciones (Opcional)
Para habilitar las notificaciones externas via Telegram:
1.  a. **Crear un Bot de Telegram:**
    *   Habla con [@BotFather](https://t.me/BotFather) en Telegram.
    *   Usa el comando `/newbot` y sigue las instrucciones para crear tu bot.
    *   Copia el **Token de acceso HTTP** que te proporciona BotFather.
2.  **Obtener tu ID de Chat de Telegram:**
    *   Habla con tu nuevo bot recién creado (envía `/start`).
    *   Usa un bot como [@userinfobot](https://t.me/userinfobot) para obtener tu ID de usuario de Telegram.
3.  **Configurar las Variables de Entorno:**
    *   En la raíz de tu proyecto, crea una carpeta llamada `.streamlit` (si no existe).
    *   Dentro de la carpeta `.streamlit`, crea un archivo llamado `secrets.toml`.
    *   Agrega las siguientes líneas al archivo `secrets.toml`, reemplazando `TU_TOKEN_AQUI` y `TU_ID_AQUI` con los valores obtenidos anteriormente:
      ```toml
      # .streamlit/secrets.toml
      TELEGRAM_BOT_TOKEN = "TU_TOKEN_AQUI"
      TELEGRAM_CHAT_ID = "TU_ID_AQUI"
      ```
    *   **Importante:** Asegúrate de que el archivo `.streamlit/secrets.toml` esté incluido en tu archivo `.gitignore` para que no se suba accidentalmente a tu repositorio de código.
4.  La aplicación leerá automáticamente estas credenciales al iniciar.


## Uso
1.  Desde la raíz del proyecto, ejecuta la aplicación Streamlit:
    ```bash
    streamlit run app.py
    ```
2.  Abre tu navegador en la URL que indique la consola (generalmente `http://localhost:8501`).
3.  Utiliza el botón "Sube una imagen (JPG, PNG) o un video (MP4, MOV, AVI, etc.)" para seleccionar un archivo multimedia.
4.  Ajusta el "Umbral de confianza" según sea necesario.
5.  Haz clic en el botón "Procesar Imagen" o "Procesar Video".
6.  Observa el multimedia procesado con las detecciones superpuestas en la sección "Multimedia Procesada".
7.  (Si se configuraron notificaciones) Si se detectan objetos ortopédicos, se mostrará una alerta en la interfaz y (opcionalmente) se enviará una notificación a Telegram.

## Estructura del Proyecto
*   `app.py`: Punto de entrada de la aplicación Streamlit. Contiene la lógica del frontend y orquesta la inferencia.
*   `requirements.txt`: Lista las dependencias de Python necesarias.
*   `model/best.pt`: Modelo YOLOv8 entrenado para la detección de objetos ortopédicos.
*   `src/inference.py`: Contiene la lógica principal para cargar el modelo y realizar la inferencia sobre frames de video.
*   `src/utils.py`: Funciones auxiliares para tareas como el envío de notificaciones.
*   `.streamlit/secrets.toml`: Archivo opcional para almacenar credenciales de notificaciones (no debe subirse al repositorio).
*   `.gitignore`: Indica al sistema de control de versiones qué archivos o carpetas ignorar (por ejemplo, el modelo `best.pt` si es muy grande, o archivos temporales).

## Notas adicionales
*   Este proyecto es un MVP. La funcionalidad principal es analizar videos de entrada y mostrar las detecciones.
*   El modelo `best.pt` debe haber sido entrenado previamente con un dataset de imágenes etiquetadas que contengan los objetos de interés (sillas de ruedas, bastones, etc.).
*   La configuración de notificaciones via Telegram es opcional y requiere pasos adicionales de configuración.
