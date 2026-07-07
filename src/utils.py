import requests
import logging
import os

logger = logging.getLogger(__name__)

def send_telegram_notification(message: str):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    # Verificar si las credenciales están disponibles
    if not bot_token or not chat_id:
        logger.error("Faltan credenciales de Telegram (TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID) en las variables de entorno.")
        st.error("Error: Configuración de notificaciones incompleta. Contacte al administrador.") # Opcional: mensaje en Streamlit si es relevante aquí
        return # Salir si no hay credenciales
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        if response.status_code == 200:
            logger.info("Notificacion Telegram enviada.")
        else:
            # Imprimir el código de estado y el texto de error de Telegram para diagnóstico
            logger.error(f"Error Telegram {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error de red o conexión al enviar notificación Telegram: {e}")
