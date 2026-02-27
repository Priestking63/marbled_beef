
import os
import sys
import subprocess
import threading
import time
import asyncio
import aiohttp
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load ENV
load_dotenv()

# Config
BOT_TOKEN = os.getenv('BOT_TOKEN')
API_URL = os.getenv('API_URL', 'http://localhost:8000/predict')
BASE_URL = 'http://localhost:8000'
PORT = 8000

# Check token
if not BOT_TOKEN or BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
    print("="*60)
    print("BOT_TOKEN не настроен!")
    print("="*60)
    print("\n1. Получите токен в @BotFather (Telegram)")
    print("2. Откройте файл .env")
    print("3. Замените YOUR_BOT_TOKEN_HERE на ваш токен")
    print("="*60)
    sys.exit(1)

# Server
server_process = None
server_ready = False

def start_fastapi_server():
    """Запуск сервера FastAPI в отдельном потоке"""
    global server_process, server_ready

    # Используем Python из venv, где установлены все зависимости
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_executable = os.path.join(script_dir, '.venv', 'Scripts', 'python.exe')
    app_path = os.path.join(script_dir, 'app.py')

    print("\nЗапуск сервера FastAPI...", flush=True)
    print(f"   Python: {python_executable}", flush=True)
    print(f"   App: {app_path}", flush=True)
    print("   URL: http://localhost:8000", flush=True)
    print("   Docs: http://localhost:8000/docs", flush=True)

    server_process = subprocess.Popen(
        [python_executable, app_path],
        cwd=script_dir,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
    )

    # Ожидание сервера с выводом статуса
    print("   Ожидание сервера...")
    max_wait = 60
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            import requests
            response = requests.get(f'{BASE_URL}/health', timeout=5)
            if response.status_code == 200:
                print("   Сервер готов!")
                server_ready = True
                return True
        except Exception as e:
            print(f"   Ожидание... ({e})")
        time.sleep(2)

    print("   Превышено время ожидания!")
    return False

# API Client
async def process_image_api(image_bytes: bytes, conf_threshold: float = 0.5) -> bytes:
    """Отправка изображения в API и получение обработанного результата"""
    global server_ready

    if not server_ready:
        raise Exception("Сервер не готов. Пожалуйста, подождите...")
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', image_bytes, filename='photo.jpg')
        data.add_field('conf_threshold', str(conf_threshold))
        
        try:
            async with session.post(API_URL, data=data, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {str(e)}")

# Telegram Handlers
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка фотографий"""
    try:
        photo = await update.message.photo[-1].get_file()
        temp_path = f"temp_{update.message.message_id}.jpg"
        await photo.download_to_drive(temp_path)
        
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        
        status_msg = await update.message.reply_text(
            "Анализ стейка...\nПожалуйста, подождите..."
        )

        try:
            result_image_bytes = await process_image_api(image_bytes)
            await status_msg.delete()

            await update.message.reply_photo(
                photo=result_image_bytes,
                caption=(
                    "**Анализ завершен!**\n\n"
                    "Обнаружены стейки с метками:\n"
                    "- Тип стейка (рибай, филе, стриплойн)\n"
                    "- Мраморность (choice, select, standard)\n\n"
                    "Отправляйте ещё фото!"
                ),
                parse_mode='Markdown'
            )
            
        except Exception as e:
            await status_msg.edit_text(f"Error: {str(e)}")
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    await update.message.reply_text(
        "**Бот для классификации стейков**\n\n"
        "Отправьте фото стейка, и я определю:\n"
        "- Тип стейка (рибай, филе, стриплойн)\n"
        "- Степень мраморности (choice, select, standard)\n\n"
        "**Как использовать:**\n"
        "1. Сделайте фото стейка\n"
        "2. Отправьте в этот чат\n"
        "3. Получите анализ с метками\n\n"
        "Жду ваше фото!"
    )

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /start"""
    await update.message.reply_text(
        "**Добро пожаловать в Steak Bot!**\n\n"
        "Я могу определить тип стейка и мраморность по фото.\n\n"
        "**Отправьте фото стейка**, чтобы получить:\n"
        "- Тип стейка (рибай, филе, стриплойн)\n"
        "- Степень мраморности (choice, select, standard)\n"
        "- Визуальные метки на изображении\n\n"
        "Введите /help для дополнительной информации."
    )

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /help"""
    await update.message.reply_text(
        "**Помощь**\n\n"
        "**Команды:**\n"
        "/start - Запустить бота\n"
        "/help - Эта справка\n"
        "/status - Статус сервера\n\n"
        "**Как использовать:**\n"
        "1. Сделайте четкое фото стейка\n"
        "2. Хорошее освещение, вид сверху\n"
        "3. Отправьте фото в чат\n"
        "4. Получите анализ через несколько секунд\n\n"
        "**Советы:**\n"
        "- Фото должно быть четким\n"
        "- Стейк должен быть виден\n"
        "- Избегайте теней"
    )

async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /status"""
    try:
        import requests
        response = requests.get(f'{BASE_URL}/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            await update.message.reply_text(
                "**Сервер онлайн!**\n\n"
                f"Устройство: {data['device']}\n"
                f"Модели:\n"
                f"  {'[OK]' if data['models']['yolo'] else '[ERR]'} YOLO\n"
                f"  {'[OK]' if data['models']['steak'] else '[ERR]'} Steak\n"
                f"  {'[OK]' if data['models']['marbling'] else '[ERR]'} Marbling"
            )
        else:
            await update.message.reply_text(f"Ошибка сервера: {response.status_code}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка подключения: {str(e)}")

# Main
def main():
    """Запуск бота"""
    global server_process

    print("="*60)
    print("ТЕЛЕГРАМ-БОТ ДЛЯ КЛАССИФИКАЦИИ СТЕЙКОВ")
    print("="*60)
    print(f"Токен бота: {BOT_TOKEN[:10]}...")
    print(f"API URL: {API_URL}")
    print("="*60)

    # Запуск сервера
    if not start_fastapi_server():
        print("\nНе удалось запустить сервер!")
        sys.exit(1)

    print("\nЗапуск телеграм-бота...")

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(MessageHandler(filters.COMMAND and filters.Command(['start']), handle_start))
    application.add_handler(MessageHandler(filters.COMMAND and filters.Command(['help']), handle_help))
    application.add_handler(MessageHandler(filters.COMMAND and filters.Command(['status']), handle_status))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен!")
    print("Нажмите Ctrl+C для остановки\n")

    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\nОстановка...")
    finally:
        if server_process:
            print("Остановка сервера...")
            server_process.terminate()
        print("Все процессы остановлены")

if __name__ == '__main__':
    main()
