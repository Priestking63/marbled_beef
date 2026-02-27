# Steak Classification System

Система для автоматической классификации стейков по типу и мраморности с использованием компьютерного зрения и глубокого обучения.

## Возможности

- **Детекция стейков** на изображении с помощью YOLO
- **Классификация типа стейка**: рибай (ribeye), филе (filet), стриплойн (strip)
- **Определение мраморности**: choice, select, prine
- **Визуализация результатов**: bounding boxes с подписями
- **Telegram-бот** для удобной отправки фото
- **FastAPI сервер** для обработки запросов

## Архитектура

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Telegram Bot   │────▶│  FastAPI Server  │────▶│   YOLO Detector │
│   (bot.py)      │     │    (app.py)      │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ├────▶ DINOv2 Steak Classifier
                               │
                               └────▶ Marbling Classifier
```

## Структура проекта

```
marbled_beef/
├── bot.py                      # Telegram-бот
├── app.py                      # FastAPI сервер
├── requirements.txt            # Зависимости
├── .env                        # Переменные окружения
├── .gitignore                  # Игнорируемые файлы
│
├── best_steak_detector.pt      # YOLO модель детекции
├── dinov2_steak_model.pth      # Классификатор типа стейка
├── marbling_model_balanced.pth # Классификатор мраморности
│
└── notebooks/                  # Jupyter ноутбуки для обучения
    ├── yolo_detection_optimized.ipynb
    ├── dinov2_optuna.ipynb
    └── marbling_classification_balanced.ipynb
```

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd marbled_beef
```

### 2. Создание виртуального окружения

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
BOT_TOKEN=your_telegram_bot_token_here
API_URL=http://localhost:8000/predict
```

**Получение токена Telegram:**
1. Откройте @BotFather в Telegram
2. Отправьте команду `/newbot`
3. Следуйте инструкциям
4. Скопируйте полученный токен в `.env`

## Запуск

### Быстрый старт (бот + сервер)

```bash
python bot.py
```

Эта команда запустит:
1. FastAPI сервер на `http://localhost:8000`
2. Telegram-бота для приёма фото

### Только FastAPI сервер

```bash
python app.py
```

Сервер будет доступен по адресу:
- API: `http://localhost:8000`
- Документация: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## Использование

### Через Telegram-бота

1. Запустите бота: `python bot.py`
2. Отправьте боту фото стейка
3. Получите результат с метками:
   - Тип стейка
   - Степень мраморности
   - Confidence scores

### Через API

**Запрос:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@steak_photo.jpg" \
  -F "conf_threshold=0.5"
```

**Python пример:**

```python
import requests

with open('steak.jpg', 'rb') as f:
    files = {'file': f}
    data = {'conf_threshold': 0.5}
    response = requests.post('http://localhost:8000/predict', files=files, data=data)

with open('result.jpg', 'wb') as f:
    f.write(response.content)
```

### Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Запуск бота, приветствие |
| `/help` | Справка по использованию |
| `/status` | Статус сервера и моделей |

## API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Информация об API |
| `/health` | GET | Статус сервера и моделей |
| `/predict` | POST | Классификация стейка |
| `/models` | GET | Информация о загруженных моделях |

## Модели

### YOLO Detector
- **Файл**: `best_steak_detector.pt`
- **Задача**: Детекция стейков на изображении
- **Параметры**: настраиваются в `best_yolo_params.yaml`

### DINOv2 Steak Classifier
- **Файл**: `steak_model_balanced.pth`
- **Задача**: Классификация типа стейка
- **Классы**: ribeye, filet, strip

### Marbling Classifier
- **Файл**: `marbling_model_balanced.pth`
- **Задача**: Определение мраморности
- **Классы**: choice, select, prime

## Требования к изображениям

- **Формат**: JPG, PNG, WebP
- **Освещение**: Хорошее, равномерное
- **Ракурс**: Вид сверху
- **Качество**: Чёткое изображение
- **Объект**: Стейк должен быть хорошо виден

## Обучение моделей

В репозитории представлены Jupyter ноутбуки для обучения:

- `yolo_detection_optimized.ipynb` — обучение YOLO детектора
- `dinov2_optuna.ipynb` — обучение классификатора типа стейка
- `marbling_classification_balanced.ipynb` — обучение классификатора мраморности

## Зависимости

Основные библиотеки:

- **FastAPI** — веб-сервер
- **python-telegram-bot** — Telegram бот
- **torch, torchvision** — глубокое обучение
- **timm** — предобученные модели
- **transformers** — DINOv2 через Hugging Face
- **ultralytics** — YOLO
- **Pillow, opencv-python** — обработка изображений

Полный список в `requirements.txt`.

## Структура ответов API

### GET /health

```json
{
  "status": "ok",
  "device": "cuda",
  "models": {
    "yolo": true,
    "steak": true,
    "marbling": true
  }
}
```

### POST /predict

Возвращает изображение с визуализацией (JPEG) или ошибку:

```json
{
  "success": false,
  "message": "No steaks detected"
}
```

## Устранение неполадок

### Ошибка "BOT_TOKEN not configured"

- Проверьте, что токен указан в `.env`
- Убедитесь, что нет лишних пробелов
- Перезапустите бота

### Ошибка подключения к серверу

- Убедитесь, что сервер запущен
- Проверьте порт 8000
- Проверьте `/health` endpoint

### Модель не загружается

- Проверьте наличие файлов `.pth` и `.pt`
- Убедитесь, что зависимости установлены
- Проверьте пути к моделям в `app.py`

## Лицензия

MIT

## Контакты

Для вопросов и предложений создайте issue в репозитории.
