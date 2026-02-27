
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import tempfile
import shutil
import io
from typing import List, Tuple
import signal
import timm

# ====================== CONFIG ======================
YOLO_MODEL = 'best_steak_detector.pt'
STEAK_MODEL = 'dinov2_steak_model.pth'
MARB_MODEL = 'marbling_model_balanced.pth'
CONF_THRESHOLD = 0.3
PORT = 8000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ====================== LOAD MODELS ======================
print("Loading models...")

# Проверка наличия моделей
for model_path, name in [(YOLO_MODEL, 'YOLO'), (STEAK_MODEL, 'Steak'), (MARB_MODEL, 'Marbling')]:
    if not os.path.exists(model_path):
        print(f"⚠️  {name} model not found: {model_path}")
    else:
        print(f"✅ {name} model found: {model_path}")

# Загрузка YOLO
yolo = YOLO(YOLO_MODEL)

def load_dinov2_classifier(path, device):
    """Загрузка DINOv2 классификатора (timm или transformers model)"""
    if not os.path.exists(path):
        print(f"⚠️  Model not found: {path}")
        return None, None, None

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model_name = checkpoint.get('model_name', 'facebook/dinov2-base')
        classes = checkpoint.get('classes', ['class1', 'class2'])
        img_size = checkpoint.get('img_size', 224)

        print(f"\n✅ Loading: {path}")
        print(f"   Model: {model_name}")
        print(f"   Classes: {classes}")
        print(f"   Img size: {img_size}")

        # Определяем тип модели по названию
        if model_name.startswith('facebook/dinov2'):
            # Загружаем через transformers
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=len(classes),
                id2label={i: c for i, c in enumerate(classes)}
            )
            model.to(device)
            model.eval()
            # Для transformers используем processor вместо transform
            print(f"   Loaded via transformers")
            return model, processor, {
                'classes': classes,
                'img_size': img_size,
                'model_name': model_name,
                'type': 'transformers',
                'processor': processor
            }
        else:
            # Загружаем через timm
            model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            processor = None
            print(f"   Loaded via timm")
            return model, processor, {
                'classes': classes,
                'img_size': img_size,
                'model_name': model_name,
                'type': 'timm'
            }

    except Exception as e:
        print(f"Error loading {path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Загрузка DINOv2 классификаторов
steak_model, steak_processor, steak_meta = load_dinov2_classifier(STEAK_MODEL, device)
marb_model, marb_processor, marb_meta = load_dinov2_classifier(MARB_MODEL, device)

# ====================== TRANSFORMS ======================
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ====================== CLASSIFICATION ======================
def classify_image(model, meta, image, device):
    """Классификация PIL изображения (timm или transformers модель)"""
    if model is None or meta is None:
        return 'N/A', 0.0

    model_type = meta.get('type', 'timm')
    classes = meta.get('classes', ['class'])

    if model_type == 'transformers':
        # Используем processor для transformers
        processor = meta.get('processor') or meta.get('model')  # model is actually processor here
        inputs = processor(images=image, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        class_name = classes[idx.item()]
        confidence = conf.item()
    else:
        # timm модель
        img_size = meta.get('img_size', 224)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, dim=1)

        class_name = classes[idx.item()]
        confidence = conf.item()

    return class_name, confidence

# ====================== VISUALIZATION ======================
def visualize_prediction(image_bytes, conf_threshold=0.5):
    """Визуализация предсказаний с красивыми тенями"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        image_bgr = np.array(img)
        h_img, w_img = image_bgr.shape[:2]
        scale_factor = max(h_img, w_img) / 1200
        
        # YOLO детекция
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'temp.jpg')
        img.save(temp_path)
        
        results = yolo.predict(temp_path, conf=conf_threshold, verbose=False)[0]
        shutil.rmtree(temp_dir)
        
        if results.boxes is None or len(results.boxes) == 0:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            return img_bytes, {'error': 'No steaks detected'}
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)
        scores = results.boxes.conf.cpu().numpy()
        
        # Вырезаем стейки для классификации
        crops, crop_indices = [], []
        for i, (box, sc) in enumerate(zip(boxes, scores)):
            if sc < conf_threshold:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, x1), min(w_img, x2)
            y1, y2 = max(0, y1), min(h_img, y2)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_indices.append(i)
        
        # Классификация через timm
        steak_preds = []
        marb_results = []

        for crop in crops:
            crop_pil = Image.fromarray(crop)

            # Steak classification
            s_label, s_conf = classify_image(steak_model, steak_meta, crop_pil, device)
            steak_preds.append((s_label, s_conf))

            # Marbling classification
            if s_label.lower() != 'alternative' and marb_model is not None:
                m_label, m_conf = classify_image(marb_model, marb_meta, crop_pil, device)
            else:
                m_label, m_conf = 'N/A', 0.0
            marb_results.append((m_label, m_conf))
        
        # Рисуем на изображении
        pil_image = Image.fromarray(image_bgr)
        
        # Слои для визуализации
        shadow_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        line_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        bg_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        text_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        
        shadow_draw = ImageDraw.Draw(shadow_layer)
        line_draw = ImageDraw.Draw(line_layer)
        bg_draw = ImageDraw.Draw(bg_layer)
        text_draw = ImageDraw.Draw(text_layer)
        
        # Шрифт
        font_size = max(12, int(14 * scale_factor))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for drawn, i in enumerate(crop_indices):
            box = boxes[i]
            lab = labels[i]
            sc = scores[i]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            
            # Данные для подписи
            s_label, s_conf = steak_preds[drawn]
            m_label, m_conf = marb_results[drawn]
            det_name = yolo.names.get(int(lab), f'class {lab}')
            lines = [
                f"{det_name}: {sc:.2f}",
                f"Steak: {s_label} ({s_conf:.2f})",
                f"Marbling: {m_label} ({m_conf:.2f})"
            ]
            
            # Расчет размеров текста
            temp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
            text_bboxes = [temp_draw.textbbox((0, 0), line, font=font) for line in lines]
            max_width = max(b[2] - b[0] for b in text_bboxes)
            line_height = text_bboxes[0][3] - text_bboxes[0][1]
            line_spacing = int(8 * scale_factor)
            total_height = line_height * len(lines) + line_spacing * (len(lines) - 1)
            
            # Параметры фона
            bg_padding = int(10 * scale_factor)
            bg_width = max_width + bg_padding * 2
            bg_height = total_height + bg_padding * 2
            
            # Позиция текста
            text_x = x2 + bg_padding
            text_y_start = y1 - total_height - bg_padding
            
            bg_x1 = text_x - bg_padding
            bg_y1 = text_y_start - bg_padding
            bg_x2 = bg_x1 + bg_width
            bg_y2 = bg_y1 + bg_height
            
            # Проверка границ
            if bg_x2 > w_img or bg_y1 < 0 or bg_y2 > h_img:
                bg_x1 = x2 + bg_padding
                bg_x2 = bg_x1 + bg_width
                text_x = bg_x1 + bg_padding
                text_y_start = y2 + bg_padding
                bg_y1 = text_y_start - bg_padding
                bg_y2 = bg_y1 + bg_height
                
                if bg_x2 > w_img:
                    shift = bg_x2 - w_img
                    bg_x1 -= shift
                    bg_x2 -= shift
                    text_x -= shift
                
                if bg_y2 > h_img:
                    shift = bg_y2 - h_img
                    bg_y1 -= shift
                    bg_y2 -= shift
                    text_y_start -= shift
                
                top_x, top_y = int(x2), int(y2)
            else:
                top_x, top_y = int(x2), int(y1)
            
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            line_end_x = bg_x2
            
            # Рисуем тень
            shadow_offset = int(1.5 * scale_factor)
            for layer in range(4, 0, -1):
                alpha = int(40 * (layer / 4))
                width = int(1 * scale_factor) + layer
                shadow_draw.line(
                    [(center_x + shadow_offset, center_y + shadow_offset),
                     (top_x + shadow_offset, top_y + shadow_offset),
                     (line_end_x + shadow_offset, top_y + shadow_offset)],
                    fill=(0, 0, 0, alpha),
                    width=width,
                    joint='curve'
                )
            
            # Размытая тень
            soft_shadow = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            soft_draw = ImageDraw.Draw(soft_shadow)
            soft_draw.line(
                [(center_x + shadow_offset, center_y + shadow_offset),
                 (top_x + shadow_offset, top_y + shadow_offset),
                 (line_end_x + shadow_offset, top_y + shadow_offset)],
                fill=(0, 0, 0, 30),
                width=int(6 * scale_factor),
                joint='curve'
            )
            soft_shadow = soft_shadow.filter(ImageFilter.GaussianBlur(radius=int(2 * scale_factor)))
            shadow_layer = Image.alpha_composite(shadow_layer, soft_shadow)
            
            # Основная линия
            line_draw.line(
                [(center_x, center_y), (top_x, top_y), (line_end_x, top_y)],
                fill=(0, 0, 0, 255),
                width=max(1, int(1 * scale_factor)),
                joint='curve'
            )
            
            # Фон текста
            radius = int(12 * scale_factor)
            bg_draw.rounded_rectangle(
                [(bg_x1, bg_y1), (bg_x2, bg_y2)],
                radius=radius,
                fill=(30, 30, 30, 180),
                outline=(255, 255, 255, 100),
                width=int(1 * scale_factor)
            )
            
            # Текст
            current_y = text_y_start
            for line in lines:
                text_draw.text((text_x + 1, current_y + 1), line, fill=(0, 0, 0, 200), font=font)
                text_draw.text((text_x, current_y), line, fill=(255, 255, 255, 255), font=font)
                current_y += line_height + line_spacing
        
        # Размытие фона
        bg_layer_blurred = bg_layer.filter(ImageFilter.GaussianBlur(radius=int(1.2 * scale_factor)))
        
        # Собираем результат
        result = pil_image.convert('RGBA')
        result = Image.alpha_composite(result, shadow_layer)
        result = Image.alpha_composite(result, line_layer)
        result = Image.alpha_composite(result, bg_layer_blurred)
        result = Image.alpha_composite(result, text_layer)
        
        # Сохраняем в bytes
        img_bytes = io.BytesIO()
        result.convert('RGB').save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        
        prediction_data = {
            'detected': len(boxes),
            'boxes': boxes.tolist(),
            'steak_predictions': steak_preds,
            'marbling_predictions': marb_results,
            'labels': [yolo.names.get(int(lab), f'class {lab}') for lab in labels]
        }
        
        return img_bytes, prediction_data
        
    except Exception as e:
        print(f"Error in visualize_prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================== FASTAPI APP ======================
app = FastAPI(
    title="Steak Classification API (DINOv2)",
    description="API для классификации стейков (вид + мраморность)",
    version="2.0.0"
)

@app.get("/", response_model=dict)
async def root():
    return {
        'name': 'Steak Classification API',
        'version': '2.0.0',
        'docs': '/docs',
        'health': '/health'
    }

@app.get("/health", response_model=dict)
async def health_check():
    return {
        'status': 'ok',
        'device': str(device),
        'models': {
            'yolo': yolo is not None,
            'steak': steak_model is not None,
            'marbling': marb_model is not None
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf_threshold: float = CONF_THRESHOLD):
    """
    Классификация стейка с визуализацией
    
    - **file**: Изображение стейка (jpg, png, webp)
    - **conf_threshold**: Порог уверенности детекции (0.0-1.0)
    
    Returns: JPEG изображение с bounding boxes и подписями
    """
    if file is None:
        raise HTTPException(status_code=400, detail='No file provided')
    
    try:
        image_bytes = await file.read()
        img_bytes, prediction_data = visualize_prediction(image_bytes, conf_threshold)
        
        if 'error' in prediction_data:
            return JSONResponse(
                status_code=404,
                content={'success': False, 'message': prediction_data['error']}
            )
        
        return StreamingResponse(img_bytes, media_type="image/jpeg")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Информация о загруженных моделях"""
    return {
        'yolo': {
            'path': YOLO_MODEL,
            'loaded': yolo is not None
        },
        'steak': {
            'path': STEAK_MODEL,
            'loaded': steak_model is not None,
            'classes': steak_meta.get('classes', []) if steak_meta else []
        },
        'marbling': {
            'path': MARB_MODEL,
            'loaded': marb_model is not None,
            'classes': marb_meta.get('classes', []) if marb_meta else []
        }
    }

# ====================== MAIN ======================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("STEAK CLASSIFICATION API (DINOv2)")
    print("="*60)
    print(f"Server: http://localhost:{PORT}")
    print(f"Docs: http://localhost:{PORT}/docs")
    print(f"Health: http://localhost:{PORT}/health")
    print(f"Predict: POST http://localhost:{PORT}/predict")
    print("="*60)
    
    try:
        uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='info')
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
