import os
import json
import numpy as np
from PIL import Image, ImageDraw
import shutil
import random

def json_to_mask(json_path, output_mask_dir, output_color_mask_dir, class_mapping, class_colors):
    """Преобразует JSON из LabelMe в маску для сегментации и цветную визуализацию"""
    
    # Загрузка JSON-файла
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Создание пустой маски (одноканальная для тренировки)
    image_size = (data['imageWidth'], data['imageHeight'])
    mask = Image.new('L', image_size, 0)  # 'L' - 8-битная grayscale маска
    draw = ImageDraw.Draw(mask)
    
    # Создание цветной маски для визуализации
    color_mask = Image.new('RGB', image_size, (0, 0, 0))  # Черный фон
    color_draw = ImageDraw.Draw(color_mask)
    
    # Отрисовка каждого объекта на масках
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        # Пропускаем если метки нет в class_mapping
        if label not in class_mapping:
            print(f"Предупреждение: метка '{label}' не найдена в class_mapping, пропускаем")
            continue
            
        class_id = class_mapping[label]
        color = class_colors[class_id]
        
        # Конвертируем точки в кортежи
        polygon = [tuple(point) for point in points]
        
        # Рисуем полигон на grayscale маске с class_id
        draw.polygon(polygon, fill=class_id)
        
        # Рисуем полигон на цветной маске с соответствующим цветом
        color_draw.polygon(polygon, fill=color)
    
    # Сохраняем маски
    image_name = os.path.splitext(os.path.basename(data['imagePath']))[0]
    
    # Сохраняем grayscale маску (для тренировки)
    mask_path = os.path.join(output_mask_dir, f"{image_name}.png")
    mask.save(mask_path)
    
    # Сохраняем цветную маску (для визуализации)
    color_mask_path = os.path.join(output_color_mask_dir, f"{image_name}_color.png")
    color_mask.save(color_mask_path)
    
    print(f"Созданы маски: {mask_path} и {color_mask_path}")

def generate_class_colors(num_classes):
    """Генерирует случайные цвета для каждого класса"""
    colors = [(0, 0, 0)]  # Фон - черный
    for i in range(1, num_classes):
        # Генерируем яркие цвета для лучшей визуализации
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        colors.append((r, g, b))
    return colors

def process_labelme_folder(input_dir, output_base_dir, class_mapping):
    """Обрабатывает все файлы в папке LabelMe"""
    
    # Создаем выходные папки
    images_dir = os.path.join(output_base_dir, 'images')
    masks_dir = os.path.join(output_base_dir, 'masks')
    color_masks_dir = os.path.join(output_base_dir, 'color_masks')  # Для цветных масок
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(color_masks_dir, exist_ok=True)
    
    # Генерируем цвета для классов
    num_classes = len(class_mapping)
    class_colors = generate_class_colors(num_classes)
    
    # Счетчики
    processed_count = 0
    error_count = 0
    
    # Обрабатываем все JSON-файлы
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(input_dir, filename)
            
            try:
                # Создаем маски
                json_to_mask(json_path, masks_dir, color_masks_dir, class_mapping, class_colors)
                
                # Копируем изображение
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_filename = data['imagePath']
                
                # Если путь относительный, ищем в той же папке
                if not os.path.isabs(image_filename):
                    image_path = os.path.join(input_dir, os.path.basename(image_filename))
                else:
                    image_path = image_filename
                
                if os.path.exists(image_path):
                    output_image_path = os.path.join(images_dir, os.path.basename(image_path))
                    shutil.copy2(image_path, output_image_path)
                    print(f"Скопировано изображение: {output_image_path}")
                else:
                    print(f"Предупреждение: изображение {image_path} не найдено")
                
                processed_count += 1
                
            except Exception as e:
                print(f"Ошибка при обработке {json_path}: {str(e)}")
                error_count += 1
    
    print(f"\nОбработка завершена!")
    print(f"Успешно обработано: {processed_count} файлов")
    print(f"Ошибок: {error_count}")
    
    # Сохраняем информацию о цветах классов
    with open(os.path.join(output_base_dir, 'class_colors.txt'), 'w', encoding='utf-8') as f:
        f.write("Class ID, Class Name, Color (R,G,B)\n")
        for class_name, class_id in class_mapping.items():
            color = class_colors[class_id]
            f.write(f"{class_id}, {class_name}, {color}\n")

def auto_detect_classes(input_dir):
    """Автоматически определяет все классы из JSON-файлов"""
    all_labels = set()
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(input_dir, filename)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for shape in data['shapes']:
                    all_labels.add(shape['label'])
            except Exception as e:
                print(f"Ошибка при чтении {json_path}: {str(e)}")
    
    # Создаем mapping (фон всегда 0)
    class_mapping = {"background": 0}
    for i, label in enumerate(sorted(all_labels), 1):
        class_mapping[label] = i
    
    return class_mapping

# Использование
if __name__ == "__main__":
    # Пути к данным
    input_directory = "all"  # папка с JSON и изображениями
    output_directory = "datasets"  # выходная папка для датасета
    
    # Автодетект классов и запуск обработки
    class_mapping = auto_detect_classes(input_directory)
    print("Обнаруженные классы:", class_mapping)
    
    # Запуск обработки
    process_labelme_folder(input_directory, output_directory, class_mapping)
    
    # Сохраняем файл с метками классов
    with open(os.path.join(output_directory, 'labels.txt'), 'w', encoding='utf-8') as f:
        for class_name, class_id in class_mapping.items():
            f.write(f"{class_id} {class_name}\n")
    
    print(f"Файл с метками сохранен: {os.path.join(output_directory, 'labels.txt')}")
    print(f"Файл с цветами классов сохранен: {os.path.join(output_directory, 'class_colors.txt')}")