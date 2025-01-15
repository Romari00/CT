import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_closing, binary_opening
import random

# Пути к данным
metadata_path = "D:/datasets/Patients Information Of Cardiac Dataset.xlsx"
test_folder_path = "D:/datasets/Dataset/all/Patient_4"

def load_dicom_hu(dicom_path):
    """Загружает DICOM и переводит в Hounsfield Units (HU)."""
    ds = pydicom.dcmread(dicom_path)
    if 'PixelData' not in ds:
        return None, None, None

    image = ds.pixel_array.astype(np.int16)

    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        image = image * ds.RescaleSlope + ds.RescaleIntercept

    pixel_spacing = ds.get("PixelSpacing", [0.5, 0.5])  # Размер пикселя в мм
    slice_thickness = ds.get("SliceThickness", 2)  # Толщина среза в мм

    return image, float(pixel_spacing[0]), float(slice_thickness)

def detect_calcium(image, min_hu=130, max_hu=150, pixel_size=(0.5, 0.5), bone_hu_threshold=400):
    """Находит кальций и считает площадь + Agatston Score, исключая кости."""
    # Исключаем области с HU, характерные для костей
    bone_mask = image > bone_hu_threshold
    image[bone_mask] = 0  # Заменяем кости на 0 (не учитываем их)

    # Находим кальций в оставшихся областях
    calcium_mask = (image >= min_hu) & (image <= max_hu)

    # Применяем морфологические операции для устранения шума (например, закрытие и открытие)
    calcium_mask = binary_closing(calcium_mask, structure=np.ones((3, 3)))
    calcium_mask = binary_opening(calcium_mask, structure=np.ones((3, 3)))

    labeled_array, num_features = label(calcium_mask)

    calcium_area = np.sum(calcium_mask) * np.prod(pixel_size)

    # Вычисление Agatston Score
    score = 0
    for label_idx in range(1, num_features + 1):
        region = (labeled_array == label_idx)
        max_hu = np.max(image[region])

        weight = 1
        if 130 <= max_hu < 200:
            weight = 1
        elif 200 <= max_hu < 300:
            weight = 2
        elif 300 <= max_hu < 400:
            weight = 3
        elif max_hu >= 400:
            weight = 4

        score += np.sum(region) * np.prod(pixel_size) * weight

    return calcium_area, score, np.mean(image[calcium_mask]) if np.any(calcium_mask) else 0

def show_calcium_mask(image, min_hu=130, max_hu=150):
    """Отображает снимок и маску кальция."""
    calcium_mask = (image >= min_hu) & (image <= max_hu)

    # Применение морфологических операций для улучшения маски
    calcium_mask = binary_closing(calcium_mask, structure=np.ones((3, 3)))
    calcium_mask = binary_opening(calcium_mask, structure=np.ones((3, 3)))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Исходное изображение")

    plt.subplot(1, 2, 2)
    plt.imshow(calcium_mask, cmap='hot')
    plt.title("Маска кальция")

    plt.show()

def process_patient_ct(folder_path):
    """Обрабатывает DICOM файлы и считает кальциевые показатели."""
    total_area = 0
    total_agatston_score = 0
    total_hu = []
    slices_count = 0
    pixel_sizes = []
    slice_thicknesses = []
    images = []  # Список для хранения изображений

    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.startswith("IM"):
                dicom_path = os.path.join(root, file)
                image_hu, pixel_size, slice_thickness = load_dicom_hu(dicom_path)

                if image_hu is not None:
                    # Анализируем кальций в изображении
                    calcium_area, agatston_score, mean_hu = detect_calcium(image_hu,
                                                                           pixel_size=(pixel_size, pixel_size))

                    # Проверка размера снимка
                    image_size = image_hu.shape[0] * image_hu.shape[1] * (pixel_size ** 2)
                    print(f"Срез {file}: размер {image_hu.shape}, площадь {image_size:.2f} мм²")

                    total_area += calcium_area
                    total_agatston_score += agatston_score
                    total_hu.append(mean_hu)
                    slices_count += 1
                    pixel_sizes.append(pixel_size)
                    slice_thicknesses.append(slice_thickness)

                    # Добавляем изображение в список
                    images.append(image_hu)

    # Проверяем, есть ли расхождения в размерах пикселей/толщине среза
    if len(set(pixel_sizes)) > 1:
        print("⚠️ Внимание! Разные размеры пикселей в снимках:", set(pixel_sizes))
    if len(set(slice_thicknesses)) > 1:
        print("⚠️ Внимание! Разная толщина срезов в снимках:", set(slice_thicknesses))

    # Визуализация первого и случайных 5 срезов
    if images:
        # Добавляем первый срез
        images_to_show = [images[0]]
        # Выбираем случайные 5 срезов, если их достаточно
        if len(images) > 5:
            images_to_show += random.sample(images[1:], 5)

        # Отображаем выбранные срезы
        plt.figure(figsize=(15, 10))
        for i, img in enumerate(images_to_show, 1):
            plt.subplot(2, 3, i)
            plt.imshow(img, cmap='gray')
            plt.title(f"Срез {i}")
        plt.show()

    total_volume = total_area * np.mean(slice_thicknesses) if slice_thicknesses else 0
    return total_area, total_volume, total_agatston_score, np.mean(total_hu) if total_hu else 0


# Загружаем данные пациентов
metadata = pd.read_excel(metadata_path, header=1)

# Извлекаем данные для конкретного пациента
calcium_area, calcium_volume, agatston_score, mean_hu = process_patient_ct(test_folder_path)
patient_number = os.path.basename(test_folder_path)

# Приводим типы к одинаковому формату
metadata["Number_Patient"] = metadata["Number_Patient"].astype(str)
patient_number = str(patient_number)

# Находим значение calcium_score в таблице
calcium_score_row = metadata.loc[metadata["Number_Patient"] == patient_number, "Calcium_score"]
if not calcium_score_row.empty:
    calcium_score_in_table = calcium_score_row.values[0]
else:
    calcium_score_in_table = None  # Если данных нет, поставим None

# Разница с таблицей
difference = None
if calcium_score_in_table is not None:
    difference = abs(agatston_score - calcium_score_in_table)

# Вывод результатов
print(f"\nПациент: {patient_number}")
print(f"Средний HU кальция: {mean_hu:.2f}")
print(f"Вычисленная площадь кальция: {calcium_area:.2f} мм²")
print(f"Вычисленный объем кальция: {calcium_volume:.2f} мм³")
print(f"Вычисленный Agatston Score: {agatston_score:.2f}")

if calcium_score_in_table is not None:
    print(f"Calcium_score в таблице: {calcium_score_in_table}")
    print(f"Разница между расчетным и табличным: {difference:.2f}")
else:
    print("Calcium_score в таблице: данные отсутствуют")
