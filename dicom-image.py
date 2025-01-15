import pydicom
import os
import numpy as np

base_path = "D:/datasets/Dataset/all/"  # Путь к основной папке с данными
# Функция для загрузки изображений DICOM из папки пациента
countImages = 0
def load_dicom_images(patient_folder):
    images = []
    for file in os.listdir(patient_folder):
        if file.startswith("IM"):
            filepath = os.path.join(patient_folder, file)
            dicom_data = pydicom.dcmread(filepath)
            images.append(dicom_data.pixel_array)  # Загружаем пиксельные данные
    return images

# Список всех папок с пациентами
patient_folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

# Загрузка изображений для всех пациентов
for patient in patient_folders:
    patient_folder = os.path.join(base_path, patient, "SE0001")
    images = load_dicom_images(patient_folder)
    print(f"Загружено {len(images)} изображений для {patient}")

    countImages = countImages + len(images)


print(f"Количество изображений для пациента: {countImages}")
print(f"Размер первого изображения: {images[0].shape}")
