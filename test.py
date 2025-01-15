import os
import numpy as np
import pydicom
from PIL import Image
import tensorflow as tf

def process_dicom_file(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        if 'PixelData' not in ds:
            print(f"Файл {dicom_path} не содержит изображения.")
            return None
        image = ds.pixel_array
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image * 255).astype(np.uint8)
        image_resized = Image.fromarray(image).resize((512, 512))
        return np.array(image_resized)
    except Exception as e:
        print(f"Ошибка при обработке {dicom_path}: {e}")
        return None


def process_folder(folder_path):
    images = []
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith("IM"):
                dicom_path = os.path.join(root, file)
                processed_image = process_dicom_file(dicom_path)
                if processed_image is not None:
                    images.append(processed_image)
                    file_paths.append(dicom_path)
    return np.array(images), file_paths


model_health = tf.keras.models.load_model('model_health_31-12-2024.h5')
model_risk = tf.keras.models.load_model('model_risk_new_31-12-2024.h5')

test_folder_path = "D:/datasets/Dataset/Patient/Patient_63/SE0001"

images, file_paths = process_folder(test_folder_path)

if len(images) > 0:
    images = images[..., np.newaxis]

    health_predictions = model_health.predict(images)
    health_classes = np.argmax(health_predictions, axis=1)

    risk_predictions = model_risk.predict(images)
    risk_classes = np.argmax(risk_predictions, axis=1)

    health_issues_detected = np.any(health_classes == 1)
    risk_issues_detected = np.any(risk_classes == 1)

    print("\nРезультаты предсказаний по каждому слою:")
    for i, file_path in enumerate(file_paths):
        print(f"Файл: {file_path}")
        print(f"  Health Prediction: {health_predictions[i]}, Predicted Class: {health_classes[i]}")
        print(f"  Risk Prediction: {risk_predictions[i]}, Predicted Class: {risk_classes[i]}")

    print("\nИтоговый отчёт:")
    print(f"Обработано изображений: {len(images)}")
    total_issues = np.sum(health_classes == 1) + np.sum(risk_classes == 1)
    print(f"Обнаружено проблемных изображений: {total_issues}")

    if health_issues_detected or risk_issues_detected:
        print("Заключение: Проблемы обнаружены.")
    else:
        print("Заключение: Проблем не выявлено.")
else:
    print("Изображения для предсказаний не найдены.")
