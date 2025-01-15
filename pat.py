import os
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.ndimage import label
from tqdm import tqdm  # Добавляем tqdm для отображения прогресса

# Параметры, заданные в начале кода
metadata_path = "D:/datasets/updated_metadata.xlsx"
dataset_folder = "D:/datasets/Dataset/all/"
pixel_size = (0.5, 0.5)  # Размер пикселя в мм
slice_thickness = 1  # Толщина среза в мм


# Функция для загрузки DICOM-изображения и конвертации в Hounsfield Units (HU)
def load_dicom_hu(dicom_path):
    """
    Загружает DICOM файл и конвертирует изображение в Hounsfield Units (HU).
    """
    ds = pydicom.dcmread(dicom_path)

    if 'PixelData' not in ds:
        return None

    image = ds.pixel_array.astype(np.int16)

    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        image = image * ds.RescaleSlope + ds.RescaleIntercept

    return image

# Функция для выделения кальциевых отложений на изображении
def detect_calcium(image, min_hu=130, max_hu=600, pixel_size=(1, 1), slice_thickness=1):
    calcium_mask = (image >= min_hu) & (image <= max_hu)
    labeled_array, num_features = label(calcium_mask)

    calcium_area = np.sum(calcium_mask) * np.prod(pixel_size)
    calcium_volume = calcium_area * slice_thickness

    return calcium_area, calcium_volume, num_features

# Основная функция для обработки всех DICOM файлов пациента
def process_patient_ct(folder_path, pixel_size=(0.5, 0.5), slice_thickness=1):
    total_area = 0
    total_volume = 0

    for root, dirs, files in os.walk(folder_path):
        for file in sorted(files):
            if file.startswith("IM"):
                dicom_path = os.path.join(root, file)
                image_hu = load_dicom_hu(dicom_path)

                if image_hu is not None:
                    calcium_area, calcium_volume, num_features = detect_calcium(
                        image_hu, pixel_size=pixel_size, slice_thickness=slice_thickness
                    )
                    total_area += calcium_area
                    total_volume += calcium_volume

    return total_area, total_volume

# Чтение таблицы данных пациентов
metadata = pd.read_excel(metadata_path)
print(metadata.columns)


# Список для хранения входных данных и целевых значений
X = []  # Входные данные
y = []  # Целевые значения (например, 1 для ИБС, 0 для здоровых)

# Обрабатываем всех пациентов в папке
for patient_folder in tqdm(os.listdir(dataset_folder), desc="Обработка пациентов"):  # Добавляем tqdm для отображения прогресса
    patient_path = os.path.join(dataset_folder, patient_folder)

    # Получаем площадь и объем кальциевых отложений для пациента
    calcium_area, calcium_volume = process_patient_ct(patient_path, pixel_size, slice_thickness)

    # Получаем метки для пациента из таблицы (например, на основе столбца 'Health')
    patient_number = patient_folder
    calcium_score = metadata.loc[metadata["Number_Patient"] == patient_number, "Calcium_score"].values
    health_status = metadata.loc[metadata["Number_Patient"] == patient_number, "Health"].values

    if health_status.size > 0:
        diagnosis_label = health_status[0]  # Используем значение 0 или 1 из столбца 'Health'

        # Добавляем данные пациента в список
        X.append([calcium_area, calcium_volume, calcium_score[0]])
        y.append(diagnosis_label)

# Преобразуем X и y в numpy массивы
X = np.array(X)
y = np.array(y)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель случайного леса
clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)  # Устанавливаем verbose для вывода процесса обучения
clf.fit(X_train, y_train)

# Предсказываем результаты на тестовой выборке
y_pred = clf.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Выводим результаты
print(f"Точность модели: {accuracy:.4f}")
print("Отчет по классификации:\n", report)
