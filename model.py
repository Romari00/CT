import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# Настройка GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Используется GPU:", physical_devices[0])
else:
    print("GPU не обнаружен. Используется CPU.")

# Функция обработки DICOM-файла
def process_dicom_file(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        if 'PixelData' not in ds:
            print(f"Файл {dicom_path} не содержит изображения.")
            return None
        image = ds.pixel_array
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Нормализация в диапазон [0, 1]
        image = (image * 255).astype(np.uint8)  # Преобразование в диапазон [0, 255]
        image_resized = Image.fromarray(image).resize((512, 512))  # Изменение размера до 512x512
        return np.array(image_resized)
    except Exception as e:
        print(f"Ошибка при обработке {dicom_path}: {e}")
        return None

# Функция обработки папки с метками
def process_folder_with_labels(folder_path, labels_dict):
    images = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith("IM"):
                file_path = os.path.join(root, file)
                patient_id = os.path.basename(os.path.dirname(root))  # Извлекаем ID пациента
                if patient_id in labels_dict:
                    processed_image = process_dicom_file(file_path)
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(labels_dict[patient_id])
    return np.array(images), np.array(labels)

def load_labels(metadata_path):
    metadata = pd.read_excel(metadata_path)
    metadata['Health'] = metadata['Health'].astype(int)
    metadata['Risk'] = metadata['Risk'].astype(int)
    health_dict = dict(zip(metadata['Number_Patient'], metadata['Health']))
    risk_dict = dict(zip(metadata['Number_Patient'], metadata['Risk']))
    return health_dict, risk_dict

# Загрузка данных
print("Загрузка меток Health и Risk...")
health_labels, risk_labels = load_labels("D:/datasets/normalized_patients_with_health_and_risk.xlsx")

print("Обработка изображений для Health...")
images_health, health = process_folder_with_labels("D:/datasets/Dataset/all", health_labels)
print("Обработка изображений для Risk...")
images_risk, risk = process_folder_with_labels("D:/datasets/Dataset/all", risk_labels)

# Преобразование меток в one-hot
health_one_hot = to_categorical(health, num_classes=2)
risk_one_hot = to_categorical(risk, num_classes=4)

# Разделение на обучающую и тестовую выборки
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(images_health, health_one_hot, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(images_risk, risk_one_hot, test_size=0.2, random_state=42)

# Добавление оси канала
X_train_h = X_train_h[..., np.newaxis]
X_test_h = X_test_h[..., np.newaxis]
X_train_r = X_train_r[..., np.newaxis]
X_test_r = X_test_r[..., np.newaxis]

# Создание модели Health
model_health = Sequential([
    Conv2D(32, (3, 3), activation='relu',
    kernel_regularizer=l2(0.001), input_shape=(512, 512, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 класса для Health
])

model_health.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ранняя остановка
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("Обучение модели Health...")
with tf.device('/GPU:0'):
    model_health.fit(X_train_h, y_train_h, epochs=10, batch_size=32, validation_data=(X_test_h, y_test_h), callbacks=[early_stopping])

model_health.save('model_health_31-12-2024.h5')
print("Модель Health сохранена как 'model_health_31-12-2024.h5'")

# Создание модели Risk
model_risk = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(512, 512, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 класса для Risk
])

model_risk.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Обучение модели Risk...")
with tf.device('/GPU:0'):
    model_risk.fit(X_train_r, y_train_r, epochs=10, batch_size=32, validation_data=(X_test_r, y_test_r), callbacks=[early_stopping])

model_risk.save('model_risk_new_31-12-2024.h5')
print("Модель Risk сохранена как 'model_risk_new_31-12-2024.h5'")

# Оценка моделей
print("Оценка модели Health...")
test_loss_h, test_acc_h = model_health.evaluate(X_test_h, y_test_h)
print(f"Точность модели Health на тестовых данных: {test_acc_h:.2f}")

print("Оценка модели Risk...")
test_loss_r, test_acc_r = model_risk.evaluate(X_test_r, y_test_r)
print(f"Точность модели Risk на тестовых данных: {test_acc_r:.2f}")
