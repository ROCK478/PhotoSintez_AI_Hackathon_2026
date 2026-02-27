import cv2
import numpy as np
from ultralytics import YOLO

# пока закомментировано, если модели нет
# model = YOLO("model/best.pt")

def run_inference(image_path):
    """
    Принимает путь к изображению.
    Возвращает обработанное изображение и метрики.
    Пока используется mock, позже заменим на YOLO.
    """

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # ---- MOCK инференс (пока модели нет) ----
    overlay = image.copy()
    cv2.rectangle(overlay, (width//4, height//4), (width//2, height//2), (0,255,0), 3)
    metrics = {
        "root_length": 10.0,
        "stem_length": 15.0,
        "leaf_area": 20.0
    }

    # ---- Настройка для будущей YOLO ----
    # if model is not None:
    #     results = model(image_path)
    #     result = results[0]
    #     overlay = result.plot()
    #     metrics = calculate_metrics_from_yolo(result)

    return overlay, metrics

# пример функции для будущих метрик
def calculate_metrics_from_yolo(result):
    """
    Здесь будет ваш код для подсчета длины корня, стебля
    и площади листьев на основе YOLO сегментации.
    """
    # Пока заглушка
    return {
        "root_length": 0.0,
        "stem_length": 0.0,
        "leaf_area": 0.0
    }