import cv2
import traceback
import numpy as np

def run_inference(image_path, model):
    try:
        img = cv2.imread(image_path)

        if img is None:
            raise Exception("OpenCV не может прочитать изображение")

        results = model.predict(
            source=image_path,
            task="segment",
            conf=0.1,
            save=False,
            verbose=True
        )

        if len(results) == 0:
            raise Exception("YOLO не вернул результатов")

        result = results[0]

        print("Boxes:", result.boxes)
        print("Masks:", result.masks)

        # рисуем результат
        annotated = result.plot(boxes=False, conf=False)
        

        if annotated is None:
            raise Exception("result.plot() вернул None")

        metrics = {
            "test": "ok"
        }

        return annotated, metrics

    except Exception as e:

        print("\n=== INFERENCE ERROR ===")
        traceback.print_exc()
        print("=======================\n")

        raise e