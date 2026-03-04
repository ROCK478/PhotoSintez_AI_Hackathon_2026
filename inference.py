import cv2
import traceback
import numpy as np
from skimage.morphology import skeletonize

PIXELS_PER_CM = 95.0
CM_PER_PIXEL = 1.0 / PIXELS_PER_CM
CM2_PER_PIXEL2 = CM_PER_PIXEL ** 2

def mask_area_pixels(mask):
    return np.count_nonzero(mask)

def mask_length_pixels(mask):

    mask_bool = mask.astype(bool)
    skeleton = skeletonize(mask_bool)
    length_pixels = np.count_nonzero(skeleton)
    return length_pixels

def run_inference(image_path, model):
    try:
        img = cv2.imread(image_path)

        if img is None:
            raise Exception("OpenCV не может прочитать изображение")

        results = model.predict(
            source=image_path,
            task="segment",
            conf=0.6,
            save=False,
            verbose=False
        )

        if len(results) == 0:
            raise Exception("YOLO не вернул результатов")

        result = results[0]

        if result.masks is None:
            raise Exception("Нет масок")

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        class_names = model.names

        root_length_px = 0
        root_area_px = 0
        stem_length_px = 0
        stem_area_px = 0
        leaf_area_px = 0

        for i, mask in enumerate(masks):

            class_id = classes[i]
            class_name = class_names[class_id]

            area_px = mask_area_pixels(mask)

            if class_name == "root":

                length_px = mask_length_pixels(mask)

                root_length_px += length_px
                root_area_px += area_px


            elif class_name == "stem":

                length_px = mask_length_pixels(mask)

                stem_length_px += length_px
                stem_area_px += area_px


            elif class_name == "leaf":

                leaf_area_px += area_px
                print(f"Leaf area pixels: {area_px}")

        root_length_cm = root_length_px * CM_PER_PIXEL
        root_area_cm2 = root_area_px * CM2_PER_PIXEL2
        stem_length_cm = stem_length_px * CM_PER_PIXEL
        stem_area_cm2 = stem_area_px * CM2_PER_PIXEL2
        leaf_area_cm2 = leaf_area_px * CM2_PER_PIXEL2
        print(f"Leaf area cm2: {leaf_area_cm2}")


        metrics = {
            "root_length_cm": float(round(root_length_cm, 3)),
            "root_area_cm2": float(round(root_area_cm2, 3)),

            "stem_length_cm": float(round(stem_length_cm, 3)),
            "stem_area_cm2": float(round(stem_area_cm2, 3)),
            "leaf_area_cm2": float(round(leaf_area_cm2, 3))
        }

        annotated = result.plot(conf=False)
        return annotated, metrics


    except Exception as e:

        print("\n=== INFERENCE ERROR ===")
        traceback.print_exc()
        print("=======================\n")

        raise e