import argparse
import os
import cv2
import traceback
import numpy as np
from skimage.morphology import skeletonize
from ultralytics import YOLO

PIXELS_PER_CM = 95.0
CM_PER_PIXEL = 1.0 / PIXELS_PER_CM
CM2_PER_PIXEL2 = CM_PER_PIXEL ** 2

def mask_area_pixels(mask):
    return np.count_nonzero(mask)

def mask_length_pixels(mask):
    mask_bool = mask.astype(bool)
    skeleton = skeletonize(mask_bool)
    return np.count_nonzero(skeleton)

def run_inference_folder_with_metrics(image_dir, model, output_dir, metrics_file="metrics.txt"):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    
    if not image_files:
        print(f"В папке {image_dir} нет изображений")
        return

    metrics_path = os.path.join(output_dir, metrics_file)
    with open(metrics_path, "w") as f_metrics:
        for image_name in image_files:
            image_path = os.path.join(image_dir, image_name)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Невозможно прочитать изображение: {image_path}")
                    continue

                results = model.predict(
                    source=image_path,
                    task="segment",
                    conf=0.6,
                    save=False,
                    verbose=False
                )

                if len(results) == 0 or results[0].masks is None:
                    print(f"YOLO не вернул масок для {image_name}")
                    continue

                result = results[0]
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                class_names = model.names

                root_length_px = root_area_px = 0
                stem_length_px = stem_area_px = 0
                leaf_area_px = 0

                annotated_img = result.plot(conf=False)

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

                metrics = {
                    "root_length_cm": round(root_length_px * CM_PER_PIXEL, 3),
                    "root_area_cm2": round(root_area_px * CM2_PER_PIXEL2, 3),
                    "stem_length_cm": round(stem_length_px * CM_PER_PIXEL, 3),
                    "stem_area_cm2": round(stem_area_px * CM2_PER_PIXEL2, 3),
                    "leaf_area_cm2": round(leaf_area_px * CM2_PER_PIXEL2, 3)
                }

                # Записываем метрики в файл
                metrics_line = f"{image_name} - " + ", ".join(f"{k}: {v}" for k, v in metrics.items())
                f_metrics.write(metrics_line + "\n")
                print(metrics_line)

                # Сохраняем аннотированное изображение
                output_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_path, annotated_img)

            except Exception as e:
                print(f"Ошибка при обработке {image_name}: {e}")
                traceback.print_exc()


def train_model(dataset_path):
    model = YOLO("yolo11m-seg.pt")  # фиксированная модель
    train_args = dict(
        data=dataset_path,
        epochs=400,
        imgsz=960,
        batch=2,
        device=0,  # фиксированное устройство
        workers=4,
        optimizer="AdamW",
        lr0=5e-4,
        cos_lr=True,
        weight_decay=5e-4,
        overlap_mask=True,
        mask_ratio=1,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        amp=True
    )
    model.train(**train_args)

def main():
    parser = argparse.ArgumentParser(description="YOLO Console Tool: Train and Inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train YOLO model")
    parser_train.add_argument("--dataset", type=str, required=True, help="Path to dataset yaml")

    parser_infer = subparsers.add_parser("inference", help="Run inference on an image")
    parser_infer.add_argument("--model", type=str, default="model/best.pt", help="Path to trained YOLO model (e.g. model/best.pt)")
    parser_infer.add_argument("--image", type=str, required=True, help="Path to input image")
    parser_infer.add_argument("--output", type=str, default="output", help="Directory to save annotated image")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.dataset)
    elif args.command == "inference":
        model = YOLO(args.model)
        run_inference_folder_with_metrics(args.image, model, args.output)

if __name__ == "__main__":
    main()