# detect_image.py
import sys
from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Foydalanish: python detect_image.py path/to/image.jpg")
        return

    image_path = sys.argv[1]

    # YOLOv8 oldindan o‘qitilgan model (COCO 80 sinf)
    model = YOLO("yolov8n.pt")  # eng yengil model

    # Inference (aniqlash)
    results = model.predict(image_path, conf=0.25)  # conf – minimal ishonch darajasi

    # Natijalarni konsolga chiqarish
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = model.names[cls_id]
            xyxy = [float(x) for x in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
            print(f"Topildi: {label} | ishonch={conf:.2f} | bbox={xyxy}")

    print("✅ Natijalar 'runs/detect/predict*' papkaga saqlandi.")

if __name__ == "__main__":
    main()
