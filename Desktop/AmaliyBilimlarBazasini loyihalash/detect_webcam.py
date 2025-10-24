# detect_webcam.py
import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)  # 0 – asosiy kamera

    if not cap.isOpened():
        print("Kamera topilmadi.")
        return

    print("Kamera ishga tushdi. ESC bosib chiqish mumkin.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO orqali aniqlash
        results = model.predict(source=frame, conf=0.30, verbose=False)

        # Natijani chizish
        annotated_frame = results[0].plot()

        # Oynada ko‘rsatish
        cv2.imshow("YOLOv8 – Predmetni tanish", annotated_frame)

        # ESC tugmasi bosilsa, chiqish
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Kamera yopildi.")

if __name__ == "__main__":
    main()
