from flask import Flask, render_template_string, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("yolov8n.pt")

# Kamerani ochish
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model.predict(source=frame, conf=0.3, verbose=False)
            annotated_frame = results[0].plot()

            # JPEG formatda oqimga o‘tkazish
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # index.html faylni o‘qiydi
    return open("index.html", encoding="utf-8").read()

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
