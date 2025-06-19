from flask import Flask, render_template, Response
import cv2
import torch
import pathlib

# Fix for Windows: Convert PosixPath to WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# Load model (make sure yolov5 repo is cloned in your project directory)
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform inference
            results = model(frame)
            annotated_frame = results.render()[0]  # draw predictions on the frame

            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield the frame as byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
