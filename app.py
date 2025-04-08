from flask import Flask, render_template, request, Response, jsonify
from flask_sock import Sock
import cv2
import os
import platform
from face_recognition import load_or_create_cache, process_frame
import logging
from datetime import datetime
import time
import atexit
import multiprocessing
import warnings
import json

# Suppress multiprocessing resource tracker warning
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Force spawn method for multiprocessing on macOS
if platform.system() == "Darwin":
    multiprocessing.set_start_method('spawn', force=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

# Load person faces at startup
person_faces = load_or_create_cache()

# Attendance file name
attendance_file = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.txt"

# Webcam setup
video_capture = None

def initialize_webcam(force_reset=False):
    global video_capture
    is_macos = platform.system() == "Darwin"
    max_attempts = 5
    attempt = 0
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if is_macos else [cv2.CAP_ANY]

    if force_reset and video_capture is not None and video_capture.isOpened():
        video_capture.release()
        logger.info("Webcam forcefully released for reset")
        video_capture = None

    while attempt < max_attempts:
        for backend in backends:
            try:
                logger.info(f"Attempt {attempt + 1}/{max_attempts}: Initializing webcam {'on macOS' if is_macos else ''} with backend {backend}...")
                video_capture = cv2.VideoCapture(0, backend)

                if not video_capture.isOpened():
                    logger.warning(f"Camera index 0 failed to open with backend {backend}")
                    continue

                if is_macos:
                    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                video_capture.set(cv2.CAP_PROP_FPS, 30)

                logger.info(f"Webcam opened with backend {backend}. Testing in stream...")
                return True

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} with backend {backend} failed: {str(e)}")

        attempt += 1
        time.sleep(2)

    logger.error("All attempts to initialize webcam failed")
    return False

# Initialize webcam at startup
if not initialize_webcam():
    logger.error("Could not initialize webcam at startup. Will retry on demand.")

# Cleanup function for atexit
def shutdown_cleanup():
    global video_capture
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
        logger.info("Webcam released during shutdown")
        video_capture = None

atexit.register(shutdown_cleanup)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global person_faces
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'No file or name provided'}), 400

    file = request.files['file']
    name = request.form['name'].strip()

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    existing_files = [f for f in os.listdir('known_faces') if f.startswith(name + '_')]
    image_number = len(existing_files) + 1
    filename = f"{name}_{image_number}.jpg"
    file_path = os.path.join('known_faces', filename)

    file.save(file_path)
    person_faces = load_or_create_cache()
    return jsonify({'message': f'Image {filename} uploaded successfully'}), 200

@app.route('/webcam')
def webcam():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        logger.warning("Webcam not available. Attempting to reinitialize...")
        if not initialize_webcam():
            return render_template('webcam.html', error="Webcam not available. Please check your camera and permissions.")
    return render_template('webcam.html', error=None)

def gen_frames():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        logger.error("Cannot generate frames: Webcam not initialized. Attempting reinitialization...")
        if not initialize_webcam():
            return

    while True:
        success, frame = video_capture.read()
        if not success:
            logger.warning("Frame read failed. Attempting reinitialization...")
            if not initialize_webcam(force_reset=True):
                return
            continue

        try:
            frame, recognized_in_frame = process_frame(frame, person_faces, attendance_file)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame")
                continue
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            if recognized_in_frame:
                for ws in active_websockets:
                    ws.send(json.dumps({'recognized_names': list(recognized_in_frame)}))
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            continue

@app.route('/video_feed')
def video_feed():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        logger.error("Video feed requested but webcam not initialized. Attempting reinitialization...")
        if not initialize_webcam():
            return Response("Webcam not available. Please ensure your camera is connected and accessible.", status=500)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket for real-time notifications
active_websockets = set()

@sock.route('/ws')
def websocket(ws):
    active_websockets.add(ws)
    try:
        while True:
            ws.receive()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        active_websockets.remove(ws)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)