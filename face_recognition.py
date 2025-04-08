from deepface import DeepFace
import cv2
import numpy as np
import os
from collections import defaultdict
import pickle
import time
import logging
from scipy.spatial import distance
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache file name
CACHE_FILE = "face_cache.pkl"
KNOWN_FACES_DIR = "known_faces"

# Track recognized users in the current session
recognized_users = set()

def load_or_create_cache(known_faces_dir=KNOWN_FACES_DIR):
    if os.path.exists(CACHE_FILE):
        logger.info("Loading cached face encodings...")
        start_time = time.time()
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
            logger.info(f"Cached data loaded in {time.time() - start_time:.2f} seconds")
            return cached_data
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}. Regenerating cache...")
            os.remove(CACHE_FILE)

    logger.info("No cache found or cache invalid. Analyzing known faces and creating cache...")
    start_time = time.time()

    person_faces = defaultdict(list)

    if not os.path.isdir(known_faces_dir):
        os.makedirs(known_faces_dir)
        logger.info(f"Created directory {known_faces_dir}")

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            parts = filename.split('_')
            if len(parts) < 1:
                logger.warning(f"Skipping {filename}: Invalid filename format. Use 'personName_imageNumber.jpg'")
                continue

            person_name = parts[0]
            image_path = os.path.join(known_faces_dir, filename)

            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
                img = cv2.resize(img, (160, 160))

                face_encoding = DeepFace.represent(img_path=img, model_name="Facenet", enforce_detection=False)[0]
                person_faces[person_name].append(face_encoding)
                logger.info(f"Loaded image for {person_name} from {filename}, Encoding shape: {np.array(face_encoding['embedding']).shape}")

            except Exception as e:
                logger.error(f"Failed to load {filename} for {person_name}: {str(e)}")

    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(person_faces, f)
        logger.info(f"Cache created and saved in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to save cache: {str(e)}")

    return person_faces

def compare_face(unknown_encoding, known_encodings, person_faces, threshold=0.6):
    min_distance = float('inf')
    matched_name = "Unknown"
    all_distances = []

    unknown_vec = np.array(unknown_encoding["embedding"])
    logger.debug(f"Unknown encoding shape: {unknown_vec.shape}")

    for known_encoding in known_encodings:
        try:
            known_vec = np.array(known_encoding["embedding"])
            logger.debug(f"Known encoding shape: {known_vec.shape}")

            if unknown_vec.shape != known_vec.shape:
                logger.error(f"Shape mismatch: unknown {unknown_vec.shape} vs known {known_vec.shape}")
                continue

            dist = distance.cosine(unknown_vec, known_vec)
            all_distances.append(dist)
            logger.debug(f"Cosine distance: {dist}")

            if dist < min_distance:
                min_distance = dist
                if dist < threshold:
                    for name, encodings in person_faces.items():
                        if known_encoding in encodings:
                            matched_name = name
                            logger.info(f"Match found with {name} at distance {dist}")
                            break

        except Exception as e:
            logger.error(f"Error comparing encodings: {str(e)}")

    logger.info(f"All cosine distances for this face: {all_distances}")
    logger.info(f"Best cosine distance: {min_distance}, Matched name: {matched_name}")
    return matched_name

def mark_attendance(name, attendance_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(attendance_file, 'a') as f:
        f.write(f"{name}, {timestamp}\n")

def process_frame(frame, person_faces, attendance_file):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.1, beta=10)
    recognized_in_frame = set()

    try:
        faces = DeepFace.extract_faces(img_path=rgb_frame, detector_backend='mtcnn', enforce_detection=False)
        logger.info(f"Detected {len(faces)} faces with mtcnn backend")

        for face in faces:
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            x, y, w, h = max(0, x), max(0, y), min(w, rgb_frame.shape[1] - x), min(h, rgb_frame.shape[0] - y)

            if w <= 0 or h <= 0:
                logger.warning("Invalid face dimensions. Skipping this face...")
                continue

            face_img = rgb_frame[y:y+h, x:x+w]
            if face_img.size == 0:
                logger.warning("Empty face image cropped. Skipping...")
                continue
            face_img = cv2.resize(face_img, (160, 160))

            try:
                unknown_encoding = DeepFace.represent(img_path=face_img, model_name="Facenet", enforce_detection=False)[0]
                name = compare_face(unknown_encoding, [enc for encodings in person_faces.values() for enc in encodings], person_faces)
            except Exception as e:
                logger.error(f"Failed to process face encoding: {str(e)}")
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if name != "Unknown" and name not in recognized_users:
                recognized_users.add(name)
                recognized_in_frame.add(name)
                mark_attendance(name, attendance_file)
                logger.info(f"Attendance marked for {name} in {attendance_file}")

    except Exception as e:
        logger.error(f"Error in face detection or comparison: {str(e)}")

    return frame, recognized_in_frame