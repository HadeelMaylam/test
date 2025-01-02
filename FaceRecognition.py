import sqlite3
from deepface import DeepFace
import cv2
import numpy as np
import base64
import io
import os
import shutil

def init_database():

    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
   
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_data BLOB NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def image_to_blob(image_path):
    # Read image and convert to blob
    img = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def blob_to_temp_file(blob_data):
    # Convert blob back to image and save temporarily
    nparr = np.frombuffer(blob_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_path = f"temp_{os.urandom(4).hex()}.jpg"
    cv2.imwrite(temp_path, img)
    return temp_path

def register_face(image_path, person_name):
    try:
        # Verify if the image contains a face
        face = DeepFace.extract_faces(image_path, detector_backend='retinaface')
        
        # Convert image to blob
        image_blob = image_to_blob(image_path)
        
        # Store information in SQLite database
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (name, image_data)
            VALUES (?, ?)
        ''', (person_name, image_blob))
        
        conn.commit()
        conn.close()
        
        return True, "Face registered successfully!"
        
    except Exception as e:
        return False, f"Error registering face: {str(e)}"

def verify_face(image_path):
    try:
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        # Check if there are any registered users
        cursor.execute('SELECT COUNT(*) FROM users')
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.close()
            return False, "No faces in database"

        # Create a temporary file from the uploaded image
        temp_input_path = f"temp_input_{os.urandom(4).hex()}.jpg"
        
        try:
            # If image_path is a file-like object (UploadFile)
            if hasattr(image_path, 'file'):
                with open(temp_input_path, 'wb') as f:
                    shutil.copyfileobj(image_path.file, f)
            # If image_path is a string path
            elif isinstance(image_path, str):
                if os.path.isfile(image_path):
                    shutil.copy(image_path, temp_input_path)
                else:
                    return False, "Invalid image path"
            else:
                return False, "Unsupported image format"

            # Get all registered faces
            cursor.execute('SELECT id, name, image_data FROM users')
            registered_faces = cursor.fetchall()
            
            best_match = None
            min_distance = float('inf')

            # Try different face detection backends
            backends = ["retinaface", "mtcnn", "opencv"]
            face_detected = False
            successful_backend = None
            
            for backend in backends:
                try:
                    faces = DeepFace.extract_faces(temp_input_path, detector_backend=backend)
                    if faces:
                        face_detected = True
                        successful_backend = backend
                        print(f"Face detected using {backend}")
                        break
                except Exception as e:
                    print(f"Backend {backend} failed: {str(e)}")
                    continue

            if not face_detected:
                return False, "No face detected in the image. Please try again."

            # Get embedding for the input image
            try:
                input_embedding = DeepFace.represent(
                    temp_input_path,
                    model_name="GhostFaceNet",
                    detector_backend=successful_backend,
                    enforce_detection=False,
                    align=True
                )[0]['embedding']
            except Exception as e:
                print(f"Error getting input embedding: {str(e)}")
                return False, "Failed to process input image"

            for face_id, name, face_data in registered_faces:
                temp_path = f"temp_{face_id}_{os.urandom(4).hex()}.jpg"
                
                try:
                    with open(temp_path, 'wb') as f:
                        f.write(face_data)
                    
                    # Get embedding for database image
                    db_embedding = DeepFace.represent(
                        temp_path,
                        model_name="GhostFaceNet",
                        detector_backend=successful_backend,
                        enforce_detection=False,
                        align=True
                    )[0]['embedding']
                    
                    # Calculate cosine distance
                    def cosine_distance(vector_a, vector_b):
                        dot_product = np.dot(vector_a, vector_b)
                        norm_a = np.linalg.norm(vector_a)
                        norm_b = np.linalg.norm(vector_b)
                        return 1 - (dot_product / (norm_a * norm_b))
                    
                    distance = cosine_distance(input_embedding, db_embedding)
                    
                    print(f"Distance for {name}: {distance}")
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = (name, 1 - distance)
                    
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        finally:
            # Clean up input temp file
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            
        conn.close()
        
        # Threshold for GhostFaceNet (may need adjustment)
        threshold = 0.35
        if best_match and min_distance < threshold:
            confidence = best_match[1]
            return True, f"Match found! Person: {best_match[0]} (Confidence: {confidence:.2%})"
        else:
            return False, f"No match found in database (Best distance: {min_distance:.2f})"
            
    except Exception as e:
        print(f"Verification error: {str(e)}")  # Debug print
        return False, f"Error during verification: {str(e)}"

def capture_image():
    """Capture image from webcam"""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image (Press SPACE to capture, Q to quit)', frame)
        
        key = cv2.waitKey(1)
        if key == 32:  # SPACE key
            img_path = f"temp_{os.urandom(4).hex()}.jpg"
            cv2.imwrite(img_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return img_path
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

def get_all_users():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, name, image_data 
        FROM users 
        ORDER BY id DESC
    ''')
    
    users = []
    for row in cursor.fetchall():
        # Convert image blob to base64 string
        image_base64 = base64.b64encode(row[2]).decode('utf-8')
        users.append({
            'id': row[0],
            'name': row[1],
            'image': image_base64
        })
    
    conn.close()
    return users

def main():
    db_path = init_database()
    
    while True:
        print("\n1. Register new face")
        print("2. Verify face")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            name = input("Enter person's name: ")
            print("1. Upload image")
            print("2. Capture from webcam")
            img_choice = input("Enter choice (1-2): ")
            
            if img_choice == "1":
                image_path = input("Enter image path: ")
            else:
                image_path = capture_image()
                if not image_path:
                    continue
            
            success, message = register_face(image_path, name)
            print(message)
            
            # Clean up temporary capture file
            if img_choice == "2" and os.path.exists(image_path):
                os.remove(image_path)
                
        elif choice == "2":
            print("1. Upload image")
            print("2. Capture from webcam")
            img_choice = input("Enter choice (1-2): ")
            
            if img_choice == "1":
                image_path = input("Enter image path: ")
            else:
                image_path = capture_image()
                if not image_path:
                    continue
            
            success, message = verify_face(image_path)
            print(message)
            
            # Clean up temporary capture file
            if img_choice == "2" and os.path.exists(image_path):
                os.remove(image_path)
                
        elif choice == "3":
            break

if __name__ == "__main__":
    main()