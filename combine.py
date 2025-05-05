from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
import os
from PIL import Image
import json

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
yolo_model = YOLO('models/best.pt')  # YOLOv8 model for detection
vgg_model = load_model('models/model.h5')  # VGG19 model for classification

# Class labels
yolo_class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
vgg_class_labels = ['glioma', 'notumor', 'pituitary', 'meningioma']


def predict_tumor_vgg(image_path):
    """Use VGG19 model for tumor classification"""
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = vgg_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    return vgg_class_labels[predicted_class_index], confidence_score


def detect_tumor_yolo(image_path):
    """Use YOLOv8 model for tumor detection and bounding box"""
    results = yolo_model(image_path)
    result = results[0]

    if len(result.boxes) > 0:
        # Get the detection with highest confidence
        highest_conf_idx = np.argmax(result.boxes.conf.cpu().numpy())
        box = result.boxes.xyxy[highest_conf_idx].cpu().numpy()
        return box
    return None


def generate_3d_coordinates(box, image_dimensions):
    """Generate 3D coordinates from 2D bounding box"""
    height, width = image_dimensions

    # Convert 2D coordinates to normalized coordinates (0-1 range)
    x1_norm, y1_norm, x2_norm, y2_norm = box / np.array([width, height, width, height])

    # For 3D coordinates, we'll assume:
    # - x,y are the 2D coordinates
    # - z is depth, which we'll estimate based on tumor size
    # (In a real application, you would use depth information from MRI scans)

    # Calculate tumor size (normalized)
    tumor_size = np.sqrt((x2_norm - x1_norm) * (y2_norm - y1_norm))

    # Generate 3D bounding box coordinates (8 corners)
    z_front = 0.5 - tumor_size / 2  # Front plane
    z_back = 0.5 + tumor_size / 2  # Back plane

    coordinates_3d = [
        # Front face
        [x1_norm, y1_norm, z_front],
        [x2_norm, y1_norm, z_front],
        [x2_norm, y2_norm, z_front],
        [x1_norm, y2_norm, z_front],
        # Back face
        [x1_norm, y1_norm, z_back],
        [x2_norm, y1_norm, z_back],
        [x2_norm, y2_norm, z_back],
        [x1_norm, y2_norm, z_back]
    ]

    return coordinates_3d


def process_image(image_path):
    """Process image with both models"""
    # Get VGG19 prediction
    predicted_label, confidence = predict_tumor_vgg(image_path)

    # Get YOLOv8 detection
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    box = detect_tumor_yolo(image_path)

    if box is not None and predicted_label != 'notumor':
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{predicted_label}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Generate 3D coordinates
        coordinates_3d = generate_3d_coordinates(box, (height, width))

        # Save processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(image_path))
        cv2.imwrite(output_path, img)

        return {
            'output_path': output_path,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'bbox_2d': box.tolist(),
            'coordinates_3d': coordinates_3d,
            'has_tumor': True
        }
    else:
        # No tumor case
        return {
            'output_path': image_path,
            'predicted_label': 'notumor',
            'confidence': 1.0,
            'bbox_2d': None,
            'coordinates_3d': None,
            'has_tumor': False
        }


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save uploaded file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Process image
            result = process_image(file_location)

            # Prepare response
            response = {
                'result': f"Tumor Type: {result['predicted_label']}",
                'confidence': f"{result['confidence'] * 100:.2f}%",
                'file_path': f'/uploads/{file.filename}',
                'localized_file_path': f'/uploads/{os.path.basename(result["output_path"])}',
                'has_tumor': result['has_tumor']
            }

            if result['has_tumor']:
                response['bbox_coordinates'] = f"2D Bounding Box: {result['bbox_2d']}"
                response['coordinates_3d'] = json.dumps(result['coordinates_3d'])
            else:
                response['bbox_coordinates'] = "No Tumor Detected"
                response['coordinates_3d'] = None

            return render_template('index2.html', **response)

    return render_template('index2.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)