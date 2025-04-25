from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO('models/best.pt')  # Make sure this is your YOLOv8 model file

# Class labels (update these to match your YOLO model's classes)
class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}


def predict_and_detect(image_path):
    # Run YOLOv8 inference
    results = model(image_path)

    # Get the first result (since we're processing one image at a time)
    result = results[0]

    # Get class predictions and confidence scores
    if len(result.boxes) > 0:
        # Get the detection with highest confidence
        highest_conf_idx = np.argmax(result.boxes.conf.cpu().numpy())
        class_id = int(result.boxes.cls[highest_conf_idx])
        confidence = float(result.boxes.conf[highest_conf_idx])
        predicted_label = class_labels.get(class_id, 'unknown')

        # Get bounding box coordinates
        box = result.boxes.xyxy[highest_conf_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box on the image
        img = cv2.imread(image_path)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{predicted_label}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(image_path))
        cv2.imwrite(output_path, img)

        return output_path, predicted_label, confidence, f"Bounding Box: {box}"
    else:
        # No detections
        return image_path, "notumor", 0.0, "No Tumor Detected"


# Flask route for main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save uploaded file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Run prediction and detection
            output_path, predicted_label, confidence, bbox_coordinates = predict_and_detect(file_location)

            return render_template('index.html',
                                   result=f"Tumor Type: {predicted_label}",
                                   confidence=f"{confidence * 100:.2f}%",
                                   file_path=f'/uploads/{file.filename}',
                                   localized_file_path=f'/uploads/{os.path.basename(output_path)}',
                                   bbox_coordinates=bbox_coordinates)

    return render_template('index.html', result=None)


# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)