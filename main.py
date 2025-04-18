# from flask import Flask, render_template, request, send_from_directory
# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import os
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Load the trained model
# model = load_model('models/model.h5')
#
# # Class labels
# class_labels = ['glioma', 'notumor','pituitary','meningioma']
#
# # Define the uploads folder
# UPLOAD_FOLDER = './uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# # Helper function to predict tumor type
# def predict_tumor(image_path):
#     IMAGE_SIZE = 128
#     img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#     img_array = img_to_array(img) / 255.0  # Normalize pixel values
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#
#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     confidence_score = np.max(predictions, axis=1)[0]
#
#     if class_labels[predicted_class_index] == 'notumor':
#         return "No Tumor", confidence_score
#     else:
#         return f"Tumor: {class_labels[predicted_class_index]}", confidence_score
#
# # Route for the main page (index.html)
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Handle file upload
#         file = request.files['file']
#         if file:
#             # Save the file
#             file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_location)
#
#             # Predict the tumor
#             result, confidence = predict_tumor(file_location)
#
#             # Return result along with image path for display
#             return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')
#
#     return render_template('index.html', result=None)
#
# # Route to serve uploaded files
# @app.route('/uploads/<filename>')
# def get_uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, send_from_directory
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load VGG19 classification model
vgg_model = load_model('models/model.h5')

# Load Faster R-CNN model (pretrained on COCO dataset)
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval()  # Set model to evaluation mode

# Class labels for VGG19
class_labels = ['glioma', 'notumor', 'pituitary', 'meningioma']

# Function to classify tumor type using VGG19
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = vgg_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    predicted_label = class_labels[predicted_class_index]

    return predicted_label, confidence_score

# Function to detect tumor using Faster R-CNN (only if VGG19 detects a tumor)
def detect_tumor(image_path, vgg_prediction):
    if vgg_prediction == "notumor":
        return image_path, "No Tumor Detected"

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Run Faster R-CNN
    with torch.no_grad():
        predictions = faster_rcnn(image_tensor)

    # Extract bounding boxes
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Keep only high-confidence bounding boxes
    filtered_boxes = []
    for i in range(len(scores)):
        if scores[i] > 0.6:  # Confidence threshold
            filtered_boxes.append((boxes[i], scores[i]))

    if not filtered_boxes:
        return image_path, "Tumor not detected by Faster R-CNN"

    # Select the best bounding box (highest confidence)
    best_box, best_score = max(filtered_boxes, key=lambda x: x[1])
    x1, y1, x2, y2 = map(int, best_box)

    # Load image with OpenCV to draw the bounding box
    image_cv = cv2.imread(image_path)
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_cv, f"{best_score:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save processed image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'localized_' + os.path.basename(image_path))
    cv2.imwrite(output_path, image_cv)

    return output_path, f"Bounding Box: {best_box}"

# Flask route for main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save uploaded file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor type using VGG19
            vgg_result, confidence = predict_tumor(file_location)

            # Only use Faster R-CNN if VGG19 detects a tumor
            if vgg_result != "notumor":
                localized_image_path, bbox_coordinates = detect_tumor(file_location, vgg_result)
            else:
                localized_image_path, bbox_coordinates = file_location, "No Tumor Detected"

            return render_template('index.html',
                                   result=f"Tumor Type: {vgg_result}",
                                   confidence=f"{confidence*100:.2f}%",
                                   file_path=f'/uploads/{file.filename}',
                                   localized_file_path=f'/uploads/{os.path.basename(localized_image_path)}',
                                   bbox_coordinates=bbox_coordinates)

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
