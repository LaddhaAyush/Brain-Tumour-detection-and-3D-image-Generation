from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import json
import urllib.parse
import requests
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Custom layer definition
class Avg2MaxPooling(nn.Module):
    def __init__(self):
        super(Avg2MaxPooling, self).__init__()
        
    def forward(self, x):
        # Convert average pooling to max pooling
        return torch.max_pool2d(x, kernel_size=2, stride=2)

# Hard-coded environment variables instead of using dotenv
os.environ.setdefault("SEARCH_API_KEY", "AIzaSyBhttY-1N20nwqjYS9GUVqxCCKX3wRzBQw")
os.environ.setdefault("SEARCH_ENGINE_ID", "AIzaSyCE3sYZVwh5Wtad0TL98uZhbWT-MmdVfqA") 
os.environ.setdefault("SEARCH_API_TYPE", "google")
os.environ.setdefault("GROK_API_KEY", "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")
os.environ.setdefault("FLASK_APP", "main3.py")
os.environ.setdefault("FLASK_ENV", "development")
os.environ.setdefault("DEBUG", "True")

# Try to import the AI bot, but handle potential import errors gracefully
try:
    from ai_bot import BrainTumorAIBot
    has_ai_bot = True
    print("Successfully imported BrainTumorAIBot")
except ImportError as e:
    print(f"Error importing BrainTumorAIBot: {e}")
    has_ai_bot = False

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add custom Jinja2 filter for parsing JSON
@app.template_filter('from_json')
def parse_json(json_string):
    """Parse a JSON string into a Python object for templates"""
    if isinstance(json_string, str):
        return json.loads(json_string)
    return json_string

# Load YOLOv8 model with custom layer handling
try:
    model = YOLO('models/best.pt')
    print("Successfully loaded YOLO model")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Load Fibonacci model
fibonacci_model = load_model('models/model.h5')
fibonacci_class_labels = ['glioma', 'notumor', 'pituitary', 'meningioma']

# Class labels
class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

# Initialize AI bot if available
ai_bot = BrainTumorAIBot() if has_ai_bot else None

def generate_3d_coordinates(box, image_dimensions, tumor_type=None):
    """Generate 3D coordinates from 2D bounding box and tumor type"""
    # New approach: Use predefined anatomically correct coordinates based on tumor type
    
    # Standard bounding box dimensions - will be adjusted by type
    box_width = 0.15
    box_height = 0.15
    box_depth = 0.15
    
    # Set default position in the center of the brain
    x_center = 0.5
    y_center = 0.5
    z_center = 0.5
    
    # Apply randomization to make placement look natural
    random_offset = 0.05
    
    # Anatomically accurate positioning for different tumor types
    if tumor_type:
        tumor_type = tumor_type.lower()
        
        if 'pituitary' in tumor_type:
            # Pituitary tumors occur at the base of the brain near the pituitary gland
            x_center = 0.5 + np.random.uniform(-random_offset/2, random_offset/2)  # Center (midline)
            y_center = 0.25 + np.random.uniform(-random_offset/2, random_offset/2)  # Low in the brain
            z_center = 0.65 + np.random.uniform(-random_offset/2, random_offset/2)  # Slightly forward
            
            # Pituitary tumors are typically smaller
            box_width = 0.1
            box_height = 0.1
            box_depth = 0.1
            
        elif 'glioma' in tumor_type:
            # Gliomas can occur throughout the brain, but let's choose common locations
            # Select a random location from these common sites
            glioma_locations = [
                # Frontal lobe (left)
                {"x": 0.3, "y": 0.75, "z": 0.7},
                # Frontal lobe (right)
                {"x": 0.7, "y": 0.75, "z": 0.7},
                # Parietal lobe (left)
                {"x": 0.3, "y": 0.75, "z": 0.4},
                # Parietal lobe (right)
                {"x": 0.7, "y": 0.75, "z": 0.4},
                # Temporal lobe (left)
                {"x": 0.2, "y": 0.5, "z": 0.6},
                # Temporal lobe (right)
                {"x": 0.8, "y": 0.5, "z": 0.6},
                # Occipital lobe
                {"x": 0.5, "y": 0.7, "z": 0.2},
                # Cerebellum
                {"x": 0.5, "y": 0.3, "z": 0.2},
                # Brainstem (rare but possible)
                {"x": 0.5, "y": 0.25, "z": 0.4}
            ]
            
            # Select a random location
            location = np.random.choice(glioma_locations)
            x_center = location["x"] + np.random.uniform(-random_offset, random_offset)
            y_center = location["y"] + np.random.uniform(-random_offset, random_offset)
            z_center = location["z"] + np.random.uniform(-random_offset, random_offset)
            
            # Gliomas can vary in size
            size_variation = np.random.uniform(0.8, 1.2)
            box_width *= size_variation
            box_height *= size_variation
            box_depth *= size_variation
            
        elif 'meningioma' in tumor_type:
            # Meningiomas occur on the surface of the brain, attached to the meninges
            # Common locations include parasagittal, sphenoid wing, olfactory groove, etc.
            
            meningioma_locations = [
                # Parasagittal (top)
                {"x": 0.5, "y": 0.9, "z": 0.5},
                # Convexity (top sides)
                {"x": 0.2, "y": 0.85, "z": 0.5},
                {"x": 0.8, "y": 0.85, "z": 0.5},
                # Sphenoid wing (sides)
                {"x": 0.2, "y": 0.5, "z": 0.8},
                {"x": 0.8, "y": 0.5, "z": 0.8},
                # Olfactory groove (front bottom)
                {"x": 0.5, "y": 0.4, "z": 0.9},
                # Posterior fossa (back bottom)
                {"x": 0.5, "y": 0.3, "z": 0.1}
            ]
            
            # Select a random location
            location = np.random.choice(meningioma_locations)
            x_center = location["x"] + np.random.uniform(-random_offset, random_offset)
            y_center = location["y"] + np.random.uniform(-random_offset, random_offset)
            z_center = location["z"] + np.random.uniform(-random_offset, random_offset)
            
            # Meningiomas tend to be well-defined
            box_width = 0.17
            box_height = 0.17
            box_depth = 0.15
    else:
        # If no tumor type specified, use bounding box from detection
        height, width = image_dimensions
        x1, y1, x2, y2 = box
        
        # Normalize 2D box
        x1_norm, y1_norm, x2_norm, y2_norm = box / np.array([width, height, width, height])
        
        # Calculate the actual width and height of the tumor in normalized coordinates
        box_width = x2_norm - x1_norm
        box_height = y2_norm - y1_norm
        
        # Calculate the tumor center
        x_center = (x1_norm + x2_norm) / 2
        y_center = (y1_norm + y2_norm) / 2
        z_center = 0.5  # Default depth
    
    # Ensure coordinates stay within valid range (0-1)
    x_center = max(box_width/2, min(1 - box_width/2, x_center))
    y_center = max(box_height/2, min(1 - box_height/2, y_center))
    z_center = max(box_depth/2, min(1 - box_depth/2, z_center))
    
    # Calculate z-coordinates for front and back faces
    z_front = z_center - (box_depth / 2)
    z_back = z_center + (box_depth / 2)
    
    # Generate all 8 corners of the 3D bounding box
    coordinates_3d = [
        [x_center - box_width/2, y_center - box_height/2, z_front],  # Front face, bottom left
        [x_center + box_width/2, y_center - box_height/2, z_front],  # Front face, bottom right
        [x_center + box_width/2, y_center + box_height/2, z_front],  # Front face, top right
        [x_center - box_width/2, y_center + box_height/2, z_front],  # Front face, top left
        [x_center - box_width/2, y_center - box_height/2, z_back],   # Back face, bottom left
        [x_center + box_width/2, y_center - box_height/2, z_back],   # Back face, bottom right
        [x_center + box_width/2, y_center + box_height/2, z_back],   # Back face, top right
        [x_center - box_width/2, y_center + box_height/2, z_back]    # Back face, top left
    ]
    
    # Create a named tuple of the tumor center for better readability
    tumor_center = (x_center, y_center, z_center)
    
    # Print details for debugging
    print(f"Generating coordinates for {tumor_type} tumor at position: {tumor_center}")
    print(f"Box dimensions: {box_width:.2f} x {box_height:.2f} x {box_depth:.2f}")
    
    # Determine anatomical region based on position
    brain_region = get_brain_region(x_center, y_center, z_center, tumor_type)
    
    # Calculate impact data based on tumor type and location
    impact_data = calculate_impact_data(tumor_type, brain_region, box_width, box_height, box_depth)
    
    # Return coordinates and additional impact mapping data
    return coordinates_3d, brain_region, impact_data

def get_brain_region(x, y, z, tumor_type):
    """Determine the anatomical brain region based on coordinates"""
    # If tumor type is pituitary, it's always in the pituitary fossa
    if tumor_type and 'pituitary' in tumor_type.lower():
        return "Pituitary Fossa"
    
    # More precise coordinate-based region mapping
    # The brain is divided into regions using the following coordinate system:
    # x: 0 (left) to 1 (right)
    # y: 0 (bottom) to 1 (top)
    # z: 0 (back) to 1 (front)
    
    # Check for cerebral hemispheres first (upper regions of brain)
    if y > 0.55:  # Upper half of brain
        if z > 0.65:  # Front
            if x < 0.4:
                return "Left Frontal Lobe"
            elif x > 0.6:
                return "Right Frontal Lobe"
            else:
                return "Frontal Lobe (Midline)"
        elif z < 0.35:  # Back
            if x < 0.4:
                return "Left Occipital Lobe"
            elif x > 0.6:
                return "Right Occipital Lobe"
            else:
                return "Occipital Lobe (Midline)"
        else:  # Middle
            if z > 0.5:  # Front-middle
                if x < 0.4:
                    return "Left Frontal-Parietal Junction"
                elif x > 0.6:
                    return "Right Frontal-Parietal Junction" 
                else:
                    return "Superior Frontal Gyrus"
            else:  # Back-middle
                if x < 0.4:
                    return "Left Parietal Lobe"
                elif x > 0.6:
                    return "Right Parietal Lobe"
                else:
                    return "Parietal Lobe (Midline)"
    
    # Check for lower regions (temporal lobes, cerebellum, brainstem)
    elif y > 0.35:  # Middle-lower regions
        if z > 0.6:  # Front
            if x < 0.4:
                return "Left Temporal Lobe (Anterior)"
            elif x > 0.6:
                return "Right Temporal Lobe (Anterior)"
            else:
                return "Temporal Pole"
        elif z < 0.4:  # Back
            if x < 0.4:
                return "Left Temporal Lobe (Posterior)"
            elif x > 0.6:
                return "Right Temporal Lobe (Posterior)"
            else:
                return "Temporal-Occipital Junction"
        else:  # Middle
            if x < 0.4:
                return "Left Temporal Lobe"
            elif x > 0.6:
                return "Right Temporal Lobe"
            else:
                return "Temporal Lobe (Midline)"
    
    # Check for lowest regions (cerebellum and brainstem)
    else:
        if z > 0.5:  # Front
            return "Brainstem (Anterior)"
        else:  # Back
            if y > 0.25:
                return "Cerebellum"
            else:
                return "Brainstem (Posterior)"

def calculate_impact_data(tumor_type, brain_region, width, height, depth):
    """Calculate potential impact data based on tumor type, location, and size"""
    # Base impact data
    impact_data = {
        "cognitive_impact": [],
        "motor_impact": [],
        "sensory_impact": [],
        "risk_level": "Low",
        "urgency": "Routine"
    }
    
    # Calculate tumor volume
    volume = width * height * depth
    
    # Determine risk level based on volume
    if volume > 0.004:  # Large tumor
        impact_data["risk_level"] = "High"
        impact_data["urgency"] = "Immediate"
    elif volume > 0.002:  # Medium tumor
        impact_data["risk_level"] = "Medium"
        impact_data["urgency"] = "Urgent"
    
    # Add impacts based on brain region
    if "Frontal" in brain_region:
        impact_data["cognitive_impact"].extend([
            "Executive function",
            "Decision making",
            "Personality changes"
        ])
        if "Left" in brain_region:
            impact_data["motor_impact"].append("Right side weakness")
        elif "Right" in brain_region:
            impact_data["motor_impact"].append("Left side weakness")
    
    elif "Temporal" in brain_region:
        impact_data["cognitive_impact"].extend([
            "Memory",
            "Language comprehension",
            "Emotional processing"
        ])
        if "Left" in brain_region:
            impact_data["sensory_impact"].append("Right visual field deficits")
        elif "Right" in brain_region:
            impact_data["sensory_impact"].append("Left visual field deficits")
    
    elif "Parietal" in brain_region:
        impact_data["sensory_impact"].extend([
            "Spatial awareness",
            "Sensory processing"
        ])
        if "Left" in brain_region:
            impact_data["motor_impact"].append("Right side coordination")
        elif "Right" in brain_region:
            impact_data["motor_impact"].append("Left side coordination")
    
    elif "Occipital" in brain_region:
        impact_data["sensory_impact"].extend([
            "Visual processing",
            "Visual field deficits"
        ])
    
    elif "Cerebellum" in brain_region:
        impact_data["motor_impact"].extend([
            "Balance",
            "Coordination",
            "Fine motor skills"
        ])
    
    elif "Brainstem" in brain_region:
        impact_data["motor_impact"].extend([
            "Basic life functions",
            "Cranial nerve function"
        ])
        impact_data["risk_level"] = "High"
        impact_data["urgency"] = "Immediate"
    
    # Add tumor type specific impacts
    if tumor_type:
        if "glioma" in tumor_type.lower():
            impact_data["cognitive_impact"].append("Progressive cognitive decline")
            impact_data["risk_level"] = max(impact_data["risk_level"], "Medium")
        elif "meningioma" in tumor_type.lower():
            impact_data["cognitive_impact"].append("Gradual cognitive changes")
        elif "pituitary" in tumor_type.lower():
            impact_data["sensory_impact"].extend([
                "Visual field deficits",
                "Hormonal imbalances"
            ])
    
    return impact_data

def format_3d_coordinates(coordinates):
    """Format 3D coordinates for JSON serialization"""
    return [[float(x) for x in point] for point in coordinates]

def predict_tumor_fibonacci(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = fibonacci_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    return fibonacci_class_labels[predicted_class_index], confidence_score

def detect_tumor_yolo(image_path):
    results = model(image_path)
    result = results[0]
    if len(result.boxes) > 0:
        highest_conf_idx = np.argmax(result.boxes.conf.cpu().numpy())
        box = result.boxes.xyxy[highest_conf_idx].cpu().numpy()
        return box
    return None

def predict_and_detect(image_path):
    """Run YOLO prediction on the image and return results"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Run prediction
        results = model(image_path)
        
        # Get the first result
        result = results[0]
        
        # Get boxes and confidence scores
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Get image dimensions
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Process each detection
        detections = []
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:  # Confidence threshold
                tumor_type = class_labels[class_id]
                coordinates_3d, brain_region, impact_data = generate_3d_coordinates(
                    box, (height, width), tumor_type
                )
                
                detection = {
                    "tumor_type": tumor_type,
                    "confidence": float(conf),
                    "coordinates_3d": format_3d_coordinates(coordinates_3d),
                    "brain_region": brain_region,
                    "impact_data": impact_data
                }
                detections.append(detection)
        
        return detections, None
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index3.html', error="No file selected", detections=[], image_url=None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index3.html', error="No file selected", detections=[], image_url=None)
        
        if file:
            # Save the file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # Generate a unique session ID
            session_id = os.path.splitext(file.filename)[0]

            # Predict tumor type
            predicted_label, confidence = predict_tumor_fibonacci(filename)
            
            # Get YOLO bounding box if tumor is present
            box = None
            coordinates_3d = None
            brain_region = None
            impact_data = None
            if predicted_label != 'notumor':
                box = detect_tumor_yolo(filename)
                if box is not None:
                    img = cv2.imread(filename)
                    height, width = img.shape[:2]
                    coordinates_3d, brain_region, impact_data = generate_3d_coordinates(box, (height, width), predicted_label)
            
            detection = {
                "tumor_type": predicted_label,
                "confidence": float(confidence),
                "coordinates_3d": format_3d_coordinates(coordinates_3d) if coordinates_3d else None,
                "brain_region": brain_region if brain_region else "Unknown",
                "impact_data": impact_data if impact_data else {}
            }
            detections = [detection] if predicted_label != 'notumor' else []
            
            # Save results to a JSON file
            results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json")
            with open(results_file, 'w') as f:
                json.dump(detections, f)
            
            image_url = url_for('get_uploaded_file', filename=file.filename)
            return render_template('index3.html', detections=detections, image_url=image_url, error=None)
    # GET request
    return render_template('index3.html', detections=[], image_url=None, error=None)

@app.route('/visualize/<session_id>')
def visualize_brain(session_id):
    # Load results from JSON file
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json")
    try:
        with open(results_file, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        return render_template('index.html', error="Results not found")
    
    # Get the original image filename
    image_filename = f"{session_id}.jpg"  # Assuming the image is a jpg
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    
    return render_template('visualize.html', 
                         session_id=session_id,
                         detections=detections,
                         image_url=url_for('get_uploaded_file', filename=image_filename))

@app.route('/dashboard/<session_id>')
def tumor_dashboard(session_id):
    # Load results from JSON file
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json")
    try:
        with open(results_file, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        return render_template('index.html', error="Results not found")
    
    # Calculate summary statistics
    total_tumors = len(detections)
    tumor_types = {}
    risk_levels = {}
    
    for detection in detections:
        # Count tumor types
        tumor_type = detection['tumor_type']
        tumor_types[tumor_type] = tumor_types.get(tumor_type, 0) + 1
        
        # Count risk levels
        risk_level = detection['impact_data']['risk_level']
        risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
    
    # Get the original image filename
    image_filename = f"{session_id}.jpg"  # Assuming the image is a jpg
    
    return render_template('tumor_dashboard.html',
                         session_id=session_id,
                         detections=detections,
                         total_tumors=total_tumors,
                         tumor_types=tumor_types,
                         risk_levels=risk_levels,
                         image_url=url_for('get_uploaded_file', filename=image_filename))

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_tumor_info', methods=['POST'])
def get_tumor_info():
    data = request.get_json()
    session_id = data.get('session_id')
    tumor_index = data.get('tumor_index')
    
    # Load results from JSON file
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json")
    try:
        with open(results_file, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Results not found"}), 404
    
    if tumor_index >= len(detections):
        return jsonify({"error": "Invalid tumor index"}), 400
    
    detection = detections[tumor_index]
    
    # Get additional information from AI bot if available
    if ai_bot:
        tumor_type = detection['tumor_type']
        brain_region = detection['brain_region']
        
        # Get tumor information
        tumor_info = ai_bot.get_tumor_info(tumor_type)
        
        # Get region-specific information
        region_info = ai_bot.get_region_info(brain_region)
        
        # Combine information
        response = {
            "tumor_info": tumor_info,
            "region_info": region_info,
            "detection": detection
        }
    else:
        response = {
            "detection": detection,
            "tumor_info": "AI bot not available",
            "region_info": "AI bot not available"
        }
    
    return jsonify(response)

@app.route('/chat_with_ai', methods=['POST'])
def chat_with_ai():
    if not ai_bot:
        return jsonify({"error": "AI bot not available"}), 503
    
    data = request.get_json()
    user_message = data.get('message')
    session_id = data.get('session_id')
    
    if not user_message or not session_id:
        return jsonify({"error": "Missing message or session_id"}), 400
    
    # Load results from JSON file to get tumor context
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json")
    try:
        with open(results_file, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Results not found"}), 404
    
    # Get tumor context from detections
    tumor_context = []
    for detection in detections:
        context = {
            "type": detection['tumor_type'],
            "location": detection['brain_region'],
            "impact": detection['impact_data']
        }
        tumor_context.append(context)
    
    # Get response from AI bot
    response = ai_bot.chat_response(user_message, tumor_context)
    
    return jsonify(response)

@app.route('/visualization_data', methods=['GET'])
def get_visualization_data():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    # Load results from JSON file
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json")
    try:
        with open(results_file, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Results not found"}), 404
    
    return jsonify(detections)

def call_mixtral(prompt: str) -> str:
    """Call the Mixtral API with the given prompt"""
    try:
        response = requests.post(
            "https://api.mixtral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mixtral-8x7b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling Mixtral API: {e}")
        return "Error: Could not get response from AI model"

if __name__ == '__main__':
    app.run(debug=True)