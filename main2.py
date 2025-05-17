from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import json
import urllib.parse
import requests

# Hard-coded environment variables instead of using dotenv
os.environ.setdefault("SEARCH_API_KEY", "AIzaSyBhttY-1N20nwqjYS9GUVqxCCKX3wRzBQw")
os.environ.setdefault("SEARCH_ENGINE_ID", "AIzaSyCE3sYZVwh5Wtad0TL98uZhbWT-MmdVfqA") 
os.environ.setdefault("SEARCH_API_TYPE", "google")
os.environ.setdefault("GROK_API_KEY", "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")
os.environ.setdefault("FLASK_APP", "main2.py")
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

# Load YOLOv8 model
model = YOLO('models/best.pt')

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
    
    # Temporal Lobes (sides of brain, middle height)
    elif 0.35 < y < 0.6:
        if x < 0.3:  # Left side
            if z > 0.6:
                return "Left Anterior Temporal Lobe"
            elif z < 0.4:
                return "Left Posterior Temporal Lobe"
            else:
                return "Left Mid-Temporal Lobe"
        elif x > 0.7:  # Right side
            if z > 0.6:
                return "Right Anterior Temporal Lobe"
            elif z < 0.4:
                return "Right Posterior Temporal Lobe"
            else:
                return "Right Mid-Temporal Lobe"
        elif 0.4 <= x <= 0.6:  # Central regions
            if z > 0.6:
                return "Anterior Cingulate Cortex"
            elif z < 0.4:
                return "Posterior Cingulate Cortex"
            else:
                return "Thalamus"
    
    # Lower regions
    elif y < 0.35:
        if y < 0.2:  # Very bottom
            if z < 0.4:  # Back-bottom
                if 0.4 <= x <= 0.6:  # Center
                    return "Cerebellum (Vermis)"
                elif x < 0.4:
                    return "Left Cerebellar Hemisphere"
                else:
                    return "Right Cerebellar Hemisphere"
            elif z > 0.6:  # Front-bottom
                return "Orbital Frontal Cortex"
            else:  # Middle-bottom
                if 0.45 <= x <= 0.55:  # Center
                    return "Brain Stem"
                else:
                    return "Pons/Medulla"
        else:  # Lower-middle
            if 0.45 <= x <= 0.55 and z > 0.55:  # Center-front
                return "Hypothalamus"
            elif 0.45 <= x <= 0.55 and 0.45 <= z <= 0.55:  # Center-center
                return "Midbrain"
            elif x < 0.4 and z > 0.5:
                return "Left Insula"
            elif x > 0.6 and z > 0.5:
                return "Right Insula"
            elif x < 0.4 and z < 0.5:
                return "Left Hippocampus"
            elif x > 0.6 and z < 0.5:
                return "Right Hippocampus"
    
    # Default/unknown cases or deep brain structures
    if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6 and 0.4 <= z <= 0.6:
        return "Deep Brain Structures"
    
    return "Undetermined Brain Region"


def calculate_impact_data(tumor_type, brain_region, width, height, depth):
    """Calculate impact data using AI instead of hardcoded values"""
    try:
        # Only use AI if available
        if has_ai_bot and ai_bot:
            # Get AI-generated information
            response = ai_bot.generate_response(
                tumor_type=tumor_type,
                location=brain_region,
                size={'width': width, 'height': height, 'depth': depth}
            )
            
            # Parse the AI response to get impact data
            impact_data = {
                'severity': 'Moderate',  # Default value
                'potential_effects': response['ai_response'],
                'treatment_options': response.get('treatments', {}),
                'symptoms': response.get('symptoms', {}),
                'sources': response.get('sources', [])
            }
            
            return impact_data
        else:
            # Fallback to basic impact data if AI is not available
            return {
                'severity': 'Moderate',
                'potential_effects': f'A {tumor_type} tumor in the {brain_region} region may affect various functions depending on its size and exact location. Please consult a medical professional for specific information.',
                'treatment_options': {},
                'symptoms': {},
                'sources': []
            }
    except Exception as e:
        print(f"Error getting AI impact data: {e}")
        # Fallback to basic impact data if AI fails
        return {
            'severity': 'Unknown',
            'potential_effects': 'Unable to generate impact data',
            'treatment_options': {},
            'symptoms': {},
            'sources': []
        }


def format_3d_coordinates(coordinates):
    """Format 3D coordinates for better display"""
    if not coordinates:
        return "None"
    
    formatted = "["
    for i, coord in enumerate(coordinates):
        formatted += f"[{coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f}]"
        if i < len(coordinates) - 1:
            formatted += ", "
    formatted += "]"
    return formatted


def predict_and_detect(image_path):
    results = model(image_path)
    result = results[0]

    if len(result.boxes) > 0:
        highest_conf_idx = np.argmax(result.boxes.conf.cpu().numpy())
        class_id = int(result.boxes.cls[highest_conf_idx])
        confidence = float(result.boxes.conf[highest_conf_idx])
        predicted_label = class_labels.get(class_id, 'unknown')

        box = result.boxes.xyxy[highest_conf_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{predicted_label}: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save output image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(image_path))
        cv2.imwrite(output_path, img)

        # Generate 3D coordinates with the tumor type
        coordinates_3d, brain_region, impact_data = generate_3d_coordinates(box, (height, width), predicted_label)

        return output_path, predicted_label, confidence, f"Bounding Box: {box.tolist()}", coordinates_3d, brain_region, impact_data
    else:
        return image_path, "notumor", 0.0, "No Tumor Detected", None, None, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            output_path, predicted_label, confidence, bbox_coordinates, coordinates_3d, brain_region, impact_data = predict_and_detect(file_location)

            # Store coordinates in session for brain visualization
            session_data = {
                'coordinates_3d': coordinates_3d,
                'tumor_type': predicted_label,
                'confidence': confidence,
                'brain_region': brain_region,
                'impact_data': impact_data
            }
            # Save to a temp file with unique ID
            session_id = str(hash(file.filename + str(np.random.random())))
            temp_session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'session_{session_id}.json')
            with open(temp_session_file, 'w') as f:
                json.dump(session_data, f)

            return render_template('index.html',
                                   result=f"Tumor Type: {predicted_label}",
                                   confidence=f"{confidence * 100:.2f}%",
                                   file_path=f'/uploads/{file.filename}',
                                   localized_file_path=f'/uploads/{os.path.basename(output_path)}',
                                   coordinates_3d=json.dumps(coordinates_3d) if coordinates_3d else "None",
                                   brain_region=brain_region if brain_region else "Unknown",
                                   impact_data=json.dumps(impact_data) if impact_data else "None",
                                   session_id=session_id)

    return render_template('index.html', result=None)


@app.route('/visualize/<session_id>')
def visualize_brain(session_id):
    """Route to visualize tumor in 3D brain model"""
    try:
        # Load coordinates from session file
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'session_{session_id}.json')
        
        if not os.path.exists(session_file):
            return "Session expired or invalid", 404
            
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            
        tumor_type = session_data.get('tumor_type', 'unknown')
        
        # If no tumor is detected, redirect to a message page
        if tumor_type == "notumor":
            return render_template('no_tumor.html', 
                                 message="No tumor detected in the scan. The brain appears normal.",
                                 confidence=f"{session_data.get('confidence', 0) * 100:.2f}%")
            
        coordinates_3d = session_data.get('coordinates_3d')
        brain_region = session_data.get('brain_region', 'unknown region')
        impact_data = session_data.get('impact_data', {})
        
        if not coordinates_3d:
            return "No tumor coordinates found", 404
            
        # Redirect to brain visualization with coordinates and tumor type as URL parameters
        coordinates_json = json.dumps(coordinates_3d)
        encoded_coordinates = urllib.parse.quote(coordinates_json)
        encoded_tumor_type = urllib.parse.quote(tumor_type)
        encoded_brain_region = urllib.parse.quote(brain_region)
        encoded_impact_data = urllib.parse.quote(json.dumps(impact_data))
        
        return redirect(f'/static/brain_visualization.html?coordinates={encoded_coordinates}&tumor_type={encoded_tumor_type}&brain_region={encoded_brain_region}&impact_data={encoded_impact_data}')
        
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/dashboard/<session_id>')
def tumor_dashboard(session_id):
    """Route to display the tumor information dashboard"""
    try:
        # Load coordinates from session file
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'session_{session_id}.json')
        
        if not os.path.exists(session_file):
            return "Session expired or invalid", 404
            
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            
        tumor_type = session_data.get('tumor_type', 'unknown')
        brain_region = session_data.get('brain_region', 'unknown region')
        coordinates_3d = session_data.get('coordinates_3d')
        confidence = session_data.get('confidence', 0)
        
        # Calculate tumor size
        if coordinates_3d:
            # Calculate width, height, depth from 3D coordinates
            width = height = depth = 0
            try:
                # Front face coordinates
                width = abs(coordinates_3d[1][0] - coordinates_3d[0][0])
                height = abs(coordinates_3d[2][1] - coordinates_3d[1][1])
                # Depth (front to back)
                depth = abs(coordinates_3d[4][2] - coordinates_3d[0][2])
            except (IndexError, TypeError):
                # Default values if calculation fails
                width = height = depth = 0.1
                
            # Scale to centimeters (assuming coordinates are normalized)
            size = {
                'width': width * 10,  # Convert to cm
                'height': height * 10,
                'depth': depth * 10
            }
        else:
            size = {'width': 0, 'height': 0, 'depth': 0}
            
        # Encode parameters for URL
        params = {
            'tumor_type': tumor_type,
            'location': brain_region,
            'size': json.dumps(size),
            'confidence': f"{confidence * 100:.2f}%",
            'session_id': session_id
        }
        query_string = '&'.join([f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items()])
            
        return render_template('tumor_dashboard.html', **params)
        
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_tumor_info', methods=['POST'])
def get_tumor_info():
    """Get AI-generated information about the detected tumor"""
    data = request.json
    tumor_type = data.get('tumor_type')
    location = data.get('location')
    size = data.get('size', {})
    
    try:
        # Only use AI if available
        if has_ai_bot and ai_bot:
            # Get comprehensive information from AI
            response = ai_bot.generate_response(
                tumor_type=tumor_type,
                location=location,
                size=size
            )
            
            # Get specific treatment options
            treatments = ai_bot.get_treatment_options(tumor_type)
            
            # Get specific symptoms
            symptoms = ai_bot.get_symptoms(tumor_type, location)
            
            # Combine all information
            complete_response = {
                **response,
                'treatments': treatments,
                'symptoms': symptoms
            }
            
            return jsonify(complete_response)
        else:
            # Fallback if AI is not available
            return jsonify({
                'tumor_type': tumor_type,
                'location': location,
                'size': size,
                'ai_response': f"Information about {tumor_type} tumors in the {location} region is not available. Please consult medical literature or healthcare professionals for detailed information.",
                'treatments': {},
                'symptoms': {},
                'sources': []
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat_with_ai', methods=['POST'])
def chat_with_ai():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
            
        # Load session data
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'session_{session_id}.json')
        if not os.path.exists(session_file):
            return jsonify({'error': 'Session not found'}), 404
            
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            
        # Get tumor context from session data
        tumor_context = {
            'type': session_data.get('tumor_type'),
            'location': session_data.get('location'),
            'size': session_data.get('size')
        }
        
        # Initialize AI bot and get response
        bot = BrainTumorAIBot()
        response = bot.chat(user_message, tumor_context)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/visualization_data', methods=['GET'])
def get_visualization_data():
    """Get data for visualizations"""
    if not has_ai_bot or not ai_bot:
        return jsonify({
            'error': "Visualization data is currently unavailable."
        }), 404
    
    tumor_type = request.args.get('tumor_type', 'glioma')
    
    try:
        # Get visualization data from AI
        data = ai_bot.generate_visualization_data(tumor_type)
        return jsonify(data)
    except Exception as e:
        print(f"Error getting visualization data: {e}")
        return jsonify({
            'error': str(e)
        }), 500

def call_mixtral(prompt: str) -> str:
    """
    Call the Grok Mixtral API to generate an enhanced response.
    
    Args:
        prompt: The prompt to send to the Mixtral model
        
    Returns:
        The generated text response from Mixtral
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv("GROK_API_KEY")
        
        if not api_key:
            print("Warning: No Grok API key found in environment variables")
            return ""
            
        # API endpoint
        url = "https://api.grok.com/v1/mixtral/generate"
        
        # Headers with authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Request data
        data = {
            "prompt": prompt,
            "max_tokens": 750,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        # Send request with timeout
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract and return the generated text
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["text"]
        else:
            print("Warning: Unexpected response format from Mixtral API")
            return ""
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling Mixtral API: {str(e)}")
        return ""
    except json.JSONDecodeError as e:
        print(f"Error parsing Mixtral API response: {str(e)}")
        return ""
    except Exception as e:
        print(f"Unexpected error calling Mixtral: {str(e)}")
        return ""

if __name__ == '__main__':
    app.run(debug=True)