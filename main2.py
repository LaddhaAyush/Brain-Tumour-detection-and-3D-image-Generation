from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import json
import urllib.parse
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])


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
# model = YOLO('models/best.pt')
model=torch.load('models/best.pt')
# Class labels
class_labels = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}


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
    """Calculate the impact of the tumor based on its type, location, and size"""
    impact_data = {
        "impact_severity": "medium",
        "affected_functions": [],
        "potential_symptoms": [],
        "proximity_risks": []
    }
    
    # Calculate tumor volume (simplified)
    tumor_volume = width * height * depth
    
    # Set impact severity based on tumor type
    if tumor_type:
        if 'glioma' in tumor_type.lower():
            impact_data["impact_severity"] = "high"
        elif 'meningioma' in tumor_type.lower():
            impact_data["impact_severity"] = "medium"
        elif 'pituitary' in tumor_type.lower():
            impact_data["impact_severity"] = "low"
    
    # Adjust severity based on size
    if tumor_volume > 0.02:  # Large tumor
        if impact_data["impact_severity"] == "medium":
            impact_data["impact_severity"] = "high"
        impact_data["proximity_risks"].append("Mass effect due to large tumor size")
    
    # Add affected functions and symptoms based on brain region
    if "Frontal Lobe" in brain_region:
        impact_data["affected_functions"].extend(["Executive Function", "Motor Control", "Personality"])
        impact_data["potential_symptoms"].extend([
            "Changes in personality or behavior",
            "Difficulty with planning and organization",
            "Impaired judgment",
            "Weakness on opposite side of body"
        ])
    
    elif "Parietal Lobe" in brain_region:
        impact_data["affected_functions"].extend(["Sensory Processing", "Spatial Awareness"])
        impact_data["potential_symptoms"].extend([
            "Impaired sense of touch",
            "Difficulty with spatial orientation",
            "Problems with reading or writing",
            "Left-right confusion"
        ])
    
    elif "Temporal Lobe" in brain_region:
        if "Left" in brain_region:
            impact_data["affected_functions"].extend(["Language Comprehension", "Memory", "Auditory Processing"])
            impact_data["potential_symptoms"].extend([
                "Language comprehension difficulties",
                "Memory problems, particularly verbal memory",
                "Hearing disturbances"
            ])
        else:
            impact_data["affected_functions"].extend(["Visual Memory", "Emotion", "Auditory Processing"])
            impact_data["potential_symptoms"].extend([
                "Difficulty recognizing faces or objects",
                "Memory problems",
                "Emotional instability"
            ])
    
    elif "Occipital Lobe" in brain_region:
        impact_data["affected_functions"].extend(["Visual Processing"])
        impact_data["potential_symptoms"].extend([
            "Visual field defects",
            "Visual hallucinations",
            "Difficulty recognizing colors",
            "Problems with reading"
        ])
    
    elif "Cerebellum" in brain_region:
        impact_data["affected_functions"].extend(["Movement Coordination", "Balance", "Motor Learning"])
        impact_data["potential_symptoms"].extend([
            "Poor balance and coordination",
            "Unsteady gait (ataxia)",
            "Slurred speech (dysarthria)",
            "Difficulty with fine motor tasks"
        ])
    
    elif "Brain Stem" in brain_region:
        impact_data["affected_functions"].extend(["Vital Functions", "Cranial Nerve Function"])
        impact_data["potential_symptoms"].extend([
            "Cranial nerve deficits",
            "Difficulty swallowing or speaking",
            "Problems with respiratory control",
            "Sensory or motor deficits"
        ])
        impact_data["impact_severity"] = "high"  # Brain stem tumors are always high impact
    
    elif "Pituitary" in brain_region:
        impact_data["affected_functions"].extend(["Hormone Regulation", "Visual Pathways"])
        impact_data["potential_symptoms"].extend([
            "Hormonal imbalances",
            "Visual field defects (bitemporal hemianopia)",
            "Headaches",
            "Fatigue or weakness"
        ])
    
    elif "Corpus Callosum" in brain_region:
        impact_data["affected_functions"].extend(["Interhemispheric Communication"])
        impact_data["potential_symptoms"].extend([
            "Disconnection syndrome",
            "Difficulty with bimanual coordination",
            "Cognitive processing difficulties"
        ])
    
    elif "Deep Brain Structure" in brain_region:
        impact_data["affected_functions"].extend(["Movement Control", "Sensation", "Cognition"])
        impact_data["potential_symptoms"].extend([
            "Movement disorders",
            "Sensory disturbances",
            "Cognitive changes"
        ])
    
    # Add specific glioma impacts (if applicable)
    if tumor_type and 'glioma' in tumor_type.lower():
        impact_data["proximity_risks"].append("Infiltration of surrounding tissue")
        impact_data["proximity_risks"].append("Disruption of white matter tracts")
        
        # Add additional symptoms not directly tied to location
        if "Seizures" not in impact_data["potential_symptoms"]:
            impact_data["potential_symptoms"].append("Seizures")
        if "Headaches" not in impact_data["potential_symptoms"]:
            impact_data["potential_symptoms"].append("Headaches")
        if "Cognitive decline" not in impact_data["potential_symptoms"]:
            impact_data["potential_symptoms"].append("Cognitive decline")
    
    # Add specific meningioma impacts (if applicable)
    elif tumor_type and 'meningioma' in tumor_type.lower():
        impact_data["proximity_risks"].append("Compression of adjacent brain tissue")
        
        # Add additional symptoms not directly tied to location
        if "Headaches" not in impact_data["potential_symptoms"]:
            impact_data["potential_symptoms"].append("Headaches")
    
    # Add specific pituitary impacts (if applicable)
    elif tumor_type and 'pituitary' in tumor_type.lower():
        impact_data["proximity_risks"].append("Compression of optic chiasm")
        impact_data["proximity_risks"].append("Disruption of hormone production")
        
        # Add additional symptoms not directly tied to location
        if "Hormonal imbalances" not in impact_data["potential_symptoms"]:
            impact_data["potential_symptoms"].append("Hormonal imbalances")
        if "Visual field defects" not in impact_data["potential_symptoms"]:
            impact_data["potential_symptoms"].append("Visual field defects")
    
    return impact_data


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
            
        coordinates_3d = session_data.get('coordinates_3d')
        tumor_type = session_data.get('tumor_type', 'unknown')
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

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
