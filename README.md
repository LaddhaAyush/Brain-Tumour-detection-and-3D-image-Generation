# Brain Tumor Detection and 3D Visualization System

This project is an advanced brain tumor detection and visualization system that uses deep learning to detect brain tumors from MRI scans and provides an interactive 3D visualization of the detected tumors in a brain model. The system can classify tumors into different types (glioma, meningioma, pituitary) and provides detailed information about their location and potential impact.

## Features

- Brain tumor detection using YOLOv8
- Classification of tumors into different types
- Interactive 3D visualization of the brain with tumor locations
- Anatomically accurate tumor placement
- Detailed impact analysis based on tumor type and location
- Web-based interface for easy interaction
- Support for multiple tumor types:
  - Glioma
  - Meningioma
  - Pituitary
  - No tumor detection

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv myenv
myenv\Scripts\activate
```

For Linux/Mac:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── main2.py              # Main application file
├── fibonacciNet.py       # Neural network implementation
├── combine.py           # Utility functions
├── models/              # Contains the trained model
│   └── best.pt         # YOLOv8 model weights
├── static/             # Static files (CSS, JS, images)
├── templates/          # HTML templates
├── uploads/           # Temporary storage for uploaded images
└── requirements.txt    # Python dependencies
```

## Usage

1. Start the application:
```bash
python main2.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an MRI scan image through the web interface

4. The system will:
   - Detect any tumors in the image
   - Classify the tumor type
   - Generate a 3D visualization
   - Provide detailed information about the tumor's location and potential impact

## Demonstration

[Add your demonstration video link here]

## Technical Details

The system uses:
- Flask for the web framework
- YOLOv8 for tumor detection
- Three.js for 3D visualization
- OpenCV for image processing
- NumPy for numerical computations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 team for the detection model
- Three.js community for 3D visualization tools
- Medical imaging community for dataset and research

## Contact

[Your Name] - [Your Email]

Project Link: [https://github.com/yourusername/your-repo-name] 