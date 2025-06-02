# Brain Tumor Detection & Analysis System

This application is designed to detect and analyze brain tumors from MRI scans, providing comprehensive information, 3D visualization, and an AI-powered chat interface for medical queries.

## Features

- Brain tumor detection and classification using YOLOv8
- 3D visualization of tumor location in the brain
- Detailed information about the tumor type, location, and potential impacts
- AI-powered chat assistant with real-time web search integration
- Mixtral/Grok LLM integration for enhanced responses

## Setup Instructions

### Prerequisites

- Python 3.8+
- Flask
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- API keys for web search and Mixtral/Grok (optional but recommended)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Open `ai_bot.py` and `main2.py`
   - Replace the placeholder API key values with your actual API keys
   - Save the files

4. Start the application:
   ```
   python main2.py
   ```

5. Open the application in your browser at `http://localhost:5000`

### API Keys Setup

#### Web Search API (choose one)

1. **Google Custom Search API (recommended)**
   - Create a Custom Search Engine at [https://cse.google.com/cse/](https://cse.google.com/cse/)
   - Get API key at [https://developers.google.com/custom-search/v1/overview](https://developers.google.com/custom-search/v1/overview)
   - Update the values in `ai_bot.py` and `main2.py`:
     ```python
     os.environ.setdefault("SEARCH_API_KEY", "your_google_api_key_here")
     os.environ.setdefault("SEARCH_ENGINE_ID", "your_search_engine_id_here") 
     os.environ.setdefault("SEARCH_API_TYPE", "google")
     ```

2. **Bing Search API**
   - Get API key at [https://portal.azure.com/#create/microsoft.bingsearch](https://portal.azure.com/#create/microsoft.bingsearch)
   - Update the values in `ai_bot.py` and `main2.py`:
     ```python
     os.environ.setdefault("SEARCH_API_KEY", "your_bing_api_key_here")
     os.environ.setdefault("SEARCH_API_TYPE", "bing")
     ```

3. **SerpAPI**
   - Get API key at [https://serpapi.com/](https://serpapi.com/)
   - Update the values in `ai_bot.py` and `main2.py`:
     ```python
     os.environ.setdefault("SEARCH_API_KEY", "your_serpapi_key_here")
     os.environ.setdefault("SEARCH_API_TYPE", "serpapi")
     ```

#### Mixtral/Grok API (Optional)
- Get API key from Grok/xAI
- Update the value in `ai_bot.py` and `main2.py`:
  ```python
  os.environ.setdefault("GROK_API_KEY", "your_grok_api_key_here")
  ```

## Usage

1. Upload an MRI scan image on the home page
2. View detection results, including tumor type and confidence level
3. Explore the 3D visualization by clicking "View in 3D"
4. Access detailed tumor information on the dashboard
5. Use the AI chat feature to ask questions about the detected tumor

## AI Chat Features

The AI chat assistant provides information about:
- Tumor types, locations, and sizes
- Common symptoms and treatment options
- Prognosis and research developments
- References to medical sources for further reading

The AI bot performs real-time web searches for the most up-to-date information and, if configured, uses the Mixtral LLM to enhance response quality.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 team for the detection model
- Three.js community for 3D visualization tools
- Medical imaging community for dataset and research
