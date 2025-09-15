"""
Configuration Template for PDF Form AutoFill System
Copy this file to config.py and add your actual API keys
"""

# OCR.space API Configuration
OCR_SPACE_API_KEY = "YOUR_OCR_SPACE_API_KEY_HERE"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://your-resource.cognitiveservices.azure.com/openai/deployments/your-deployment/chat/completions"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT = "your-deployment-name"
AZURE_OPENAI_API_KEY = "YOUR_AZURE_OPENAI_API_KEY_HERE"

# File Paths
INPUT_PDF_PATH = "input/Medi Assist.pdf"
INPUT_JSON_PATH = "input/B.json"
OUTPUT_DIR = "output"

# Processing Settings
DPI = 300
FONT_SIZE = 10
TEXT_COLOR = (0, 0, 0)  # Black text

# Feature Flags
# If True, the pipeline will send a single annotated image (with field IDs
# rendered on top of the form) to the vision-capable LLM for mapping.
# If False, it uses the text-based prompt with OCR labels and box coordinates.
USE_VISION_IMAGE_MAPPING = False
