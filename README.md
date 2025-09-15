# PDF Form AutoFill System

An intelligent PDF form auto-fill system that combines OCR, computer vision, and LLM to automatically detect form fields and fill them with provided data.

## Features

- **Box Detection**: Uses OpenCV to detect input fields in PDF forms
- **OCR Text Extraction**: Extracts labels using OCR.space API
- **LLM Mapping**: Uses Azure OpenAI to intelligently map JSON data to form fields
- **Accurate Placement**: Fixed coordinate system for precise text placement
- **Multiple Field Types**: Supports letter-by-letter and entire text filling

## Setup

1. **Install Dependencies**:
```bash
pip install opencv-python pdf2image pytesseract PyMuPDF requests pillow
```

2. **Configure API Keys**:
   - Copy `config_template.py` to `config.py`
   - Add your API keys:
     - OCR.space API key
     - Azure OpenAI API key and endpoint

3. **Run the System**:
```bash
python llm_autofill_mapper.py
```

## How It Works

1. **PDF to Image**: Converts PDF to image for processing
2. **Field Detection**: Detects input boxes using OpenCV contour detection
3. **OCR Extraction**: Extracts text labels using OCR.space API
4. **LLM Mapping**: Uses Azure OpenAI to map JSON data to detected fields
5. **Text Placement**: Writes text on image using correct coordinates
6. **PDF Generation**: Converts filled image back to PDF

## Output

- `output/autofilled_form.pdf` - Final filled PDF
- `output/filled_image.png` - Filled image for verification
- `output/field_mappings.json` - Complete mapping details
- `output/detected_all_field_types.png` - Visualization of detected fields

## Configuration

Edit `config.py` to customize:
- API keys and endpoints
- File paths
- Processing settings (DPI, font size, etc.)

## Requirements

- Python 3.7+
- OpenCV
- PyMuPDF
- OCR.space API key
- Azure OpenAI API key
