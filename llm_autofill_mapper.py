"""
LLM-based AutoFill Mapper
Combines OCR text extraction, box detection, and LLM mapping to automatically fill PDF forms
"""

import os
import json
import logging
import requests
import base64
from character_box_detector import CharacterBoxDetector
from ocr_text_extractor import OCRTextExtractor
import fitz  # PyMuPDF
from config import (
    OCR_SPACE_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_API_VERSION, 
    AZURE_OPENAI_DEPLOYMENT, 
    AZURE_OPENAI_API_KEY,
    USE_VISION_IMAGE_MAPPING,
    ENABLE_VALIDATION_AGENT,
    VALIDATION_MAX_PASSES
)

class LLMAutoFillMapper:
    def __init__(self, llm_api_key=None, ocr_api_key=None):
        """
        Initialize LLM AutoFill Mapper
        
        Args:
            llm_api_key (str): LLM API key (optional, uses config if not provided)
            ocr_api_key (str): OCR.space API key (optional, uses config if not provided)
        """
        # Azure OpenAI Configuration
        self.azure_endpoint = AZURE_OPENAI_ENDPOINT
        self.azure_api_version = AZURE_OPENAI_API_VERSION
        self.azure_deployment = AZURE_OPENAI_DEPLOYMENT
        self.azure_api_key = llm_api_key or AZURE_OPENAI_API_KEY
        
        self.ocr_api_key = ocr_api_key or OCR_SPACE_API_KEY
        self.setup_logging()
        
        # Initialize components
        self.box_detector = CharacterBoxDetector()
        self.ocr_extractor = OCRTextExtractor(self.ocr_api_key)
        self.validation_enabled = ENABLE_VALIDATION_AGENT
        self.validation_max_passes = VALIDATION_MAX_PASSES
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('llm_autofill.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_json_data(self, json_path):
        """Load JSON data for form filling"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"✅ Loaded JSON data from {json_path}")
            return data
        except Exception as e:
            self.logger.error(f"❌ Failed to load JSON data: {e}")
            return None
    
    def create_llm_prompt(self, ocr_data, box_data, json_data):
        """
        Create LLM prompt for field mapping
        
        Args:
            ocr_data (dict): OCR extracted text with coordinates
            box_data (dict): Detected form fields with coordinates
            json_data (dict): Data to fill in the form
            
        Returns:
            str: Formatted LLM prompt
        """
        # Extract text blocks with coordinates
        text_blocks = []
        for block in ocr_data.get('text_blocks', []):
            text_blocks.append({
                'text': block['text'],
                'x': block['bbox']['x'],
                'y': block['bbox']['y'],
                'width': block['bbox']['width'],
                'height': block['bbox']['height']
            })
        
        # Extract form fields with coordinates
        form_fields = []
        for field_type, fields in box_data.get('all_fields', {}).items():
            for i, field in enumerate(fields):
                form_fields.append({
                    'id': f"{field_type}_{i+1}",
                    'type': field_type,
                    'x': field['x'],
                    'y': field['y'],
                    'width': field['width'],
                    'height': field['height']
                })
        
        prompt = f"""
You are a deterministic field mapper for a scanned PDF form. Given OCR label lines with coordinates, detected field boxes with coordinates, and a JSON data object, output a STRICT JSON mapping with no extra commentary.

INPUTS
- FORM_FIELDS (each has id, type, x, y, width, height):
{json.dumps(form_fields, indent=2)}
- OCR_LABELS (each has text, x, y, width, height):
{json.dumps(text_blocks, indent=2)}
- JSON_DATA:
{json.dumps(json_data, indent=2)}

OUTPUT FORMAT (return exactly this schema):
{
  "field_mappings": [
    {
      "field_id": "letter_by_letter_filling_1",
      "field_type": "letter_by_letter_filling" | "entire_text_filling",
      "label_text": "label text used for this mapping",
      "data_value": "value from JSON_DATA (string)",
      "confidence": 0.0-1.0 (number),
      "reasoning": "short justification"
    }
  ],
  "unmapped_fields": [
    {
      "field_id": "...",
      "reason": "e.g., no nearby label or no matching JSON key"
    }
  ],
  "unused_data": [
    {
      "key": "json key not used",
      "value": "stringified value",
      "reason": "no suitable field or conflicting labels"
    }
  ]
}

MAPPING RULES
1) Nearest-label rule: Compute distance from a field box center to label line boxes; prefer the closest label that semantically matches.
2) Directional bias: Prefer labels above or left of fields over below/right when distances are similar.
3) Semantic normalization: Normalize both label and key (lowercase, remove punctuation/whitespace). Use synonyms: name→patient_name, phone/telephone/mobile→patient_contact_number, pincode/zip→pincode, dob/date of birth→date_of_birth, age→age_years, sex→gender, city/town→city.
4) One-to-one preference: Avoid assigning the same JSON key to many fields unless clearly intended (e.g., repeated member_id boxes). If duplicate, still include each mapping with lowered confidence.
5) Field types: If type is letter_by_letter_filling, prefer compact values (codes, IDs, dates) and output the full value in data_value (the renderer will split if needed). If type is entire_text_filling, use full strings.
6) Confidence: 0.9+ when label text closely matches a JSON key and is near; 0.6-0.9 when approximate; <0.6 if weak.
7) Completeness: Attempt to map every field; if uncertain, include in unmapped_fields with a clear reason.

CONSTRAINTS
- Return ONLY raw JSON. No markdown, no code fences, no prose.
- Do not fabricate values. Only use values present in JSON_DATA.
"""
        return prompt
    
    def call_llm_api(self, prompt):
        """
        Call Azure OpenAI API for field mapping
        
        Args:
            prompt (str): The prompt to send to LLM
            
        Returns:
            dict: LLM response with field mappings
        """
        try:
            self.logger.info("🤖 Calling Azure OpenAI API for field mapping...")
            
            # Prepare the request payload for Azure OpenAI
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_api_key
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at mapping form data to PDF form fields. Analyze the provided form fields, OCR text labels, and JSON data to create accurate field mappings. Return only valid JSON in the exact format requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            # Make the API request
            response = requests.post(
                self.azure_endpoint,
                headers=headers,
                json=payload,
                params={"api-version": self.azure_api_version},
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from the response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse the JSON response
                try:
                    # Clean the response to extract JSON
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    field_mappings = json.loads(content)
                    self.logger.info("✅ Azure OpenAI mapping completed successfully")
                    return field_mappings
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
                    self.logger.error(f"Raw response: {content}")
                    return None
            else:
                self.logger.error("❌ No valid response from Azure OpenAI API")
                return None
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Azure OpenAI API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ LLM API call failed: {e}")
            return None

    def call_llm_api_with_image(self, prompt_text, image_path):
        """Call Azure image-capable LLM with an inline data URI image and instructions.

        The model must support image inputs. The response must be raw JSON in the
        same schema our text-only call uses.
        """
        try:
            self.logger.info("🖼️ Calling Azure OpenAI with annotated image for mapping...")

            # Load and base64-encode the image as data URI
            with open(image_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{b64}"

            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_api_key
            }

            messages = [
                {"role": "system", "content": "You are a deterministic field mapper. Return only valid JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ]

            payload = {
                "messages": messages,
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9
            }

            response = requests.post(
                self.azure_endpoint,
                headers=headers,
                json=payload,
                params={"api-version": self.azure_api_version},
                timeout=90
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                return json.loads(content)

            self.logger.error("❌ No valid response from Azure OpenAI vision API")
            return None

        except Exception as e:
            self.logger.error(f"❌ Vision LLM API call failed: {e}")
            return None
    
    def fill_image_with_mappings(self, image_path, field_mappings, box_results, output_path="output/filled_image.png"):
        """
        Fill image with mapped data using image coordinates
        
        Args:
            image_path (str): Path to the image file
            field_mappings (list): List of field mappings from LLM
            box_results (dict): Box detection results with coordinates
            output_path (str): Output path for filled image
            
        Returns:
            str: Path to filled image
        """
        try:
            self.logger.info("📝 Filling image with mapped data...")
            
            # Load the image
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"❌ Could not load image: {image_path}")
                return None
            
            # Helper: compute adaptive font scale to fit inside field box with padding
            def compute_font_scale_to_fit(text, box_width, box_height, font, thickness):
                if not text:
                    return 0.4
                # Base size at scale=1.0
                (base_width, base_height), base_baseline = cv2.getTextSize(text, font, 1.0, thickness)
                if base_width == 0 or base_height == 0:
                    return 0.4
                # Use padding
                usable_width = max(1, int(box_width * 0.9))
                usable_height = max(1, int(box_height * 0.7))
                scale_w = usable_width / base_width
                scale_h = usable_height / base_height
                scale = max(0.3, min(2.0, min(scale_w, scale_h)))
                return float(scale)

            # Fill each mapped field
            filled_count = 0
            for mapping in field_mappings:
                field_id = mapping['field_id']
                data_value = mapping['data_value']
                field_type = mapping['field_type']
                
                # Get the actual field coordinates from box detection results
                field_coords = self.get_field_coordinates(field_id, box_results)
                
                if field_coords:
                    # Use image coordinates directly
                    x, y, width, height = field_coords
                    
                    # Common drawing config
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = 1
                    color = (0, 0, 0)  # Black text

                    # Special handling for letter-by-letter: distribute characters across contiguous boxes in the same row
                    if field_type == 'letter_by_letter_filling':
                        # Prepare the sequence of target boxes in the same row (including this one and boxes to the right)
                        all_letter_boxes = box_results.get('all_fields', {}).get('letter_by_letter_filling', [])
                        # Find this box index in the list
                        start_index = None
                        for idx, fld in enumerate(all_letter_boxes):
                            if fld['x'] == x and fld['y'] == y and fld['width'] == width and fld['height'] == height:
                                start_index = idx
                                break
                        
                        # Build row sequence starting at this index: same-line boxes within vertical tolerance
                        row_sequence = []
                        if start_index is not None:
                            base_y = y
                            vertical_tol = max(2, int(height * 0.6))
                            for fld in all_letter_boxes[start_index:]:
                                if abs(fld['y'] - base_y) <= vertical_tol:
                                    row_sequence.append((fld['x'], fld['y'], fld['width'], fld['height']))
                                else:
                                    break
                        else:
                            row_sequence = [(x, y, width, height)]

                        # Filter data value to characters suitable for individual boxes (alnum only, preserve case for letters uppercase)
                        raw_text = str(data_value)
                        filtered_chars = []
                        for ch in raw_text:
                            if ch.isalnum():
                                filtered_chars.append(ch.upper())
                        
                        # Draw per character in each box
                        chars_drawn = 0
                        max_chars = min(len(filtered_chars), len(row_sequence))
                        for i in range(max_chars):
                            ch = filtered_chars[i]
                            bx, by, bw, bh = row_sequence[i]
                            ch_scale = compute_font_scale_to_fit(ch, bw, bh, font, thickness)
                            (tw, th), bl = cv2.getTextSize(ch, font, ch_scale, thickness)
                            # Center the character in its box
                            cx = bx + (bw - tw) // 2
                            cy = by + (bh + th) // 2 - max(0, bl // 2)
                            cv2.putText(image, ch, (cx, cy), font, ch_scale, color, thickness, lineType=cv2.LINE_AA)
                            chars_drawn += 1

                        filled_count += chars_drawn
                        self.logger.info(f"✅ Filled {chars_drawn} character box(es) starting at {field_id}")

                    else:
                        # Entire text: left-align with padding, vertically center
                        font_scale = compute_font_scale_to_fit(str(data_value), width, height, font, thickness)
                        (text_width, text_height), baseline = cv2.getTextSize(str(data_value), font, font_scale, thickness)
                        pad_x = max(1, int(width * 0.05))
                        text_x = x + pad_x
                        text_y = y + (height + text_height) // 2 - max(0, baseline // 2)
                        max_text_x = x + width - text_width - 1
                        if text_x > max_text_x:
                            text_x = max(x + 1, max_text_x)
                        cv2.putText(image, str(data_value), (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
                        filled_count += 1
                        self.logger.info(f"✅ Filled field {field_id} with '{data_value}' at ({text_x}, {text_y})")
                else:
                    self.logger.warning(f"⚠️ Could not find coordinates for field {field_id}")
            
            # Save filled image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            
            self.logger.info(f"✅ Image filled successfully: {filled_count} fields filled")
            self.logger.info(f"📄 Saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fill image: {e}")
            return None
    
    def convert_image_to_pdf(self, image_path, output_path="output/autofilled_form.pdf"):
        """
        Convert filled image back to PDF
        
        Args:
            image_path (str): Path to the filled image
            output_path (str): Output path for PDF
            
        Returns:
            str: Path to PDF
        """
        try:
            self.logger.info("🔄 Converting filled image to PDF...")
            
            # Load image
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"❌ Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to PDF
            pil_image.save(output_path, "PDF", resolution=300.0)
            
            self.logger.info(f"✅ Image converted to PDF: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert image to PDF: {e}")
            return None
    
    def fill_pdf_with_mappings(self, pdf_path, field_mappings, box_results, output_path="output/autofilled_form.pdf"):
        """
        Fill PDF with mapped data by first filling image then converting to PDF
        
        Args:
            pdf_path (str): Path to original PDF
            field_mappings (list): List of field mappings from LLM
            box_results (dict): Box detection results with coordinates
            output_path (str): Output path for filled PDF
            
        Returns:
            str: Path to filled PDF
        """
        try:
            self.logger.info("📝 Filling PDF with mapped data using image coordinates...")
            
            # Get the image path from box detection results
            image_path = "output/page_1.png"  # This should be the same image used for box detection
            
            if not os.path.exists(image_path):
                self.logger.error(f"❌ Image file not found: {image_path}")
                return None
            
            # Step 1: Fill the image with text
            filled_image_path = self.fill_image_with_mappings(
                image_path, 
                field_mappings, 
                box_results, 
                "output/filled_image.png"
            )
            
            if not filled_image_path:
                self.logger.error("❌ Failed to fill image")
                return None
            
            # Step 2: Convert filled image to PDF
            filled_pdf_path = self.convert_image_to_pdf(filled_image_path, output_path)
            
            if not filled_pdf_path:
                self.logger.error("❌ Failed to convert image to PDF")
                return None
            
            self.logger.info(f"✅ PDF filled successfully using image coordinates")
            self.logger.info(f"📄 Saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fill PDF: {e}")
            return None
    
    def get_field_coordinates(self, field_id, box_results):
        """
        Get coordinates for a field ID from the actual detected fields
        
        Args:
            field_id (str): Field ID like "letter_by_letter_filling_1"
            box_results (dict): Results from box detection
            
        Returns:
            tuple: (x, y, width, height) coordinates
        """
        try:
            # Parse field_id to get type and index
            parts = field_id.split('_')
            if len(parts) >= 3:
                field_type = '_'.join(parts[:-1])  # e.g., "letter_by_letter_filling"
                field_index = int(parts[-1]) - 1   # Convert to 0-based index
                
                # Get the field from box_results
                all_fields = box_results.get('all_fields', {})
                fields_of_type = all_fields.get(field_type, [])
                
                if 0 <= field_index < len(fields_of_type):
                    field = fields_of_type[field_index]
                    return (field['x'], field['y'], field['width'], field['height'])
            
            self.logger.warning(f"⚠️ Could not find coordinates for field: {field_id}")
            return None
            
        except (ValueError, KeyError, IndexError) as e:
            self.logger.error(f"❌ Error getting coordinates for {field_id}: {e}")
            return None
    
    def run_complete_pipeline(self, pdf_path, json_path, output_dir="output"):
        """
        Run the complete autofill pipeline
        
        Args:
            pdf_path (str): Path to PDF file
            json_path (str): Path to JSON data file
            output_dir (str): Output directory
            
        Returns:
            dict: Pipeline results
        """
        try:
            self.logger.info("🚀 Starting complete autofill pipeline...")
            
            # Step 1: Load JSON data
            json_data = self.load_json_data(json_path)
            if not json_data:
                return None
            
            # Step 2: Extract OCR text
            self.logger.info("📄 Step 1: Extracting OCR text...")
            ocr_data = self.ocr_extractor.extract_text_from_pdf(pdf_path, output_dir)
            if not ocr_data:
                self.logger.error("❌ OCR extraction failed")
                return None
            
            # Step 3: Detect form fields (use same image as OCR to avoid scale mismatch)
            self.logger.info("📦 Step 2: Detecting form fields...")
            image_path = f"{output_dir}/page_1.png"
            if not os.path.exists(image_path):
                # Fallback to PDF conversion if image missing
                self.logger.warning(f"⚠️ Expected OCR image not found at {image_path}, falling back to PDF conversion")
                box_results = self.box_detector.process_pdf(pdf_path)
            else:
                box_results = self.box_detector.process_image_path(image_path)
            if not box_results:
                self.logger.error("❌ Box detection failed")
                return None
            
            # Step 4: Create LLM mappings (vision mode or text mode)
            self.logger.info("🤖 Step 3: Creating LLM mappings...")
            if USE_VISION_IMAGE_MAPPING:
                # Generate annotated image with IDs for the LLM to analyze
                annotated_path = self.box_detector.save_annotated_image_with_ids(
                    cv2.imread(image_path), box_results.get('all_fields', {}), f"{output_dir}/annotated_fields.png"
                )
                prompt_text = (
                    "You are given a scanned form image with rectangles annotated and labeled with field IDs "
                    "(like letter_by_letter_filling_1, entire_text_filling_2). Read the labels printed on the form "
                    "and determine which JSON values should go into which field IDs. Return only JSON in the schema: "
                    "{\n  \"field_mappings\": [ { \"field_id\": ..., \"field_type\": ..., \"label_text\": ..., \"data_value\": ..., \"confidence\": ..., \"reasoning\": ... } ], \n  \"unmapped_fields\": [ ... ], \n  \"unused_data\": [ ... ]\n}."
                )
                field_mappings = self.call_llm_api_with_image(prompt_text, annotated_path)
            else:
                prompt = self.create_llm_prompt(ocr_data, box_results, json_data)
                field_mappings = self.call_llm_api(prompt)
            if not field_mappings:
                self.logger.error("❌ LLM mapping failed")
                return None
            
            # Step 5: Fill PDF
            self.logger.info("📝 Step 4: Filling PDF...")
            filled_pdf_path = self.fill_pdf_with_mappings(
                pdf_path, 
                field_mappings.get('field_mappings', []),
                box_results,
                f"{output_dir}/autofilled_form.pdf"
            )
            
            # Step 6: Optional validation loop
            if self.validation_enabled:
                self.logger.info("🔎 Running validation agent...")
                corrections = self.validate_filled_output(
                    image_path=f"{output_dir}/filled_image.png",
                    annotated_path=f"{output_dir}/annotated_fields.png",
                    json_data=json_data,
                    llm_prompt_mode='vision' if USE_VISION_IMAGE_MAPPING else 'text'
                )
                passes = 0
                while corrections and passes < self.validation_max_passes:
                    self.logger.info(f"♻️ Applying {len(corrections)} correction(s) (pass {passes+1})")
                    # Apply corrections to mappings, then re-fill
                    updated_field_mappings = self.apply_corrections(field_mappings.get('field_mappings', []), corrections)
                    # Re-fill image and PDF
                    self.fill_image_with_mappings(
                        image_path=f"{output_dir}/page_1.png",
                        field_mappings=updated_field_mappings,
                        box_results=box_results,
                        output_path=f"{output_dir}/filled_image.png"
                    )
                    self.convert_image_to_pdf(f"{output_dir}/filled_image.png", f"{output_dir}/autofilled_form.pdf")
                    # Ask validator again (optional one more pass)
                    corrections = self.validate_filled_output(
                        image_path=f"{output_dir}/filled_image.png",
                        annotated_path=f"{output_dir}/annotated_fields.png",
                        json_data=json_data,
                        llm_prompt_mode='vision' if USE_VISION_IMAGE_MAPPING else 'text'
                    )
                    passes += 1

            # Step 7: Save mapping results
            mapping_results_path = f"{output_dir}/field_mappings.json"
            with open(mapping_results_path, 'w', encoding='utf-8') as f:
                json.dump(field_mappings, f, indent=2, ensure_ascii=False)
            
            results = {
                'success': True,
                'ocr_data': ocr_data,
                'box_detection': box_results,
                'field_mappings': field_mappings,
                'filled_pdf': filled_pdf_path,
                'mapping_results': mapping_results_path
            }
            
            self.logger.info("✅ Complete autofill pipeline finished successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            return None

    def validate_filled_output(self, image_path, annotated_path, json_data, llm_prompt_mode='text'):
        """Ask LLM to validate the filled form. Returns list of corrections.

        Output format:
        [ { "field_id": "...", "new_value": "...", "reason": "..." } ]
        """
        try:
            if llm_prompt_mode == 'vision' and os.path.exists(annotated_path):
                with open(image_path, 'rb') as f1:
                    b64_filled = base64.b64encode(f1.read()).decode('utf-8')
                with open(annotated_path, 'rb') as f2:
                    b64_annot = base64.b64encode(f2.read()).decode('utf-8')
                prompt_text = (
                    "You are a strict validator. Compare the annotated form image (with field IDs) and "
                    "the filled image to the provided JSON data. Identify any fields that appear to be "
                    "incorrectly filled (wrong value, wrong field). Return ONLY a JSON array of corrections: "
                    "[{\\"field_id\\":\\"...\\", \\"new_value\\":\\"...\\", \\"reason\\":\\"...\\"}]. If everything is correct, return []."
                )
                return self.call_llm_api_with_images_for_validation(prompt_text, b64_annot, b64_filled, json_data)
            else:
                # Text mode: provide mappings and OCR labels summary
                prompt = {
                    "instruction": "Validate mapped values against JSON. Return corrections array only.",
                    "json_data": json_data
                }
                result = self.call_llm_api(json.dumps(prompt))
                if isinstance(result, list):
                    return result
                return []
        except Exception:
            return []

    def call_llm_api_with_images_for_validation(self, prompt_text, b64_annot, b64_filled, json_data):
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.azure_api_key
            }
            messages = [
                {"role": "system", "content": "You are a strict validator. Return only a JSON array."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_annot}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_filled}"}},
                        {"type": "text", "text": json.dumps(json_data)}
                    ]
                }
            ]
            payload = {"messages": messages, "max_tokens": 1500, "temperature": 0.0}
            response = requests.post(
                self.azure_endpoint,
                headers=headers,
                json=payload,
                params={"api-version": self.azure_api_version},
                timeout=90
            )
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                arr = json.loads(content)
                if isinstance(arr, list):
                    return arr
            return []
        except Exception:
            return []

    def apply_corrections(self, field_mappings_list, corrections):
        """Merge corrections into existing field mappings list and return updated list."""
        id_to_mapping = {m['field_id']: m for m in field_mappings_list}
        for c in corrections:
            fid = c.get('field_id')
            new_val = c.get('new_value')
            reason = c.get('reason')
            if fid in id_to_mapping and new_val is not None:
                id_to_mapping[fid]['data_value'] = str(new_val)
                id_to_mapping[fid]['reasoning'] = f"auto-corrected: {reason}"
                id_to_mapping[fid]['confidence'] = min(0.95, float(id_to_mapping[fid].get('confidence', 0.8)) + 0.05)
        return list(id_to_mapping.values())
    
    def print_pipeline_summary(self, results):
        """Print a summary of the pipeline results"""
        if not results:
            print("❌ No results to display")
            return
        
        print("\n" + "="*70)
        print("📊 LLM AUTOFILL PIPELINE SUMMARY")
        print("="*70)
        
        # OCR Summary
        ocr_data = results.get('ocr_data', {})
        print(f"📝 OCR Results:")
        print(f"   - Text blocks found: {len(ocr_data.get('text_blocks', []))}")
        print(f"   - Full text length: {len(ocr_data.get('full_text', ''))} characters")
        
        # Box Detection Summary
        box_data = results.get('box_detection', {})
        all_fields = box_data.get('all_fields', {})
        total_fields = sum(len(fields) for fields in all_fields.values())
        print(f"\n📦 Box Detection Results:")
        print(f"   - Total fields detected: {total_fields}")
        for field_type, fields in all_fields.items():
            print(f"   - {field_type}: {len(fields)} fields")
        
        # LLM Mapping Summary
        mappings = results.get('field_mappings', {})
        field_mappings = mappings.get('field_mappings', [])
        unmapped = mappings.get('unmapped_fields', [])
        unused = mappings.get('unused_data', [])
        
        print(f"\n🤖 LLM Mapping Results:")
        print(f"   - Successfully mapped: {len(field_mappings)} fields")
        print(f"   - Unmapped fields: {len(unmapped)}")
        print(f"   - Unused data items: {len(unused)}")
        
        # Show some mappings
        if field_mappings:
            print(f"\n📋 Sample Mappings:")
            for i, mapping in enumerate(field_mappings[:5]):
                print(f"   {i+1}. {mapping['field_id']} → '{mapping['data_value']}' "
                      f"(conf: {mapping['confidence']:.2f})")
        
        print(f"\n📄 Output Files:")
        print(f"   - Filled PDF: {results.get('filled_pdf', 'N/A')}")
        print(f"   - Mapping results: {results.get('mapping_results', 'N/A')}")

def main():
    """Main function to run the complete autofill pipeline"""
    try:
        # Initialize mapper
        mapper = LLMAutoFillMapper()
        
        # File paths
        from config import INPUT_PDF_PATH, INPUT_JSON_PATH
        pdf_path = INPUT_PDF_PATH
        json_path = INPUT_JSON_PATH
        
        # Check if files exist
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return
        
        if not os.path.exists(json_path):
            print(f"❌ JSON file not found: {json_path}")
            return
        
        print("🚀 Starting LLM AutoFill Pipeline...")
        print(f"📄 PDF: {pdf_path}")
        print(f"📋 JSON: {json_path}")
        
        # Run complete pipeline
        results = mapper.run_complete_pipeline(pdf_path, json_path)
        
        if results:
            # Print summary
            mapper.print_pipeline_summary(results)
            
            print(f"\n✅ LLM AutoFill pipeline completed successfully!")
            print(f"📁 All results saved in: output/")
        else:
            print("❌ LLM AutoFill pipeline failed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
