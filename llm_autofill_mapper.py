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
    AZURE_OPENAI_API_KEY
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
You are an expert at mapping form data to PDF form fields. I need you to systematically map the provided JSON data to the correct form fields based on their proximity to OCR-detected text labels.

FORM FIELDS DETECTED:
{json.dumps(form_fields, indent=2)}

OCR TEXT LABELS FOUND:
{json.dumps(text_blocks, indent=2)}

JSON DATA TO FILL:
{json.dumps(json_data, indent=2)}

TASK:
1. Systematically go through ALL form fields and try to map them to JSON data
2. For each form field, find the closest OCR text label that describes what should be filled
3. Match the JSON data to the appropriate form field based on label text similarity and proximity
4. Be COMPREHENSIVE - map as many fields as possible, not just obvious ones
5. Return a mapping in this exact JSON format:

{{
  "field_mappings": [
    {{
      "field_id": "field_id_from_form_fields",
      "field_type": "letter_by_letter_filling_or_entire_text_filling",
      "label_text": "closest_ocr_text_label",
      "data_value": "value_from_json_data",
      "confidence": 0.95,
      "reasoning": "explanation_of_why_this_mapping_makes_sense"
    }}
  ],
  "unmapped_fields": [
    {{
      "field_id": "field_id",
      "reason": "why_no_mapping_was_found"
    }}
  ],
  "unused_data": [
    {{
      "key": "json_key",
      "value": "json_value",
      "reason": "why_this_data_wasnt_used"
    }}
  ]
}}

MAPPING STRATEGY:
1. **Patient Information**: Map patient_name, age_years, gender, date_of_birth, contact numbers
2. **Hospital Information**: Map hospital_name, hospital_contact_number, hospital_city, hospital_state
3. **Medical Information**: Map presenting_complaints, clinical_findings, duration_of_ailment_days, provisional_diagnosis
4. **Address Information**: Map address_line1, address_line2, city, state, pincode
5. **Insurance Information**: Map member_id, insurer_id, policy_holder, tpa
6. **Doctor Information**: Map treating_doctor_name, treating_doctor_contact
7. **Admission Details**: Map admission_date, admission_time, expected_days_stay, room_type

RULES:
- Use proximity (distance between field and label) as the primary factor
- Consider text similarity between labels and JSON keys (e.g., "Name of patient" → patient_name)
- For letter_by_letter_filling fields, use single characters or short values
- For entire_text_filling fields, use full text values
- Map at least 50-80% of available form fields
- Be systematic and thorough, not conservative
- Provide clear reasoning for each mapping

Return ONLY the JSON response, no other text.
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
                    
                    # Adjust text position (center in field)
                    text_x = x + width // 2
                    text_y = y + height // 2
                    
                    # Draw text on image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    color = (0, 0, 0)  # Black text
                    thickness = 1
                    
                    # Get text size to center it properly
                    (text_width, text_height), baseline = cv2.getTextSize(data_value, font, font_scale, thickness)
                    
                    # Adjust position to center text
                    text_x = text_x - text_width // 2
                    text_y = text_y + text_height // 2
                    
                    # Draw text on image
                    cv2.putText(image, data_value, (text_x, text_y), font, font_scale, color, thickness)
                    
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
            
            # Step 3: Detect form fields
            self.logger.info("📦 Step 2: Detecting form fields...")
            box_results = self.box_detector.process_pdf(pdf_path)
            if not box_results:
                self.logger.error("❌ Box detection failed")
                return None
            
            # Step 4: Create LLM prompt and get mappings
            self.logger.info("🤖 Step 3: Creating LLM mappings...")
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
            
            # Step 6: Save mapping results
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
