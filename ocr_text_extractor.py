"""
OCR Text Extractor using OCR.space API
Extracts text from PDF pages using OCR.space cloud-based OCR service
"""

import os
import requests
import base64
import json
import logging
from pdf2image import convert_from_path
import fitz  # PyMuPDF

class OCRTextExtractor:
    def __init__(self, api_key="K81634588988957"):
        """
        Initialize OCR Text Extractor
        
        Args:
            api_key (str): OCR.space API key (free tier key provided)
        """
        self.api_key = api_key
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('ocr_extraction.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def convert_pdf_to_image(self, pdf_path, output_dir="output", page_num=0):
        """
        Convert PDF page to image
        
        Args:
            pdf_path (str): Path to PDF file
            output_dir (str): Output directory
            page_num (int): Page number to convert (0-indexed)
            
        Returns:
            str: Path to converted image file
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Try pdf2image first
            try:
                self.logger.info(f"📄 Converting PDF page {page_num + 1} to image using pdf2image...")
                pages = convert_from_path(pdf_path, dpi=300, first_page=page_num + 1, last_page=page_num + 1)
                
                if pages:
                    image_path = f"{output_dir}/page_{page_num + 1}.png"
                    pages[0].save(image_path, "PNG")
                    self.logger.info(f"✅ PDF converted to image: {image_path}")
                    return image_path
                else:
                    raise Exception("No pages returned from pdf2image")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ pdf2image failed: {e}")
                self.logger.info("🔄 Trying PyMuPDF fallback...")
                
                # Fallback to PyMuPDF
                doc = fitz.open(pdf_path)
                if page_num >= len(doc):
                    raise Exception(f"Page {page_num + 1} not found in PDF")
                
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                image_path = f"{output_dir}/page_{page_num + 1}.png"
                pix.save(image_path)
                doc.close()
                
                self.logger.info(f"✅ PDF converted to image using PyMuPDF: {image_path}")
                return image_path
                
        except Exception as e:
            self.logger.error(f"❌ Failed to convert PDF to image: {e}")
            return None
    
    def extract_text_with_ocr_space(self, image_path):
        """
        Extract text from image using OCR.space API
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: OCR results with text and confidence scores
        """
        try:
            self.logger.info(f"🔍 Extracting text from {image_path} using OCR.space API...")
            
            # Read image and convert to base64
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # OCR.space API endpoint
            url = "https://api.ocr.space/parse/image"
            
            # API parameters
            payload = {
                'apikey': self.api_key,
                'language': 'eng',
                'isOverlayRequired': True,
                'filetype': 'PNG',
                'base64Image': f'data:image/png;base64,{image_data}'
            }
            
            # Make API request
            response = requests.post(url, data=payload, timeout=30)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            if result.get('IsErroredOnProcessing', False):
                error_msg = result.get('ErrorMessage', 'Unknown OCR error')
                self.logger.error(f"❌ OCR.space API error: {error_msg}")
                return None
            
            # Extract text results
            parsed_results = result.get('ParsedResults', [])
            if not parsed_results:
                self.logger.warning("⚠️ No text found in OCR results")
                return None
            
            # Process results
            ocr_data = {
                'full_text': '',
                'text_blocks': [],
                'confidence': 0,
                'raw_response': result
            }
            
            all_text = []
            total_confidence = 0
            block_count = 0
            
            for parsed_result in parsed_results:
                # Get full text
                full_text = parsed_result.get('ParsedText', '').strip()
                if full_text:
                    all_text.append(full_text)
                
                # Get individual text blocks with coordinates
                overlay = parsed_result.get('TextOverlay', {})
                lines = overlay.get('Lines', [])
                
                for line in lines:
                    line_text = line.get('LineText', '').strip()
                    if line_text:
                        words = line.get('Words', [])
                        if words:
                            # Get bounding box of the line
                            first_word = words[0]
                            last_word = words[-1]
                            
                            block_data = {
                                'text': line_text,
                                'confidence': line.get('MinConfidence', 0),
                                'bbox': {
                                    'x': first_word.get('Left', 0),
                                    'y': first_word.get('Top', 0),
                                    'width': last_word.get('Left', 0) + last_word.get('Width', 0) - first_word.get('Left', 0),
                                    'height': first_word.get('Height', 0)
                                },
                                'words': words
                            }
                            ocr_data['text_blocks'].append(block_data)
                            total_confidence += block_data['confidence']
                            block_count += 1
            
            # Calculate average confidence
            if block_count > 0:
                ocr_data['confidence'] = total_confidence / block_count
            
            # Combine all text
            ocr_data['full_text'] = '\n'.join(all_text)
            
            self.logger.info(f"✅ OCR extraction successful: {len(ocr_data['text_blocks'])} text blocks found")
            self.logger.info(f"📝 Extracted text preview: {ocr_data['full_text'][:200]}...")
            
            return ocr_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ OCR.space API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ OCR extraction failed: {e}")
            return None
    
    def save_ocr_results(self, ocr_data, output_path="output/ocr_results.json"):
        """Save OCR results to JSON file"""
        if not ocr_data:
            self.logger.warning("⚠️ No OCR data to save")
            return None
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for JSON serialization
        save_data = {
            'full_text': ocr_data['full_text'],
            'confidence': ocr_data['confidence'],
            'text_blocks': []
        }
        
        # Add text blocks (remove raw_response to keep file size manageable)
        for block in ocr_data['text_blocks']:
            save_data['text_blocks'].append({
                'text': block['text'],
                'confidence': block['confidence'],
                'bbox': block['bbox']
            })
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 OCR results saved to {output_path}")
        return output_path
    
    def extract_text_from_pdf(self, pdf_path, output_dir="output", page_num=0):
        """
        Complete OCR extraction pipeline from PDF
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Output directory for results
            page_num (int): Page number to process (0-indexed)
            
        Returns:
            dict: OCR results
        """
        self.logger.info("🔍 Starting OCR text extraction pipeline...")
        
        # Step 1: Convert PDF to image
        image_path = self.convert_pdf_to_image(pdf_path, output_dir, page_num)
        if not image_path:
            self.logger.error("❌ Failed to convert PDF to image")
            return None
        
        # Step 2: Extract text using OCR.space API
        ocr_data = self.extract_text_with_ocr_space(image_path)
        if not ocr_data:
            self.logger.error("❌ Failed to extract text with OCR")
            return None
        
        # Step 3: Save OCR results
        ocr_json_path = self.save_ocr_results(ocr_data, f"{output_dir}/ocr_results.json")
        
        self.logger.info("✅ OCR extraction pipeline completed successfully!")
        return ocr_data
    
    def print_ocr_summary(self, ocr_data):
        """Print a summary of OCR results"""
        if not ocr_data:
            print("❌ No OCR data available")
            return
        
        print("\n" + "="*60)
        print("📊 OCR EXTRACTION SUMMARY")
        print("="*60)
        
        print(f"📝 Total Text Blocks: {len(ocr_data['text_blocks'])}")
        print(f"🎯 Average Confidence: {ocr_data['confidence']:.1f}%")
        print(f"📄 Full Text Length: {len(ocr_data['full_text'])} characters")
        
        print(f"\n📋 EXTRACTED TEXT:")
        print("-" * 40)
        print(ocr_data['full_text'])
        
        print(f"\n📍 TEXT BLOCKS WITH COORDINATES:")
        print("-" * 40)
        for i, block in enumerate(ocr_data['text_blocks'][:10]):  # Show first 10 blocks
            bbox = block['bbox']
            print(f"{i+1:2d}. '{block['text']}' "
                  f"at ({bbox['x']}, {bbox['y']}) "
                  f"size {bbox['width']}x{bbox['height']} "
                  f"conf: {block['confidence']:.1f}%")
        
        if len(ocr_data['text_blocks']) > 10:
            print(f"    ... and {len(ocr_data['text_blocks']) - 10} more blocks")

def main():
    """Main function to run OCR extraction"""
    try:
        # Initialize OCR extractor
        extractor = OCRTextExtractor()
        
        # PDF file path
        pdf_path = "input/Medi Assist.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return
        
        print("🚀 Starting OCR Text Extraction...")
        print(f"📄 Processing: {pdf_path}")
        
        # Extract text from first page
        ocr_results = extractor.extract_text_from_pdf(pdf_path, page_num=0)
        
        if ocr_results:
            # Print summary
            extractor.print_ocr_summary(ocr_results)
            
            print(f"\n✅ OCR extraction completed successfully!")
            print(f"📁 Results saved in: output/")
            print(f"📄 Image: output/page_1.png")
            print(f"📋 JSON: output/ocr_results.json")
        else:
            print("❌ OCR extraction failed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
