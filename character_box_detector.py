"""
Character Box Detector for PDF Form Filling
Detects individual character input boxes on PDF forms for letter-by-letter filling
"""

import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import logging
import requests
import base64
import json

class CharacterBoxDetector:
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def convert_pdf_to_image(self, pdf_path: str):
        """Convert PDF to OpenCV image"""
        self.logger.info("📄 Converting PDF to image")
        
        try:
            # Try pdf2image first
            images = convert_from_path(pdf_path, dpi=300)
            if images:
                # Convert PIL image to OpenCV format
                img_array = np.array(images[0])
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                self.logger.info("✅ PDF converted using pdf2image")
                return img_cv
        except Exception as e:
            self.logger.warning(f"pdf2image failed: {e}, trying PyMuPDF fallback")
        
        try:
            # Fallback using PyMuPDF
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)  # First page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            doc.close()
            self.logger.info("✅ PDF converted using PyMuPDF fallback")
            return img_cv
        except Exception as e:
            self.logger.error(f"Both PDF conversion methods failed: {e}")
            raise e
    
    def detect_all_input_fields(self, image):
        """
        Detect all types of input fields on the form
        Returns categorized list of input field coordinates
        """
        self.logger.info("🔍 Detecting all types of input fields")
        
        # Debug: Check image properties
        self.logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive threshold for better binary image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_fields = {
            'letter_by_letter_filling': [],
            'entire_text_filling': [],
        }
        
        height, width = image.shape[:2]
        
        self.logger.info(f"Found {len(contours)} contours in image")
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small or very large contours
            if (w < 8 or h < 8 or w > width * 0.8 or h > height * 0.3 or
                x < 5 or y < 5 or x + w > width - 5 or y + h > height - 5):
                continue
            
            # Check if it's likely an empty input field
            roi = binary[y:y+h, x:x+w]
            white_pixels = cv2.countNonZero(roi)
            total_pixels = w * h
            white_ratio = white_pixels / total_pixels
            
            # Input fields should be mostly empty (low white ratio)
            if white_ratio < 0.3:
                field_info = {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'center_x': x + w//2,
                    'center_y': y + h//2,
                    'area': w * h,
                    'aspect_ratio': w / h
                }
                
                # Categorize based on size and aspect ratio
                field_type = self._categorize_field(w, h, w/h)
                if field_type in all_fields:
                    field_info['type'] = field_type
                    all_fields[field_type].append(field_info)
        
        # Sort each category by position
        for field_type in all_fields:
            all_fields[field_type].sort(key=lambda box: (box['y'], box['x']))
        
        # Log results
        total_fields = sum(len(fields) for fields in all_fields.values())
        self.logger.info(f"📦 Found {total_fields} total input fields:")
        for field_type, fields in all_fields.items():
            if fields:
                self.logger.info(f"  - {field_type}: {len(fields)} fields")
        
        return all_fields
    
    def _categorize_field(self, width, height, aspect_ratio):
        """Categorize input field based on dimensions and aspect ratio"""
        
        # Letter by letter filling (small to medium rectangles for character input)
        if (12 <= width <= 60 and 10 <= height <= 30 and 0.5 <= aspect_ratio <= 3.0):
            return 'letter_by_letter_filling'
        
        # Entire text filling (large rectangles for full text)
        elif (width >= 80 and height >= 20 and aspect_ratio >= 1.5):
            return 'entire_text_filling'
        
        return None
    
    def save_visualization(self, image, all_fields, output_path="output/detected_all_field_types.png"):
        """Save visualization of all detected input fields with color coding by type"""
        os.makedirs('output', exist_ok=True)
        
        # Create visualization image
        vis_image = image.copy()
        
        # Define colors for different field types
        field_colors = {
            'letter_by_letter_filling': (180, 105, 255),     # Pink (BGR)
            'entire_text_filling': (0, 255, 255),    # Yellow
        }
        
        # Separate counters for each field type
        l_count = 1  # Counter for letter-by-letter (pink)
        s_count = 1  # Counter for entire text (yellow)
        
        for field_type, fields in all_fields.items():
            if not fields:
                continue
                
            color = field_colors.get(field_type, (128, 128, 128))  # Gray for unknown types
            
            for i, field in enumerate(fields):
                # Draw rectangle around the field
                cv2.rectangle(vis_image, 
                             (field['x'], field['y']), 
                             (field['x'] + field['width'], field['y'] + field['height']), 
                             color, 2)
                
                # Add field number and type with separate counters
                if field_type == 'letter_by_letter_filling':
                    label = f"L_{l_count}"
                    l_count += 1
                elif field_type == 'entire_text_filling':
                    label = f"S_{s_count}"
                    s_count += 1
                else:
                    label = f"FLD_{i+1}"
                
                cv2.putText(vis_image, label, 
                           (field['x'], field['y'] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Add size info for smaller fields
                if field['width'] < 100 and field['height'] < 50:
                    size_text = f"{field['width']}x{field['height']}"
                    cv2.putText(vis_image, size_text, 
                               (field['x'], field['y'] + field['height'] + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add legend
        legend_y = 30
        for field_type, color in field_colors.items():
            if all_fields[field_type]:  # Only show types that have fields
                cv2.rectangle(vis_image, (10, legend_y - 15), (25, legend_y - 5), color, -1)
                # Friendly legend names
                if field_type == 'letter_by_letter_filling':
                    friendly = 'Letter by Letter Filling'
                elif field_type == 'entire_text_filling':
                    friendly = 'Entire Text Filling'
                else:
                    friendly = field_type
                label = f"{friendly} ({len(all_fields[field_type])})"
                cv2.putText(vis_image, label, 
                           (30, legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                legend_y += 20
        
        # Save the visualization
        cv2.imwrite(output_path, vis_image)
        self.logger.info(f"📸 Saved input field visualization to {output_path}")
        
        return output_path
    
    def analyze_form_structure(self, all_fields):
        """Analyze the form structure and provide comprehensive field analysis"""
        self.logger.info("📊 Analyzing form structure")
        
        total_fields = sum(len(fields) for fields in all_fields.values())
        
        if total_fields == 0:
            return {}
        
        # Calculate statistics for each field type
        field_stats = {}
        for field_type, fields in all_fields.items():
            if fields:
                field_stats[field_type] = {
                    'count': len(fields),
                    'avg_width': sum(f['width'] for f in fields) / len(fields),
                    'avg_height': sum(f['height'] for f in fields) / len(fields),
                    'total_area': sum(f['area'] for f in fields)
                }
        
        # Grouping not required for number_fields and signature_boxes
        field_groups = []

  
        
        form_analysis = {
            'total_fields': total_fields,
            'field_types': field_stats,
            'field_groups': len(field_groups),
            'groups': field_groups,
            'field_distribution': {ft: len(fields) for ft, fields in all_fields.items() if fields}
        }
        
        self.logger.info(f"📋 Form analysis complete:")
        self.logger.info(f"  - Total fields: {total_fields}")
        self.logger.info(f"  - Field groups: {len(field_groups)}")
        for field_type, count in form_analysis['field_distribution'].items():
            self.logger.info(f"  - {field_type}: {count} fields")
        
        return form_analysis
    
    def process_pdf(self, pdf_path: str):
        """Main processing function to detect all input fields in PDF"""
        self.logger.info("🚀 Starting comprehensive input field detection")
        
        try:
            # Step 1: Convert PDF to image
            image = self.convert_pdf_to_image(pdf_path)
            
            # Step 2: Detect all types of input fields
            all_fields = self.detect_all_input_fields(image)
            
            # Step 3: Save visualization
            vis_path = self.save_visualization(image, all_fields)
            
            # Step 4: Analyze form structure
            form_analysis = self.analyze_form_structure(all_fields)
            
            # Step 5: Save results
            total_fields = sum(len(fields) for fields in all_fields.values())
            results = {
                'pdf_path': pdf_path,
                'total_fields': total_fields,
                'all_fields': all_fields,
                'form_analysis': form_analysis,
                'visualization_path': vis_path
            }
            
            self.logger.info(f"✅ Detection complete: {total_fields} total input fields found")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error processing PDF: {e}")
            raise e

def main():
    """Main function to test the comprehensive input field detector"""
    detector = CharacterBoxDetector()
    
    pdf_path = "input/Medi Assist.pdf"
    
    if not os.path.exists(pdf_path):
        print("❌ PDF file not found!")
        return
    
    try:
        results = detector.process_pdf(pdf_path)
        
        print(f"\n🎯 Comprehensive Detection Results:")
        print(f"📄 PDF: {results['pdf_path']}")
        print(f"📦 Total Input Fields: {results['total_fields']}")
        print(f"📸 Visualization: {results['visualization_path']}")
        
        # Show field type distribution
        print(f"\n📋 Field Type Distribution:")
        for field_type, count in results['form_analysis']['field_distribution'].items():
            print(f"  - {field_type}: {count} fields")
        
        # Show field groups
        print(f"\n📊 Field Groups: {results['form_analysis']['field_groups']}")
        
        # Show examples of each field type
        print(f"\n📝 Field Examples:")
        for field_type, fields in results['all_fields'].items():
            if fields:
                print(f"\n  {field_type.upper()}:")
                for i, field in enumerate(fields[:3]):  # Show first 3 of each type
                    print(f"    {i+1}. ({field['x']}, {field['y']}) size {field['width']}x{field['height']}")
                if len(fields) > 3:
                    print(f"    ... and {len(fields) - 3} more")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
