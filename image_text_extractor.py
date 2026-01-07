"""
Image Text Extractor
Extracts text from images using Pillow, mss, and pytesseract
Creates a clean JSON file with extracted text and metadata
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image
import pytesseract
import mss

# Configure pytesseract path (if needed on Windows/Mac)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Uncomment and adjust if needed


class ImageTextExtractor:
    """Extract text from images using OCR."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the text extractor.
        
        Args:
            output_dir: Directory to save JSON output files (default: 'Extracted_texts')
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent / "Extracted_texts"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extract text from a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Open image using Pillow
            image = Image.open(image_path)
            
            # Get image metadata
            width, height = image.size
            format_name = image.format
            mode = image.mode
            
            # Extract text using pytesseract
            # Use multiple OCR modes for better accuracy
            text_default = pytesseract.image_to_string(image, lang='eng')
            text_data = pytesseract.image_to_data(image, lang='eng', output_type=pytesseract.Output.DICT)
            
            # Extract text with bounding boxes for structured data
            text_boxes = []
            n_boxes = len(text_data['text'])
            for i in range(n_boxes):
                if int(text_data['conf'][i]) > 0:  # Confidence > 0
                    text_boxes.append({
                        'text': text_data['text'][i],
                        'confidence': int(text_data['conf'][i]),
                        'left': text_data['left'][i],
                        'top': text_data['top'][i],
                        'width': text_data['width'][i],
                        'height': text_data['height'][i],
                        'level': text_data['level'][i],
                        'page_num': text_data['page_num'][i],
                        'block_num': text_data['block_num'][i],
                        'par_num': text_data['par_num'][i],
                        'line_num': text_data['line_num'][i],
                        'word_num': text_data['word_num'][i]
                    })
            
            # Calculate average confidence
            confidences = [box['confidence'] for box in text_boxes if box['confidence'] > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Create result dictionary
            result = {
                'source_file': str(image_path.name),
                'source_path': str(image_path.absolute()),
                'extraction_timestamp': datetime.now().isoformat(),
                'image_metadata': {
                    'width': width,
                    'height': height,
                    'format': format_name,
                    'mode': mode,
                    'file_size_bytes': image_path.stat().st_size
                },
                'extracted_text': {
                    'full_text': text_default.strip(),
                    'word_count': len(text_default.split()),
                    'character_count': len(text_default),
                    'average_confidence': round(avg_confidence, 2),
                    'text_boxes': text_boxes,
                    'text_blocks_count': len([b for b in text_boxes if b['level'] == 2]),  # Block level
                    'text_lines_count': len([b for b in text_boxes if b['level'] == 3]),   # Line level
                    'text_words_count': len([b for b in text_boxes if b['level'] == 5])    # Word level
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'source_file': str(image_path.name),
                'source_path': str(image_path.absolute()),
                'extraction_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'extracted_text': None
            }
    
    def extract_text_from_screenshot(self, screenshot_path: str) -> Dict:
        """
        Extract text from a screenshot file.
        Alias for extract_text_from_image for clarity.
        
        Args:
            screenshot_path: Path to the screenshot file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        return self.extract_text_from_image(screenshot_path)
    
    def extract_text_from_directory(self, directory: str, output_filename: Optional[str] = None) -> Dict:
        """
        Extract text from all images in a directory.
        
        Args:
            directory: Directory containing image files
            output_filename: Name for the output JSON file (default: auto-generated)
            
        Returns:
            Dictionary with extraction results for all images
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Supported image formats
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
        
        # Find all image files
        image_files = [
            f for f in directory.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in directory: {directory}")
        
        results = {
            'extraction_timestamp': datetime.now().isoformat(),
            'source_directory': str(directory.absolute()),
            'total_images': len(image_files),
            'processed_images': 0,
            'failed_images': 0,
            'extractions': []
        }
        
        # Process each image
        for image_file in image_files:
            try:
                extraction_result = self.extract_text_from_image(image_file)
                results['extractions'].append(extraction_result)
                results['processed_images'] += 1
                print(f"✓ Processed: {image_file.name}")
            except Exception as e:
                error_result = {
                    'source_file': str(image_file.name),
                    'source_path': str(image_file.absolute()),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'extracted_text': None
                }
                results['extractions'].append(error_result)
                results['failed_images'] += 1
                print(f"✗ Failed: {image_file.name} - {e}")
        
        # Save to JSON file
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"extracted_texts_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        self.save_to_json(results, output_path)
        
        print(f"\n✅ Extraction complete!")
        print(f"   Processed: {results['processed_images']}/{results['total_images']}")
        print(f"   Failed: {results['failed_images']}")
        print(f"   Output saved to: {output_path}")
        
        return results
    
    def save_to_json(self, data: Dict, output_path: Path) -> None:
        """
        Save extraction results to a JSON file.
        
        Args:
            data: Dictionary containing extraction results
            output_path: Path where JSON file should be saved
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 JSON file saved: {output_path}")
    
    def extract_from_screenshots_folder(self, folder_name: str = "Screenshots") -> Dict:
        """
        Extract text from all screenshots in the Screenshots folder.
        
        Args:
            folder_name: Name of the screenshots folder (default: "Screenshots")
            
        Returns:
            Dictionary with extraction results
        """
        screenshots_dir = Path(__file__).parent / folder_name
        
        if not screenshots_dir.exists():
            raise FileNotFoundError(f"Screenshots directory not found: {screenshots_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"screenshots_text_extraction_{timestamp}.json"
        
        return self.extract_text_from_directory(screenshots_dir, output_filename)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text from images using OCR')
    parser.add_argument('input', nargs='?', help='Image file or directory path')
    parser.add_argument('--output-dir', default='Extracted_texts', help='Output directory for JSON files')
    parser.add_argument('--screenshots', action='store_true', help='Extract from Screenshots folder')
    parser.add_argument('--output', help='Output JSON filename')
    
    args = parser.parse_args()
    
    extractor = ImageTextExtractor(output_dir=args.output_dir)
    
    if args.screenshots:
        # Extract from Screenshots folder
        result = extractor.extract_from_screenshots_folder()
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            # Single image file
            result = extractor.extract_text_from_image(input_path)
            output_path = extractor.output_dir / (args.output or f"{input_path.stem}_extracted.json")
            extractor.save_to_json(result, output_path)
            print(f"\n✅ Text extracted from: {input_path.name}")
            print(f"   Output saved to: {output_path}")
        elif input_path.is_dir():
            # Directory of images
            result = extractor.extract_text_from_directory(input_path, args.output)
        else:
            print(f"Error: Path not found: {input_path}")
            return
    else:
        parser.print_help()
        return
    
    # Print summary
    if 'extractions' in result:
        total_chars = sum(
            len(ext.get('extracted_text', {}).get('full_text', '')) 
            for ext in result['extractions']
        )
        print(f"\n📊 Summary:")
        print(f"   Total characters extracted: {total_chars:,}")


if __name__ == "__main__":
    main()

