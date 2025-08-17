import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import pandas as pd
from io import BytesIO
import json

# Configure page
st.set_page_config(
    page_title="Flowchart to Editable Format",
    page_icon="📊",
    layout="wide"
)

# Initialize OCR reader
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'])

def preprocess_image(image):
    """Basic image preprocessing"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh, gray

def detect_shapes(image):
    """Detect basic shapes in the image"""
    shapes_detected = []
    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < 500:  # Minimum area threshold
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify shape based on number of vertices
        shape_type = "Unknown"
        vertices = len(approx)
        
        if vertices == 3:
            shape_type = "Triangle"
        elif vertices == 4:
            # Check if rectangle or square
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "Square"
            else:
                shape_type = "Rectangle"
        elif vertices > 4:
            # Check if circle/ellipse
            area_contour = cv2.contourArea(contour)
            area_rect = w * h
            if area_contour / area_rect > 0.7:
                shape_type = "Circle/Ellipse"
            else:
                shape_type = "Polygon"
        
        shapes_detected.append({
            'id': i,
            'type': shape_type,
            'vertices': vertices,
            'area': int(area),
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'center_x': int(x + w/2),
            'center_y': int(y + h/2)
        })
    
    return shapes_detected

def detect_text_in_shapes(image_gray, shapes, ocr_reader):
    """Extract text from detected shapes"""
    text_results = []
    
    for shape in shapes:
        # Extract ROI (Region of Interest)
        x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
        
        # Add padding
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image_gray.shape[1], x + w + padding)
        y_end = min(image_gray.shape[0], y + h + padding)
        
        roi = image_gray[y_start:y_end, x_start:x_end]
        
        if roi.size > 0:
            try:
                # Use OCR to extract text
                results = ocr_reader.readtext(roi)
                
                extracted_text = ""
                confidence_scores = []
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.3:  # Confidence threshold
                        extracted_text += text + " "
                        confidence_scores.append(confidence)
                
                if extracted_text.strip():
                    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                    text_results.append({
                        'shape_id': shape['id'],
                        'text': extracted_text.strip(),
                        'confidence': round(avg_confidence, 2)
                    })
                else:
                    text_results.append({
                        'shape_id': shape['id'],
                        'text': "",
                        'confidence': 0
                    })
                    
            except Exception as e:
                text_results.append({
                    'shape_id': shape['id'],
                    'text': "",
                    'confidence': 0
                })
    
    return text_results

def draw_detection_results(original_image, shapes, texts):
    """Draw bounding boxes and text on the image"""
    result_image = original_image.copy()
    
    # Create text lookup
    text_lookup = {text['shape_id']: text for text in texts}
    
    for shape in shapes:
        x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
        
        # Draw bounding rectangle
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add shape type label
        label = f"{shape['type']} (ID: {shape['id']})"
        cv2.putText(result_image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add detected text if any
        if shape['id'] in text_lookup and text_lookup[shape['id']]['text']:
            text = text_lookup[shape['id']]['text'][:20] + "..." if len(text_lookup[shape['id']]['text']) > 20 else text_lookup[shape['id']]['text']
            cv2.putText(result_image, f"Text: {text}", (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    return result_image

def export_to_json(shapes, texts):
    """Export results to JSON format"""
    # Merge shapes and texts
    text_lookup = {text['shape_id']: text for text in texts}
    
    export_data = {
        'flowchart_elements': []
    }
    
    for shape in shapes:
        element = shape.copy()
        if shape['id'] in text_lookup:
            element['text'] = text_lookup[shape['id']]['text']
            element['text_confidence'] = text_lookup[shape['id']]['confidence']
        else:
            element['text'] = ""
            element['text_confidence'] = 0
            
        export_data['flowchart_elements'].append(element)
    
    return json.dumps(export_data, indent=2)

# Main app
def main():
    st.title("📊 Flowchart to Editable Format Converter")
    st.markdown("Upload a flowchart image and extract shapes and text for further editing")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    min_area = st.sidebar.slider("Minimum Shape Area", 100, 2000, 500)
    confidence_threshold = st.sidebar.slider("Text Confidence Threshold", 0.1, 1.0, 0.3)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of your flowchart"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with st.spinner("Processing image..."):
            # Load OCR reader
            reader = load_ocr_reader()
            
            # Preprocess image
            processed_img, gray_img = preprocess_image(image_cv)
            
            # Detect shapes
            shapes = detect_shapes(processed_img)
            
            # Filter by minimum area
            shapes = [s for s in shapes if s['area'] >= min_area]
            
            # Extract text
            if shapes:
                texts = detect_text_in_shapes(gray_img, shapes, reader)
            else:
                texts = []
        
        with col2:
            st.subheader("Detection Results")
            if shapes:
                result_img = draw_detection_results(image_cv, shapes, texts)
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img_rgb, use_container_width=True)
            else:
                st.warning("No shapes detected. Try adjusting the minimum area threshold.")
        
        # Display results
        if shapes:
            st.subheader("Detected Elements")
            
            # Create DataFrame for better display
            display_data = []
            text_lookup = {text['shape_id']: text for text in texts}
            
            for shape in shapes:
                row = {
                    'ID': shape['id'],
                    'Type': shape['type'],
                    'Position': f"({shape['x']}, {shape['y']})",
                    'Size': f"{shape['width']} x {shape['height']}",
                    'Area': shape['area'],
                    'Text': text_lookup.get(shape['id'], {}).get('text', 'No text detected'),
                    'Confidence': text_lookup.get(shape['id'], {}).get('confidence', 0)
                }
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True)
            
            # Export options
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = export_to_json(shapes, texts)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name="flowchart_data.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV export
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name="flowchart_elements.csv",
                    mime="text/csv"
                )
            
            # Statistics
            st.subheader("Detection Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Shapes", len(shapes))
            
            with col2:
                shapes_with_text = sum(1 for text in texts if text['text'])
                st.metric("Shapes with Text", shapes_with_text)
            
            with col3:
                avg_confidence = np.mean([text['confidence'] for text in texts if text['confidence'] > 0])
                st.metric("Avg Text Confidence", f"{avg_confidence:.2f}" if not np.isnan(avg_confidence) else "0.00")
            
            with col4:
                shape_types = len(set(shape['type'] for shape in shapes))
                st.metric("Shape Types", shape_types)
        
        else:
            st.info("No shapes detected in the image. Try uploading a clearer image or adjusting the settings.")
    
    else:
        st.info("👆 Please upload an image file to get started")
        
        # Sample instructions
        st.subheader("How to use:")
        st.markdown("""
        1. **Upload** a clear image of your flowchart
        2. **Adjust** settings in the sidebar if needed
        3. **View** detected shapes and text
        4. **Download** results in JSON or CSV format
        
        **Tips for better results:**
        - Use high-contrast images
        - Ensure text is clearly readable
        - Avoid overlapping shapes
        - Use good lighting when taking photos
        """)

if __name__ == "__main__":
    main()