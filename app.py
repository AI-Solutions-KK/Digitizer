import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pandas as pd
from io import BytesIO
import json
from scipy import ndimage
from skimage import measure, morphology, filters
from skimage.segmentation import clear_border

# Configure page
st.set_page_config(
    page_title="Flowchart to Editable Format",
    page_icon="📊",
    layout="wide"
)

# Shape detection without OCR
@st.cache_data
def get_app_info():
    return {"version": "1.0", "ocr_enabled": False}

def preprocess_image(image):
    """Basic image preprocessing using PIL and skimage"""
    # Convert PIL to grayscale
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image
    
    # Convert to numpy array
    gray_array = np.array(gray)
    
    # Apply Gaussian blur using scipy
    blurred = ndimage.gaussian_filter(gray_array, sigma=1.0)
    
    # Apply threshold using skimage
    thresh_val = filters.threshold_otsu(blurred)
    thresh = blurred > thresh_val
    thresh = thresh.astype(np.uint8) * 255
    
    return thresh, gray_array

def detect_shapes(image):
    """Detect basic shapes using skimage"""
    shapes_detected = []
    
    # Convert to binary
    binary = image < 128  # Invert for skimage
    
    # Remove noise
    cleaned = morphology.remove_small_objects(binary, min_size=500)
    cleaned = clear_border(cleaned)
    
    # Label connected components
    labeled = measure.label(cleaned)
    regions = measure.regionprops(labeled)
    
    for i, region in enumerate(regions):
        # Filter small areas
        if region.area < 500:
            continue
        
        # Get bounding box
        minr, minc, maxr, maxc = region.bbox
        y, x, h, w = minr, minc, maxr - minr, maxc - minc
        
        # Get shape characteristics
        perimeter = region.perimeter
        area = region.area
        
        # Calculate shape metrics
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 1
        else:
            circularity = 0
            aspect_ratio = 1
        
        # Classify shape
        shape_type = "Unknown"
        
        if circularity > 0.7:
            shape_type = "Circle"
        elif aspect_ratio < 1.2:
            shape_type = "Square"
        elif aspect_ratio < 3.0:
            shape_type = "Rectangle"
        else:
            shape_type = "Elongated Shape"
        
        # Calculate vertices (approximation)
        vertices = 4  # Default for most flowchart shapes
        if circularity > 0.7:
            vertices = 0  # Circle
        
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
            'center_y': int(y + h/2),
            'circularity': round(circularity, 2),
            'aspect_ratio': round(aspect_ratio, 2)
        })
    
    return shapes_detected

def detect_text_in_shapes(image_gray, shapes):
    """Placeholder for text detection - returns empty text for all shapes"""
    text_results = []
    
    for shape in shapes:
        text_results.append({
            'shape_id': shape['id'],
            'text': "Text detection disabled",
            'confidence': 0.0
        })
    
    return text_results

def draw_detection_results(original_image, shapes, texts):
    """Draw bounding boxes and text on the image using PIL"""
    # Convert numpy array to PIL Image if needed
    if isinstance(original_image, np.ndarray):
        if len(original_image.shape) == 3:
            result_image = Image.fromarray(original_image, 'RGB')
        else:
            result_image = Image.fromarray(original_image, 'L').convert('RGB')
    else:
        result_image = original_image.convert('RGB')
    
    draw = ImageDraw.Draw(result_image)
    
    # Create text lookup
    text_lookup = {text['shape_id']: text for text in texts}
    
    for shape in shapes:
        x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
        
        # Draw bounding rectangle
        draw.rectangle([x, y, x + w, y + h], outline='green', width=2)
        
        # Add shape type label
        label = f"{shape['type']} (ID: {shape['id']})"
        draw.text((x, y - 15), label, fill='green')
        
        # Add detected text if any
        if shape['id'] in text_lookup and text_lookup[shape['id']]['text']:
            text = text_lookup[shape['id']]['text']
            if len(text) > 20:
                text = text[:20] + "..."
            draw.text((x, y + h + 5), f"Text: {text}", fill='red')
    
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
    st.title("📊 Flowchart Shape Detector")
    st.markdown("Upload a flowchart image and detect shapes for further editing")
    st.info("💡 **Note:** Text extraction is currently disabled. This version focuses on shape detection only.")
    
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with st.spinner("Processing image..."):
            # Preprocess image
            processed_img, gray_img = preprocess_image(image)
            
            # Detect shapes
            shapes = detect_shapes(processed_img)
            
            # Filter by minimum area
            shapes = [s for s in shapes if s['area'] >= min_area]
            
            # Extract text (disabled for now)
            if shapes:
                texts = detect_text_in_shapes(gray_img, shapes)
            else:
                texts = []
        
        with col2:
            st.subheader("Detection Results")
            if shapes:
                result_img = draw_detection_results(image, shapes, texts)
                st.image(result_img, use_container_width=True)
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
                if texts and any(text['confidence'] > 0 for text in texts):
                    avg_confidence = np.mean([text['confidence'] for text in texts if text['confidence'] > 0])
                    st.metric("Avg Text Confidence", f"{avg_confidence:.2f}")
                else:
                    st.metric("Avg Text Confidence", "0.00")
            
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
