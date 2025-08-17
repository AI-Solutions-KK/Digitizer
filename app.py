import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import pandas as pd
from io import BytesIO
import json

# Configure page
st.set_page_config(
    page_title="Flowchart Shape Detector",
    page_icon="📊",
    layout="wide"
)

def preprocess_image(image):
    """Basic image preprocessing using only PIL"""
    # Convert to grayscale
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image
    
    # Apply blur to reduce noise
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Convert to numpy for easier processing
    gray_array = np.array(gray)
    blurred_array = np.array(blurred)
    
    # Simple threshold
    threshold = 128
    thresh = blurred_array < threshold
    thresh = thresh.astype(np.uint8) * 255
    
    return thresh, gray_array

def simple_shape_detection(binary_image):
    """Very basic shape detection using simple algorithms"""
    shapes_detected = []
    
    # Convert to boolean for easier processing
    binary = binary_image > 128
    height, width = binary.shape
    
    # Simple connected components (very basic implementation)
    visited = np.zeros_like(binary, dtype=bool)
    shape_id = 0
    
    def flood_fill(start_y, start_x):
        """Simple flood fill to find connected components"""
        if (start_y < 0 or start_y >= height or 
            start_x < 0 or start_x >= width or 
            visited[start_y, start_x] or 
            not binary[start_y, start_x]):
            return []
        
        points = []
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack.pop()
            if (y < 0 or y >= height or 
                x < 0 or x >= width or 
                visited[y, x] or 
                not binary[y, x]):
                continue
                
            visited[y, x] = True
            points.append((y, x))
            
            # Add 4-connected neighbors
            stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
        
        return points
    
    # Find connected components
    for y in range(height):
        for x in range(width):
            if binary[y, x] and not visited[y, x]:
                points = flood_fill(y, x)
                
                if len(points) > 100:  # Minimum size filter
                    # Calculate bounding box
                    ys, xs = zip(*points)
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)
                    
                    w = max_x - min_x + 1
                    h = max_y - min_y + 1
                    area = len(points)
                    
                    # Simple shape classification based on dimensions
                    aspect_ratio = w / h if h > 0 else 1
                    fill_ratio = area / (w * h) if (w * h) > 0 else 0
                    
                    if fill_ratio > 0.7 and 0.8 <= aspect_ratio <= 1.2:
                        shape_type = "Square/Circle"
                    elif aspect_ratio < 0.6 or aspect_ratio > 1.8:
                        shape_type = "Rectangle"
                    else:
                        shape_type = "Rectangle"
                    
                    shapes_detected.append({
                        'id': shape_id,
                        'type': shape_type,
                        'area': area,
                        'x': min_x,
                        'y': min_y,
                        'width': w,
                        'height': h,
                        'center_x': min_x + w // 2,
                        'center_y': min_y + h // 2,
                        'aspect_ratio': round(aspect_ratio, 2),
                        'fill_ratio': round(fill_ratio, 2)
                    })
                    
                    shape_id += 1
    
    return shapes_detected

def draw_detection_results(original_image, shapes):
    """Draw bounding boxes on the image using PIL"""
    if isinstance(original_image, np.ndarray):
        if len(original_image.shape) == 3:
            result_image = Image.fromarray(original_image, 'RGB')
        else:
            result_image = Image.fromarray(original_image, 'L').convert('RGB')
    else:
        result_image = original_image.convert('RGB')
    
    draw = ImageDraw.Draw(result_image)
    
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown']
    
    for i, shape in enumerate(shapes):
        x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
        color = colors[i % len(colors)]
        
        # Draw bounding rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        
        # Add shape label
        label = f"{shape['type']} (ID: {shape['id']})"
        draw.text((x, max(0, y - 15)), label, fill=color)
        
        # Add area info
        draw.text((x, y + h + 2), f"Area: {shape['area']}", fill=color)
    
    return result_image

def export_to_json(shapes):
    """Export results to JSON format"""
    export_data = {
        'flowchart_elements': shapes,
        'metadata': {
            'total_shapes': len(shapes),
            'detection_method': 'simple_flood_fill'
        }
    }
    return json.dumps(export_data, indent=2)

# Main app
def main():
    st.title("📊 Simple Flowchart Shape Detector")
    st.markdown("Upload a flowchart image and detect basic shapes")
    st.info("💡 **Ultra-minimal version** - Basic shape detection using simple algorithms")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    min_area = st.sidebar.slider("Minimum Shape Area (pixels)", 100, 5000, 500)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear, high-contrast image of your flowchart"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with st.spinner("Detecting shapes..."):
            # Preprocess image
            processed_img, gray_img = preprocess_image(image)
            
            # Detect shapes
            shapes = simple_shape_detection(processed_img)
            
            # Filter by minimum area
            shapes = [s for s in shapes if s['area'] >= min_area]
        
        with col2:
            st.subheader("Detection Results")
            if shapes:
                result_img = draw_detection_results(image, shapes)
                st.image(result_img, use_container_width=True)
            else:
                st.warning("No shapes detected. Try adjusting the minimum area or use a higher contrast image.")
        
        # Display results
        if shapes:
            st.subheader("Detected Shapes")
            
            # Create DataFrame for display
            display_data = []
            for shape in shapes:
                row = {
                    'ID': shape['id'],
                    'Type': shape['type'],
                    'Position': f"({shape['x']}, {shape['y']})",
                    'Size': f"{shape['width']} × {shape['height']}",
                    'Area': shape['area'],
                    'Aspect Ratio': shape['aspect_ratio'],
                    'Fill Ratio': shape['fill_ratio']
                }
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True)
            
            # Export options
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = export_to_json(shapes)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name="flowchart_shapes.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV export
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name="flowchart_shapes.csv",
                    mime="text/csv"
                )
            
            # Statistics
            st.subheader("Detection Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Shapes", len(shapes))
            
            with col2:
                shape_types = len(set(shape['type'] for shape in shapes))
                st.metric("Shape Types", shape_types)
            
            with col3:
                avg_area = int(np.mean([shape['area'] for shape in shapes]))
                st.metric("Avg Area", f"{avg_area} px")
            
            with col4:
                total_area = sum(shape['area'] for shape in shapes)
                st.metric("Total Area", f"{total_area} px")
        
        else:
            st.info("No shapes detected in the image.")
            st.markdown("""
            **Tips for better detection:**
            - Use high-contrast images (black shapes on white background)
            - Ensure shapes are clearly separated
            - Avoid very small shapes
            - Try adjusting the minimum area setting
            """)
    
    else:
        st.info("👆 Please upload an image file to get started")
        
        # Instructions
        st.subheader("How to use:")
        st.markdown("""
        1. **Upload** a clear, high-contrast flowchart image
        2. **Adjust** the minimum area threshold if needed
        3. **View** detected shapes with bounding boxes
        4. **Download** results in JSON or CSV format
        
        **Current capabilities:**
        - Basic shape detection using flood-fill algorithm
        - Bounding box calculation
        - Simple shape classification
        - Export functionality
        
        **Note:** This is a minimal version focusing on core functionality only.
        """)

if __name__ == "__main__":
    main()
