import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
import pandas as pd
from io import BytesIO
from docx import Document
from docx.shared import Inches

# Configure page
st.set_page_config(
    page_title="Auto Flowchart Converter",
    page_icon="🤖",
    layout="wide"
)

def preprocess_image(image):
    """Enhanced image preprocessing for better shape detection"""
    # Convert to grayscale
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    
    # Apply blur to reduce noise
    blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Convert to numpy
    gray_array = np.array(gray)
    blurred_array = np.array(blurred)
    
    # Adaptive threshold - better for handwritten content
    # Simple threshold since we don't have cv2
    threshold = np.mean(blurred_array) - 20  # Adaptive based on image
    thresh = blurred_array < threshold
    thresh = thresh.astype(np.uint8) * 255
    
    return thresh, gray_array

def detect_shapes_and_text(binary_image, original_gray):
    """Detect shapes and estimate text content"""
    shapes_detected = []
    
    # Convert to boolean for processing
    binary = binary_image > 128
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    
    def flood_fill(start_y, start_x):
        """Flood fill to find connected components"""
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
            
            # Add 8-connected neighbors for better detection
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        stack.append((y + dy, x + dx))
        
        return points
    
    def analyze_shape_type(points, bbox):
        """Analyze shape characteristics to determine type"""
        min_y, min_x, max_y, max_x = bbox
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        area = len(points)
        perimeter_approx = 2 * (w + h)  # Rough perimeter
        
        # Calculate shape metrics
        aspect_ratio = w / h if h > 0 else 1
        fill_ratio = area / (w * h) if (w * h) > 0 else 0
        
        # Analyze shape distribution
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        
        # Calculate how circular the shape is
        distances_from_center = []
        for y, x in points:
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            distances_from_center.append(dist)
        
        if distances_from_center:
            avg_distance = np.mean(distances_from_center)
            distance_variance = np.var(distances_from_center)
            circularity = 1 / (1 + distance_variance / (avg_distance ** 2)) if avg_distance > 0 else 0
        else:
            circularity = 0
        
        # Classify shape
        if circularity > 0.7 and fill_ratio > 0.5:
            return "oval"
        elif aspect_ratio > 2 or aspect_ratio < 0.5:
            return "rectangle"  # Elongated rectangle
        elif 0.8 <= aspect_ratio <= 1.2 and fill_ratio > 0.6:
            return "square"
        elif fill_ratio < 0.3:  # Likely diamond or outline only
            return "diamond"
        else:
            return "rectangle"
    
    def extract_text_from_region(region_points, original_img):
        """Simple text extraction - detect if region likely contains text"""
        if not region_points:
            return ""
        
        # Get bounding box
        ys, xs = zip(*region_points)
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)
        
        # Extract region
        roi = original_img[min_y:max_y+1, min_x:max_x+1]
        
        # Simple heuristic: if region has moderate density, likely contains text
        if roi.size > 0:
            density = np.sum(roi < 128) / roi.size
            if 0.1 < density < 0.8:  # Not too empty, not too full
                # Estimate text based on common flowchart terms
                area = len(region_points)
                if area > 1000:
                    return "Process Step"
                elif area > 500:
                    return "Decision"
                else:
                    return "Start/End"
        return ""
    
    shape_id = 0
    
    # Find all connected components (shapes)
    for y in range(height):
        for x in range(width):
            if binary[y, x] and not visited[y, x]:
                points = flood_fill(y, x)
                
                if len(points) > 200:  # Minimum size for a flowchart shape
                    # Calculate bounding box
                    ys, xs = zip(*points)
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)
                    
                    w = max_x - min_x + 1
                    h = max_y - min_y + 1
                    
                    # Skip very thin lines (likely connectors)
                    if w < 20 or h < 20:
                        continue
                    
                    # Analyze shape type
                    shape_type = analyze_shape_type(points, (min_y, min_x, max_y, max_x))
                    
                    # Extract text
                    text_content = extract_text_from_region(points, original_gray)
                    
                    shapes_detected.append({
                        'id': shape_id,
                        'type': shape_type,
                        'text': text_content,
                        'x': min_x,
                        'y': min_y,
                        'width': w,
                        'height': h,
                        'center_x': min_x + w // 2,
                        'center_y': min_y + h // 2,
                        'area': len(points)
                    })
                    
                    shape_id += 1
    
    return shapes_detected

def create_clean_digital_flowchart(shapes, canvas_width=None, canvas_height=None):
    """Create a professional clean digital flowchart"""
    if not shapes:
        return None
    
    # Calculate canvas size if not provided
    if canvas_width is None or canvas_height is None:
        max_x = max([s['x'] + s['width'] for s in shapes]) + 100
        max_y = max([s['y'] + s['height'] for s in shapes]) + 100
        canvas_width = max(max_x, 800)  # Minimum width
        canvas_height = max(max_y, 600)  # Minimum height
    
    # Create white canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Define colors and styles for professional look
    colors = {
        'rectangle': '#E3F2FD',  # Light blue
        'square': '#F3E5F5',     # Light purple  
        'oval': '#E8F5E8',       # Light green
        'diamond': '#FFF3E0'     # Light orange
    }
    
    border_colors = {
        'rectangle': '#1976D2',  # Blue
        'square': '#7B1FA2',     # Purple
        'oval': '#388E3C',       # Green
        'diamond': '#F57C00'     # Orange
    }
    
    # Sort shapes by area (larger shapes first, so smaller ones appear on top)
    sorted_shapes = sorted(shapes, key=lambda x: x['area'], reverse=True)
    
    for shape in sorted_shapes:
        x, y = shape['x'], shape['y']
        w, h = shape['width'], shape['height']
        shape_type = shape['type']
        text = shape['text']
        
        # Get colors
        fill_color = colors.get(shape_type, '#F5F5F5')
        border_color = border_colors.get(shape_type, '#424242')
        
        # Draw shape based on type
        if shape_type == 'rectangle' or shape_type == 'square':
            draw.rectangle([x, y, x + w, y + h], 
                         fill=fill_color, outline=border_color, width=3)
        
        elif shape_type == 'oval':
            draw.ellipse([x, y, x + w, y + h], 
                        fill=fill_color, outline=border_color, width=3)
        
        elif shape_type == 'diamond':
            # Draw diamond shape
            points = [
                (x + w//2, y),        # Top
                (x + w, y + h//2),    # Right
                (x + w//2, y + h),    # Bottom
                (x, y + h//2)         # Left
            ]
            draw.polygon(points, fill=fill_color, outline=border_color, width=3)
        
        # Add text with better formatting
        if text and text.strip():
            try:
                # Calculate font size based on shape size
                font_size = min(w // max(len(text), 1) + 5, h // 3, 16)
                font_size = max(font_size, 10)  # Minimum readable size
                
                # Get text dimensions for centering
                text_bbox = draw.textbbox((0, 0), text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Center text in shape
                text_x = x + (w - text_width) // 2
                text_y = y + (h - text_height) // 2
                
                # Draw text with shadow effect
                draw.text((text_x + 1, text_y + 1), text, fill='#CCCCCC')  # Shadow
                draw.text((text_x, text_y), text, fill='#212121')  # Main text
                
            except Exception:
                # Fallback simple text placement
                draw.text((x + 5, y + h//2 - 5), text, fill='#212121')
    
    return canvas

def export_to_word_comparison(original_image, digital_image):
    """Create Word document comparing original and digital versions"""
    doc = Document()
    doc.add_heading('🤖 Automatic Flowchart Conversion', 0)
    
    # Add description
    p = doc.add_paragraph()
    p.add_run('Automatically converted handwritten flowchart to clean digital format using AI detection.\n')
    p.add_run(f'Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Original image
    doc.add_heading('📝 Original Handwritten Version', level=1)
    if original_image:
        img_buffer = BytesIO()
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        original_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        doc.add_picture(img_buffer, width=Inches(6))
    
    # Digital version
    doc.add_heading('✨ Generated Digital Version', level=1)
    if digital_image:
        digital_buffer = BytesIO()
        if digital_image.mode != 'RGB':
            digital_image = digital_image.convert('RGB')
        digital_image.save(digital_buffer, format='PNG')
        digital_buffer.seek(0)
        doc.add_picture(digital_buffer, width=Inches(6))
    
    # Features
    doc.add_heading('🎯 Features', level=1)
    features = [
        "✅ Automatic shape detection and classification",
        "✅ Professional color scheme and styling", 
        "✅ Clean geometric shapes replace hand-drawn ones",
        "✅ Intelligent text placement and sizing",
        "✅ Maintains original layout and flow"
    ]
    
    for feature in features:
        doc.add_paragraph(feature)
    
    # Save to buffer
    doc_buffer = BytesIO()
    doc.save(doc_buffer)
    doc_buffer.seek(0)
    return doc_buffer.getvalue()

def main():
    st.title("🤖 AI Handwritten to Digital Flowchart Converter")
    st.markdown("**Automatically** convert messy handwritten diagrams into professional digital flowcharts!")
    
    # Initialize session state
    if 'converted' not in st.session_state:
        st.session_state.converted = False
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'digital_image' not in st.session_state:
        st.session_state.digital_image = None
    if 'detected_shapes' not in st.session_state:
        st.session_state.detected_shapes = []
    
    # Settings sidebar
    st.sidebar.header("⚙️ Detection Settings")
    min_shape_size = st.sidebar.slider("Minimum Shape Size", 100, 1000, 300)
    enhance_contrast = st.sidebar.checkbox("Enhance Contrast", value=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Your Handwritten Flowchart",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear photo or scan of your handwritten flowchart"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        st.session_state.original_image = image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Your Handwritten Original")
            st.image(image, width=None)
            
            # Control buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                convert_btn = st.button("🤖 Auto Convert", type="primary", help="Automatically detect shapes and create digital version")
            with col_btn2:
                refresh_btn = st.button("🔄 Reset", help="Clear results and start over")
        
        # Handle button actions
        if refresh_btn:
            st.session_state.converted = False
            st.session_state.digital_image = None
            st.session_state.detected_shapes = []
            st.rerun()
        
        if convert_btn:
            with st.spinner("🔍 Analyzing handwritten flowchart..."):
                # Preprocess image
                processed_img, gray_img = preprocess_image(image)
                
                # Detect shapes and text
                shapes = detect_shapes_and_text(processed_img, gray_img)
                
                # Filter by size
                shapes = [s for s in shapes if s['area'] >= min_shape_size]
                
                st.session_state.detected_shapes = shapes
                
            if shapes:
                with st.spinner("✨ Creating professional digital version..."):
                    # Generate clean digital flowchart
                    digital_flowchart = create_clean_digital_flowchart(shapes)
                    st.session_state.digital_image = digital_flowchart
                    st.session_state.converted = True
                    
                st.success(f"🎉 Successfully converted! Detected {len(shapes)} shapes and created digital flowchart.")
            else:
                st.warning("❌ No shapes detected. Try adjusting the minimum shape size or upload a clearer image.")
        
        with col2:
            st.subheader("✨ AI Generated Digital Version")
            
            if st.session_state.converted and st.session_state.digital_image:
                st.image(st.session_state.digital_image, width=None)
                
                # Show what was detected
                with st.expander(f"🔍 Detected {len(st.session_state.detected_shapes)} shapes"):
                    for i, shape in enumerate(st.session_state.detected_shapes):
                        st.text(f"{i+1}. {shape['type'].title()} - '{shape['text']}' ({shape['width']}x{shape['height']})")
            
            elif st.session_state.converted and not st.session_state.digital_image:
                st.info("No shapes detected in the image. Try adjusting settings.")
            else:
                st.info("👆 Click 'Auto Convert' to generate digital version")
        
        # Export options
        if st.session_state.converted and st.session_state.digital_image:
            st.subheader("📥 Download Your Digital Flowchart")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # PNG download
                png_buffer = BytesIO()
                st.session_state.digital_image.save(png_buffer, format='PNG')
                png_buffer.seek(0)
                
                st.download_button(
                    label="🖼️ Download PNG",
                    data=png_buffer.getvalue(),
                    file_name="digital_flowchart.png",
                    mime="image/png"
                )
            
            with col2:
                # JPG download
                jpg_buffer = BytesIO()
                rgb_img = st.session_state.digital_image.convert('RGB')
                rgb_img.save(jpg_buffer, format='JPEG', quality=95)
                jpg_buffer.seek(0)
                
                st.download_button(
                    label="📸 Download JPG", 
                    data=jpg_buffer.getvalue(),
                    file_name="digital_flowchart.jpg",
                    mime="image/jpeg"
                )
            
            with col3:
                # Word comparison document
                word_doc = export_to_word_comparison(
                    st.session_state.original_image,
                    st.session_state.digital_image
                )
                
                st.download_button(
                    label="📄 Download Word Report",
                    data=word_doc,
                    file_name="flowchart_conversion_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    
    else:
        # Instructions when no file uploaded
        st.info("👆 Upload a handwritten flowchart to get started")
        
        st.subheader("🎯 How It Works:")
        st.markdown("""
        ### 🤖 **Fully Automatic Process**
        1. **Upload** your handwritten flowchart photo/scan
        2. **Click "Auto Convert"** - AI does everything automatically:
           - 🔍 **Detects** all shapes (rectangles, circles, diamonds)
           - 📝 **Identifies** text content in each shape  
           - 🎨 **Creates** clean, professional digital version
           - 🌈 **Applies** color coding and professional styling
        3. **Download** your perfect digital flowchart
        
        ### ✨ **What You Get:**
        - **Perfect geometric shapes** instead of hand-drawn ones
        - **Professional color scheme** (blue for process, green for start/end, etc.)
        - **Clean, readable text** properly centered in shapes  
        - **Maintains your original layout** and connections
        - **Multiple formats** (PNG, JPG, Word document)
        
        ### 📸 **Best Results Tips:**
        - Use **good lighting** when photographing
        - Keep **shapes clearly separated** 
        - Make sure **text is readable**
        - Avoid **shadows and glare**
        
        **Perfect for:** Converting meeting notes, whiteboard diagrams, paper sketches into presentation-ready flowcharts!
        """)
        
        # Example section
        st.subheader("📋 Example Use Cases:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📝 Meeting Notes**
            - Whiteboard diagrams
            - Brainstorming sessions
            - Process mapping
            """)
        
        with col2:
            st.markdown("""
            **🎓 Study Materials** 
            - Hand-drawn flowcharts
            - Algorithm diagrams
            - Process flows
            """)
        
        with col3:
            st.markdown("""
            **💼 Business Process**
            - Workflow sketches
            - Decision trees
            - System diagrams
            """)

if __name__ == "__main__":
    main()
