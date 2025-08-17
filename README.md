---
title: Diagram Digitizer
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Diagram Digitizer 📊

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/your-username/diagram-digitizer)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An AI-powered web application that converts handwritten or image-based flowcharts into editable digital formats. Built with Streamlit and computer vision models.

## 🚀 Features

- **Smart Shape Detection**: Automatically identifies rectangles, circles, diamonds, and other flowchart elements
- **OCR Text Extraction**: Reads handwritten and printed text within shapes using EasyOCR
- **Visual Analysis**: Real-time processing with visual feedback and bounding boxes
- **Multiple Export Formats**: Download results as JSON or CSV for further editing
- **Interactive Interface**: User-friendly web app with adjustable detection parameters
- **No Training Required**: Uses pre-trained models for immediate results

## 🎯 Use Cases

- Convert hand-drawn process flows to digital format
- Digitize whiteboard diagrams from meetings
- Extract flowchart data for documentation
- Create editable versions of scanned flowcharts
- Educational tools for diagram analysis

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **OCR**: EasyOCR
- **Image Processing**: PIL, NumPy
- **Data Handling**: Pandas

## 🚀 Quick Start

### Online Demo
Try the app instantly on Hugging Face Spaces: [Diagram Digitizer Demo](https://huggingface.co/spaces/your-username/diagram-digitizer)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/diagram-digitizer.git
cd diagram-digitizer
# Diagram Digitizer 📊

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/your-username/diagram-digitizer)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An AI-powered web application that converts handwritten or image-based flowcharts into editable digital formats. Built with Streamlit and computer vision models.

## 🚀 Features

- **Smart Shape Detection**: Automatically identifies rectangles, circles, diamonds, and other flowchart elements
- **OCR Text Extraction**: Reads handwritten and printed text within shapes using EasyOCR
- **Visual Analysis**: Real-time processing with visual feedback and bounding boxes
- **Multiple Export Formats**: Download results as JSON or CSV for further editing
- **Interactive Interface**: User-friendly web app with adjustable detection parameters
- **No Training Required**: Uses pre-trained models for immediate results

## 🎯 Use Cases

- Convert hand-drawn process flows to digital format
- Digitize whiteboard diagrams from meetings
- Extract flowchart data for documentation
- Create editable versions of scanned flowcharts
- Educational tools for diagram analysis

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **OCR**: EasyOCR
- **Image Processing**: PIL, NumPy
- **Data Handling**: Pandas

## 🚀 Quick Start

### Online Demo
Try the app instantly on Hugging Face Spaces: [Diagram Digitizer Demo](https://huggingface.co/spaces/your-username/diagram-digitizer)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/diagram-digitizer.git
cd diagram-digitizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 How to Use

1. **Upload Image**: Drag and drop your flowchart image (JPG, PNG, BMP)
2. **Adjust Settings**: Fine-tune detection parameters in the sidebar
3. **View Results**: See detected shapes and extracted text with visual feedback
4. **Export Data**: Download results as JSON or CSV for further processing

## 📊 Supported Formats

### Input Formats
- JPG, JPEG
- PNG
- BMP

### Output Formats
- JSON (structured data with coordinates and text)
- CSV (tabular format for spreadsheet analysis)

## 🎨 Example Results

The app detects various flowchart elements:
- **Rectangles**: Process steps
- **Diamonds**: Decision points  
- **Circles/Ovals**: Start/End points
- **Text**: Labels and descriptions within shapes

## 🔧 Configuration

Adjustable parameters in the sidebar:
- **Minimum Shape Area**: Filter out small noise (100-2000 pixels)
- **Text Confidence Threshold**: OCR accuracy threshold (0.1-1.0)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Commit changes**: `git commit -m "Add feature"`
4. **Push to branch**: `git push origin feature-name`
5. **Open a Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/diagram-digitizer.git
cd diagram-digitizer

# Install in development mode
pip install -r requirements.txt

# Run with hot reload
streamlit run app.py
```

## 📝 Roadmap

- [ ] Support for more shape types (hexagons, parallelograms)
- [ ] Arrow and connection line detection
- [ ] Export to Word/PowerPoint formats
- [ ] Batch processing for multiple images
- [ ] Custom shape training interface
- [ ] Integration with draw.io format

## 🐛 Known Issues

- Complex overlapping shapes may not be detected accurately
- Handwritten text with very low contrast might be missed
- Very small shapes (< 100 pixels) are filtered out by default

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Optical Character Recognition
- [OpenCV](https://opencv.org/) - Computer Vision Library
- [Streamlit](https://streamlit.io/) - Web App Framework

## 📧 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-username/diagram-digitizer/issues)
- **Hugging Face**: [View on HF Spaces](https://huggingface.co/spaces/your-username/diagram-digitizer)

---

⭐ **Star this repository if you find it helpful!**

## 🔄 Recent Updates

- **v1.0.0**: Initial release with basic shape and text detection
- Added support for multiple image formats
- Implemented JSON/CSV export functionality
- Enhanced UI with statistics dashboard
