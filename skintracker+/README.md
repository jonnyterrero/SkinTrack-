# SkinTrack+ - Chronic Skin Condition Tracker

A comprehensive Streamlit application for tracking chronic skin conditions with image analysis, metrics, medication scheduling, and trend visualization.

## Features

- **Multi-condition tracking**: eczema, psoriasis, keratosis pilaris, acne, melanoma, vitiligo, contact dermatitis, cold sores
- **Image analysis**: automatic segmentation, area measurement, color analysis, border irregularity, asymmetry
- **Image capture & analysis**: Take photos with camera, upload existing images, automated skin condition analysis
- **Symptom logging**: itch, pain, sleep, stress levels with trigger tracking
- **Food & stress tracking**: Log food intake and stress events and their impact on skin conditions
- **Sun exposure tracking**: Monitor time in sun and its effectiveness as a treatment
- **Medication management**: schedule tracking and adherence monitoring
- **Data visualization**: Comprehensive charts and statistical analysis
- **Data export**: CSV and PDF reports for healthcare providers
- **Trend visualization**: interactive charts showing progression over time
- **Simulation**: what-if scenarios for treatment planning

## Installation

1. **Clone or download** this repository to your local machine

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run skintrack_app.py
   ```

## Usage

### Getting Started

1. **Create a lesion**: Go to the "Log Entry" tab and create a new lesion with a label and condition type
2. **Capture/upload image**: Use your camera or upload a photo of the affected area
3. **Calibrate scale**: Optionally use an ArUco marker for accurate measurements
4. **Log symptoms**: Record itch, pain, sleep, stress levels and triggers
5. **Analyze**: The app will automatically segment the lesion and compute metrics

### Standalone CLI Version

For users who prefer a command-line interface, there's also a standalone Python version:

```bash
python skintrack_standalone.py
```

The standalone version includes all the same features as the web interface, plus:
- **Image capture and analysis**: Take photos with camera or upload existing images
- **Food and stress tracking**: Log how diet and stress affect skin conditions
- **Sun exposure monitoring**: Track time in sun and its effectiveness
- **Comprehensive data analysis**: Generate charts and statistical summaries
- **Local data storage**: All data stored locally on your device

### Tips for Best Results

- **Lighting**: Use indirect daylight for consistent photos
- **Distance**: Keep camera 30-40 cm from the lesion
- **Reference**: Place ArUco marker or color card near the lesion for calibration
- **Consistency**: Take photos at the same time of day for comparable results

### Features by Tab

- **📥 Log Entry**: Create lesions, capture images, log symptoms and medications
- **📈 Trends**: View progression charts and recent photos
- **💊 Med Schedule**: Set up medication reminders and track adherence
- **📤 Export**: Download CSV data or generate PDF summaries
- **🧪 Simulate**: Run what-if scenarios for treatment planning

### Standalone Version Features

- **📸 Image Capture & Analysis**: Take photos, upload images, analyze skin metrics
- **🍽️ Food & Stress Tracking**: Log diet and stress impact on skin conditions
- **☀️ Sun Exposure Monitoring**: Track sun exposure effectiveness
- **📊 Data Analysis & Charts**: Generate comprehensive visualizations
- **💊 Medication Logging**: Track medication usage and effectiveness

## Technical Details

### Image Processing
- **Segmentation**: K-means clustering, GrabCut, or optional U-Net
- **Metrics**: Area (cm²), redness (R/G ratio), border irregularity, asymmetry, ΔE color difference
- **Calibration**: ArUco marker detection or manual scale estimation
- **Image Capture**: Camera integration and file upload support
- **Analysis**: Automated skin condition metrics extraction

### Data Storage
- **SQLite database**: Local storage of lesions, records, and medication schedules
- **Image files**: Stored in `skintrack_data/images/` directory
- **Export formats**: CSV for data analysis, PDF for clinical summaries

### Dependencies
- **Core**: Streamlit, OpenCV, NumPy, Pandas
- **Image processing**: scikit-image, Pillow
- **Visualization**: Plotly
- **UI**: streamlit-drawable-canvas
- **Reports**: ReportLab
- **Optional ML**: PyTorch, TorchVision

## Important Notes

⚠️ **Medical Disclaimer**: This application is for tracking purposes only and is NOT a diagnostic tool. Always consult with healthcare professionals for medical decisions.

⚠️ **Data Privacy**: All data is stored locally on your device. No data is transmitted to external servers.

⚠️ **Image Quality**: Results depend on image quality, lighting, and camera positioning. Follow the provided tips for best results.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **Camera not working**: Try uploading images instead of using camera input
3. **Segmentation issues**: Ensure good lighting and clear lesion boundaries
4. **Scale calibration**: Use ArUco markers or manual measurement for accurate area calculations

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam (optional, for photo capture)
- Modern web browser

## Documentation

- **Main Application**: This README covers the Streamlit web interface
- **Standalone Version**: See `README_STANDALONE.md` for CLI version details
- **Image Features**: See `README_IMAGE_FEATURES.md` for detailed image capture and analysis documentation
- **Food & Stress Features**: See `README_FOOD_STRESS_FEATURES.md` for tracking features
- **Data Visualization**: See `README_DATA_VISUALIZATION.md` for charting capabilities

## Development

The application is built with:
- **Frontend**: Streamlit for the web interface
- **Backend**: Python with OpenCV for image processing
- **Database**: SQLite for data persistence
- **Visualization**: Plotly for interactive charts

## License

This project is for educational and personal use. Please consult healthcare professionals for medical advice.
