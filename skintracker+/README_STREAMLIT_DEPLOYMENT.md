# SkinTrack+ Streamlit Deployment Guide

## Overview

This guide will help you deploy the SkinTrack+ Image Analysis application to Streamlit Cloud from GitHub. The application provides comprehensive skin condition tracking with image capture, analysis, and data visualization capabilities.

## Files Included

### Core Application Files

1. **`streamlit_image_app.py`** - Main Streamlit application
2. **`requirements.txt`** - Python dependencies
3. **`skintrack_complete.py`** - Complete CLI version (for reference)
4. **`test_image_features.py`** - Test suite for image features
5. **`README_IMAGE_FEATURES.md`** - Detailed image features documentation

### Documentation Files

- **`README.md`** - Main project documentation
- **`README_FOOD_STRESS_FEATURES.md`** - Food and stress tracking documentation
- **`README_DATA_VISUALIZATION.md`** - Data visualization documentation

## Quick Start

### 1. Local Testing

Before deploying, test the application locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_image_app.py
```

### 2. GitHub Deployment

1. **Create a GitHub Repository**
   - Create a new repository on GitHub
   - Name it something like `skintrack-plus` or `skin-condition-tracker`

2. **Upload Files**
   - Upload all the files to your repository
   - Ensure the main file is named `streamlit_image_app.py`

3. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to: `streamlit_image_app.py`
   - Click "Deploy"

## Application Features

### 📸 Image Capture & Analysis
- **Upload Images**: Drag and drop or browse for image files
- **Image Analysis**: Automated analysis of skin conditions
- **Metrics Extraction**: Area, redness, border irregularity, asymmetry
- **Visual Results**: Display analysis results with charts

### 📝 Record Management
- **Create Lesions**: Define skin condition areas
- **Add Records**: Log symptoms with optional images
- **Symptom Tracking**: Itch, pain, sleep, stress levels
- **Image Association**: Link images to symptom records

### 📊 Data Analysis
- **Trend Visualization**: Charts showing symptom progression
- **Correlation Analysis**: Relationships between different metrics
- **Statistical Summary**: Average values and patterns
- **Data Export**: Download data as CSV files

### 🖼️ Image Gallery
- **View All Images**: Browse captured images by lesion
- **Metadata Display**: Show associated symptom data
- **Filtering**: View images by specific lesions

## Technical Requirements

### Dependencies

The application requires the following Python packages:

```
streamlit>=1.28.0
opencv-python>=4.8.0
pillow>=10.0.0
scikit-image>=0.21.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pathlib2>=2.3.7
```

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space for images and data
- **Browser**: Modern web browser with JavaScript enabled

## File Structure

```
your-repository/
├── streamlit_image_app.py      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── skintrack_complete.py       # Complete CLI version
├── test_image_features.py      # Test suite
├── README.md                   # Main documentation
├── README_STREAMLIT_DEPLOYMENT.md  # This file
├── README_IMAGE_FEATURES.md    # Image features documentation
├── README_FOOD_STRESS_FEATURES.md  # Food/stress features
├── README_DATA_VISUALIZATION.md    # Data visualization docs
└── .gitignore                  # Git ignore file (optional)
```

## Usage Instructions

### Getting Started

1. **Access the Application**
   - Open your deployed Streamlit app URL
   - You'll see the home page with navigation options

2. **Create Your First Lesion**
   - Go to "Add Record" page
   - Create a new lesion with a label and condition type
   - This will be your first skin condition entry

3. **Upload and Analyze Images**
   - Use "Image Capture" to upload photos
   - Navigate to "Image Analysis" to analyze uploaded images
   - View results and metrics

4. **Track Progress**
   - Use "Data Analysis" to see trends over time
   - Export data for healthcare providers
   - View image gallery to compare before/after photos

### Best Practices

#### Image Quality
- **Lighting**: Use indirect daylight for consistent photos
- **Distance**: Keep camera 30-40 cm from skin surface
- **Angle**: Take photos perpendicular to the skin
- **Resolution**: Use high-resolution images when possible
- **Consistency**: Take photos at the same time of day

#### Data Entry
- **Regular Logging**: Log symptoms daily for best tracking
- **Detailed Notes**: Include triggers and environmental factors
- **Image Consistency**: Use similar lighting and angles
- **Medication Tracking**: Log all treatments and their effectiveness

## Troubleshooting

### Common Issues

1. **App Won't Deploy**
   - Check that `streamlit_image_app.py` is the main file
   - Verify all dependencies are in `requirements.txt`
   - Ensure Python version is 3.8+

2. **Image Upload Fails**
   - Check file format (JPG, PNG, BMP, TIFF supported)
   - Ensure file size is under 200MB
   - Try refreshing the page

3. **Analysis Errors**
   - Verify image is not corrupted
   - Check that image contains visible skin areas
   - Try uploading a different image

4. **Database Issues**
   - Data is stored locally in the Streamlit session
   - For persistent storage, consider using Streamlit's data persistence features
   - Export data regularly to avoid loss

### Error Messages

- **"OpenCV not available"**: The app will still work but with limited image processing
- **"PIL not available"**: Image handling will be limited
- **"File not found"**: Check the image path and file existence
- **"Database error"**: Try refreshing the page or restarting the app

## Customization

### Adding New Features

1. **Modify `streamlit_image_app.py`**
   - Add new pages to the navigation
   - Implement new analysis functions
   - Customize the UI layout

2. **Update Dependencies**
   - Add new packages to `requirements.txt`
   - Test locally before deploying

3. **Database Schema**
   - Modify the `init_db()` function
   - Add new tables as needed
   - Update data insertion functions

### Styling

The application uses Streamlit's default styling with custom color schemes. You can modify:

- **Color schemes** in the `COLOR_SCHEMES` dictionary
- **Page layout** using Streamlit columns and containers
- **Charts** by modifying matplotlib/seaborn parameters

## Security Considerations

### Data Privacy
- **Local Storage**: Images and data are stored locally in the Streamlit session
- **No Cloud Upload**: Images are not automatically uploaded to external services
- **User Control**: Users have full control over their data
- **Export Options**: Users can export and delete their data

### Best Practices
- **Regular Backups**: Export data regularly
- **Secure Sharing**: Be careful when sharing exported data
- **Medical Disclaimer**: Always include appropriate medical disclaimers
- **User Education**: Inform users about data privacy

## Support and Maintenance

### Monitoring
- **Usage Analytics**: Monitor app usage through Streamlit Cloud
- **Error Logs**: Check Streamlit Cloud logs for errors
- **Performance**: Monitor app response times

### Updates
- **Regular Updates**: Keep dependencies updated
- **Feature Additions**: Add new features based on user feedback
- **Bug Fixes**: Address issues promptly

### User Support
- **Documentation**: Keep documentation updated
- **FAQ**: Create a frequently asked questions section
- **Contact**: Provide contact information for support

## Advanced Features

### Future Enhancements

1. **Machine Learning**
   - AI-powered skin condition classification
   - Automated severity assessment
   - Predictive analytics

2. **Integration**
   - Electronic health record integration
   - Telemedicine platform integration
   - Mobile app companion

3. **Advanced Analytics**
   - 3D image analysis
   - Comparative analysis tools
   - Treatment effectiveness tracking

### Research Applications

- **Clinical Trials**: Standardized data collection
- **Population Studies**: Large-scale analysis
- **Treatment Research**: Efficacy studies
- **Medical Education**: Training and demonstration

## License and Legal

### Medical Disclaimer

This application is for tracking and monitoring purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

### Data Protection

- **GDPR Compliance**: Consider data protection regulations
- **HIPAA Considerations**: Be aware of healthcare data regulations
- **User Consent**: Obtain appropriate user consent for data collection

### Open Source

This project is open source and available for educational and research purposes. Please consult healthcare professionals for medical advice.

---

## Quick Deployment Checklist

- [ ] Create GitHub repository
- [ ] Upload all files
- [ ] Verify `streamlit_image_app.py` is the main file
- [ ] Check `requirements.txt` includes all dependencies
- [ ] Deploy to Streamlit Cloud
- [ ] Test all features
- [ ] Share the app URL
- [ ] Monitor usage and feedback

For additional support or questions, please refer to the main documentation or create an issue in the GitHub repository.
