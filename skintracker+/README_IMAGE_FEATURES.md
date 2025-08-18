# SkinTrack+ Image Capture & Analysis Features

## Overview

The SkinTrack+ application now includes comprehensive image capture and analysis capabilities that allow users to take photos of their skin conditions, upload existing images, and perform automated analysis to track changes over time.

## Features

### 📸 Image Capture Options

1. **Camera Capture**: Take photos directly using your device's camera
2. **Image Upload**: Upload existing images from your device
3. **Image Management**: View and manage all captured images
4. **Image Analysis**: Automated analysis of skin condition metrics

### 🔍 Image Analysis Capabilities

The system can analyze images and extract the following metrics:

- **Area Measurement**: Calculates the approximate area of skin lesions in cm²
- **Redness Analysis**: Measures redness levels using LAB color space analysis
- **Border Irregularity**: Analyzes the irregularity of lesion borders
- **Asymmetry Analysis**: Measures the symmetry/asymmetry of lesions
- **Color Analysis**: Analyzes color saturation and other color properties
- **Texture Analysis**: Measures texture variance in the skin

## How to Use

### Accessing Image Features

1. Run the application: `python skintrack_standalone.py`
2. From the main menu, select option **15. 📸 Image Capture & Analysis**
3. Choose from the available image options

### Taking Photos with Camera

1. Select "Take photo with camera" from the image menu
2. The camera will initialize and display a live preview
3. Press 'c' to capture the image or 'q' to quit
4. The image will be automatically saved to the `skintrack_data/images/` directory
5. The system will offer to analyze the image and add it to a lesion record

### Uploading Existing Images

1. Select "Upload existing image" from the image menu
2. Enter the full path to your image file
3. Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF
4. The image will be copied to the application's image directory
5. You can then analyze the image or add it to a lesion record

### Viewing Captured Images

1. Select "View captured images" from the image menu
2. Choose a lesion to view its associated images
3. The system will display a list of all images with their associated data:
   - Record ID
   - Date captured
   - Image filename
   - Symptom levels (itch, pain, stress)

### Analyzing Images

1. Select "Analyze image metrics" from the image menu
2. Upload an image for analysis
3. The system will automatically analyze the image and display results:
   - Image size (width x height)
   - Area measurement (cm²)
   - Redness score
   - Border irregularity
   - Asymmetry score
   - Texture variance
   - Color saturation

### Adding Images to Records

When adding a record to a lesion, you can now optionally include an image:

1. Select "Add record to lesion" from the main menu
2. Complete the symptom assessment
3. When prompted, choose whether to add an image
4. If yes, you can either:
   - Take a photo with the camera
   - Upload an existing image
   - Skip adding an image
5. If an image is added, it will be automatically analyzed
6. The analysis results will be stored with the record

## Technical Implementation

### Dependencies

The image features require the following Python libraries:

- **OpenCV (cv2)**: For camera capture and image processing
- **Pillow (PIL)**: For image handling and manipulation
- **scikit-image**: For advanced color analysis (optional)
- **NumPy**: For numerical computations

### Installation

```bash
pip install opencv-python pillow scikit-image numpy
```

### File Structure

```
skintrack_data/
├── images/           # Stored images
│   ├── capture_*.jpg # Camera captures
│   └── upload_*.jpg  # Uploaded images
├── charts/           # Generated charts
└── skintrack.db      # Database with image paths
```

### Database Schema

The existing `records` table already includes an `img_path` field that stores the path to associated images. When an image is added to a record, the path is stored in this field.

### Image Analysis Algorithm

1. **Image Loading**: Images are loaded using OpenCV
2. **Preprocessing**: Images are converted to appropriate color spaces
3. **Contour Detection**: Lesions are identified using contour detection
4. **Area Calculation**: Area is calculated from contour area
5. **Color Analysis**: LAB color space is used for redness analysis
6. **Shape Analysis**: Border irregularity and asymmetry are calculated
7. **Texture Analysis**: Standard deviation of gray values is computed

## Benefits

### For Users

- **Visual Tracking**: See actual images of skin conditions over time
- **Objective Measurements**: Get quantitative measurements of lesions
- **Progress Monitoring**: Track changes in size, color, and shape
- **Medical Documentation**: Maintain a visual record for healthcare providers
- **Treatment Evaluation**: Compare before/after images of treatments

### For Healthcare Providers

- **Objective Data**: Quantitative measurements instead of subjective descriptions
- **Trend Analysis**: Visual progression of skin conditions
- **Treatment Assessment**: Before/after comparisons
- **Remote Monitoring**: Patients can share images remotely
- **Research Data**: Anonymized data for research purposes

## Limitations and Considerations

### Technical Limitations

- **Camera Quality**: Analysis accuracy depends on image quality
- **Lighting Conditions**: Poor lighting can affect color analysis
- **Image Resolution**: Higher resolution images provide better analysis
- **Lesion Segmentation**: Current algorithm uses basic contour detection
- **Calibration**: Area measurements are approximate without calibration

### Privacy and Security

- **Local Storage**: Images are stored locally on the user's device
- **No Cloud Upload**: Images are not automatically uploaded to cloud services
- **User Control**: Users have full control over their image data
- **Export Options**: Users can export their data including image paths

### Medical Disclaimer

This application is for tracking and monitoring purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## Future Enhancements

### Planned Features

- **Advanced Segmentation**: More sophisticated lesion detection algorithms
- **Machine Learning**: AI-powered analysis and classification
- **3D Analysis**: Depth and volume measurements
- **Comparative Analysis**: Side-by-side image comparison
- **Automated Reporting**: Generate reports with images and metrics
- **Cloud Integration**: Optional cloud storage and sharing

### Research Applications

- **Clinical Trials**: Standardized image collection and analysis
- **Telemedicine**: Remote skin condition assessment
- **Population Studies**: Large-scale skin condition tracking
- **Treatment Efficacy**: Automated treatment response measurement

## Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Ensure camera permissions are granted
   - Check if camera is being used by another application
   - Try restarting the application

2. **Image Analysis Fails**
   - Ensure image file is not corrupted
   - Check if image format is supported
   - Verify OpenCV and PIL are properly installed

3. **Poor Analysis Results**
   - Ensure good lighting when taking photos
   - Use high-resolution images
   - Position camera perpendicular to the skin surface
   - Avoid shadows and reflections

### Error Messages

- **"OpenCV not available"**: Install opencv-python
- **"PIL not available"**: Install Pillow
- **"Could not open camera"**: Check camera permissions and availability
- **"File not found"**: Verify the image file path is correct

## Support

For technical support or feature requests, please refer to the main SkinTrack+ documentation or contact the development team.

---

*This documentation covers the image capture and analysis features added to SkinTrack+. For general application usage, see the main README.md file.*
