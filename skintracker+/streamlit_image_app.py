#!/usr/bin/env python3
"""
SkinTrack+ Image Capture & Analysis - Streamlit Version
=======================================================

A standalone Streamlit application for capturing and analyzing skin condition images.
This can be deployed directly from GitHub to Streamlit Cloud.

Deployment Instructions:
1. Upload this file to your GitHub repository
2. Go to share.streamlit.io
3. Connect your GitHub repository
4. Set the main file path to: streamlit_image_app.py
5. Deploy!
"""

import streamlit as st
import os
import sys
import datetime as dt
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import base64
import io

# Streamlit Cloud deployment configuration
st.set_page_config(
    page_title="SkinTrack+ Image Analysis",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import image processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not available. Image processing features will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.warning("⚠️ Pillow not available. Image handling features will be limited.")

try:
    from skimage.color import rgb2lab
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    st.warning("⚠️ scikit-image not available. Color analysis features will be limited.")

# Data visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
    # Set matplotlib backend for Streamlit Cloud
    plt.switch_backend('Agg')
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("⚠️ matplotlib/seaborn not available. Data visualization features will be limited.")

# Configuration - Streamlit Cloud friendly
DATA_DIR = Path("/tmp/skintrack_data")  # Use /tmp for Streamlit Cloud
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "skintrack.db"
CHARTS_DIR = DATA_DIR / "charts"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# Initialize session state for Streamlit Cloud
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_metrics' not in st.session_state:
    st.session_state.current_metrics = None
if 'current_image_path' not in st.session_state:
    st.session_state.current_image_path = None

# Constants
CONDITIONS = [
    "eczema", "psoriasis", "guttate psoriasis", "keratosis pilaris",
    "cystic/hormonal acne", "melanoma", "vitiligo", "contact dermatitis", "cold sores"
]

IMAGE_CAPTURE_OPTIONS = [
    "Take photo with camera",
    "Upload existing image",
    "View captured images",
    "Analyze image metrics"
]

COLOR_SCHEMES = {
    "symptoms": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
    "analysis": ["#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]
}

# Database functions
def init_db():
    """Initialize the database with required tables"""
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lesions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    condition TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lesion_id INTEGER,
                    ts TEXT,
                    img_path TEXT,
                    itch INTEGER,
                    pain INTEGER,
                    sleep REAL,
                    stress INTEGER,
                    triggers TEXT,
                    new_products TEXT,
                    meds_taken TEXT,
                    adherence INTEGER,
                    notes TEXT,
                    area_cm2 REAL,
                    redness REAL,
                    border_irreg REAL,
                    asymmetry REAL,
                    depig_deltaE REAL
                );
            """)
            con.commit()
            st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        st.info("💡 This is normal for Streamlit Cloud deployment. Data will be stored in session.")

def list_lesions():
    """List all lesions in the database"""
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT id, label, condition FROM lesions ORDER BY id DESC", con)

def insert_lesion(label, condition):
    """Insert a new lesion into the database"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("INSERT INTO lesions(label, condition) VALUES(?, ?)", (label, condition))
        con.commit()
        return cur.lastrowid

def insert_record(lesion_id, ts, img_path, itch, pain, sleep, stress, triggers, new_products,
                  meds_taken, adherence, notes, metrics):
    """Insert a new record into the database"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO records(lesion_id, ts, img_path, itch, pain, sleep, stress, triggers, new_products,
                                meds_taken, adherence, notes, area_cm2, redness, border_irreg, asymmetry, depig_deltaE)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lesion_id, ts, img_path, int(itch), int(pain), float(sleep), int(stress),
            triggers, new_products, meds_taken, int(bool(adherence)), notes,
            metrics.get("area_cm2"), metrics.get("redness"), metrics.get("border_irreg"),
            metrics.get("asymmetry"), metrics.get("depig_deltaE")
        ))
        con.commit()

def get_records_with_images(lesion_id=None):
    """Get records with images"""
    with sqlite3.connect(DB_PATH) as con:
        if lesion_id:
            query = """
                SELECT id, ts, img_path, itch, pain, stress, area_cm2, redness, border_irreg, asymmetry
                FROM records 
                WHERE lesion_id = ? AND img_path != ''
                ORDER BY ts DESC
            """
            return pd.read_sql_query(query, con, params=(lesion_id,))
        else:
            query = """
                SELECT r.id, r.ts, r.img_path, r.itch, r.pain, r.stress, 
                       r.area_cm2, r.redness, r.border_irreg, r.asymmetry,
                       l.label, l.condition
                FROM records r
                JOIN lesions l ON r.lesion_id = l.id
                WHERE r.img_path != ''
                ORDER BY r.ts DESC
            """
            return pd.read_sql_query(query, con)

# Image processing functions
def analyze_image_metrics(image):
    """Analyze an image and extract skin condition metrics"""
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        st.error("❌ OpenCV or PIL not available. Cannot analyze images.")
        return {}
    
    try:
        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            # Convert PIL to numpy array
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Convert RGB to BGR for OpenCV
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
        else:
            img = image
        
        if img is None:
            st.error("❌ Could not load image")
            return {}
        
        # Convert BGR to RGB for analysis
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        metrics = {}
        
        # Basic image properties
        height, width = img.shape[:2]
        metrics['image_size'] = f"{width}x{height}"
        
        # Area measurement (simplified)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            area_cm2 = area * 0.0001  # Rough approximation
            metrics['area_cm2'] = round(area_cm2, 2)
        else:
            metrics['area_cm2'] = None
        
        # Redness analysis
        if SKIMAGE_AVAILABLE:
            try:
                img_lab = rgb2lab(img_rgb)
                a_channel = img_lab[:, :, 1]
                redness_score = np.mean(a_channel)
                metrics['redness'] = round(redness_score, 2)
                
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1])
                metrics['saturation'] = round(saturation, 2)
                
            except Exception as e:
                st.warning(f"⚠️ Color analysis failed: {e}")
                metrics['redness'] = None
                metrics['saturation'] = None
        else:
            metrics['redness'] = None
            metrics['saturation'] = None
        
        # Border irregularity
        if contours:
            perimeter = cv2.arcLength(largest_contour, True)
            if area > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                irregularity = 1 - circularity
                metrics['border_irreg'] = round(irregularity, 3)
            else:
                metrics['border_irreg'] = None
        else:
            metrics['border_irreg'] = None
        
        # Asymmetry analysis
        if contours:
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            asymmetry = abs(1 - aspect_ratio)
            metrics['asymmetry'] = round(asymmetry, 3)
        else:
            metrics['asymmetry'] = None
        
        # Texture analysis
        if len(img.shape) == 3:
            gray_std = np.std(gray)
            metrics['texture_variance'] = round(gray_std, 2)
        else:
            metrics['texture_variance'] = None
        
        return metrics
        
    except Exception as e:
        st.error(f"❌ Error analyzing image: {e}")
        return {}

def save_uploaded_image(uploaded_file):
    """Save uploaded image to local storage"""
    try:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{uploaded_file.name}"
        filepath = IMAGES_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(filepath)
    except Exception as e:
        st.error(f"❌ Error saving image: {e}")
        return None

# Streamlit UI functions
def main():
    st.title("📸 SkinTrack+ Image Capture & Analysis")
    st.markdown("---")
    
    # Initialize database
    if not st.session_state.db_initialized:
        init_db()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📸 Image Capture", "🔍 Image Analysis", "📊 View Images", "📝 Add Record", "📈 Data Analysis"]
    )
    
    # Add deployment info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **🚀 Deployed on Streamlit Cloud**
    
    This app is running on Streamlit Cloud for easy access and sharing.
    """)
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📸 Image Capture":
        show_image_capture_page()
    elif page == "🔍 Image Analysis":
        show_image_analysis_page()
    elif page == "📊 View Images":
        show_view_images_page()
    elif page == "📝 Add Record":
        show_add_record_page()
    elif page == "📈 Data Analysis":
        show_data_analysis_page()

def show_home_page():
    """Display the home page"""
    st.header("Welcome to SkinTrack+ Image Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is SkinTrack+?
        
        SkinTrack+ is a comprehensive skin condition tracking application that helps you:
        
        - **📸 Capture Images**: Take photos or upload existing images of skin conditions
        - **🔍 Analyze Metrics**: Get automated analysis of area, redness, border irregularity, and more
        - **📊 Track Progress**: Monitor changes over time with visual and quantitative data
        - **📝 Log Symptoms**: Record itch, pain, stress levels and other symptoms
        - **📈 Generate Reports**: Create charts and export data for healthcare providers
        
        ### 🚀 Getting Started
        
        1. **Create a Lesion**: Go to "Add Record" to create your first skin condition entry
        2. **Capture Images**: Use the "Image Capture" page to take or upload photos
        3. **Analyze Results**: View automated analysis on the "Image Analysis" page
        4. **Track Progress**: Monitor changes over time with the "Data Analysis" page
        """)
    
    with col2:
        st.info("""
        **💡 Tips for Best Results:**
        
        - Use good lighting (indirect daylight)
        - Keep camera 30-40 cm from skin
        - Take photos at consistent angles
        - Avoid shadows and reflections
        - Use high-resolution images
        """)
        
        # Quick stats
        lesions = list_lesions()
        records = get_records_with_images()
        
        st.metric("Total Lesions", len(lesions))
        st.metric("Images Captured", len(records))
        
        if len(records) > 0:
            st.metric("Latest Image", records.iloc[0]['ts'][:10])

def show_image_capture_page():
    """Display the image capture page"""
    st.header("📸 Image Capture")
    
    tab1, tab2, tab3 = st.tabs(["📷 Camera Capture", "📁 Upload Image", "📊 View Images"])
    
    with tab1:
        st.subheader("Take Photo with Camera")
        
        if not CV2_AVAILABLE:
            st.error("❌ Camera capture requires OpenCV. Please install opencv-python.")
            st.code("pip install opencv-python")
        else:
            st.info("Camera capture feature requires a local installation. For web deployment, use the upload feature.")
            
            # Placeholder for camera capture
            st.markdown("""
            **Camera Capture Instructions:**
            1. Run this application locally with `streamlit run streamlit_image_app.py`
            2. Use the camera capture feature
            3. For web deployment, use the upload feature instead
            """)
    
    with tab2:
        st.subheader("Upload Existing Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
            help="Upload an image of your skin condition for analysis"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save the image
            if st.button("💾 Save Image"):
                filepath = save_uploaded_image(uploaded_file)
                if filepath:
                    st.success(f"✅ Image saved successfully: {filepath}")
                    
                    # Store in session state for analysis
                    st.session_state['current_image'] = image
                    st.session_state['current_image_path'] = filepath
                    
                    st.info("🔄 Navigate to 'Image Analysis' to analyze this image!")
    
    with tab3:
        st.subheader("View Captured Images")
        show_image_gallery()

def show_image_analysis_page():
    """Display the image analysis page"""
    st.header("🔍 Image Analysis")
    
    # Check if we have an image to analyze
    if 'current_image' in st.session_state:
        st.subheader("Analyze Current Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(st.session_state['current_image'], caption="Image to Analyze", use_column_width=True)
        
        with col2:
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing image..."):
                    metrics = analyze_image_metrics(st.session_state['current_image'])
                
                if metrics:
                    st.success("✅ Analysis completed!")
                    
                    # Display metrics
                    st.subheader("📊 Analysis Results")
                    
                    # Create metrics display
                    cols = st.columns(2)
                    
                    with cols[0]:
                        if metrics.get('image_size'):
                            st.metric("Image Size", metrics['image_size'])
                        if metrics.get('area_cm2'):
                            st.metric("Area (cm²)", f"{metrics['area_cm2']}")
                        if metrics.get('redness'):
                            st.metric("Redness Score", f"{metrics['redness']}")
                    
                    with cols[1]:
                        if metrics.get('border_irreg'):
                            st.metric("Border Irregularity", f"{metrics['border_irreg']}")
                        if metrics.get('asymmetry'):
                            st.metric("Asymmetry", f"{metrics['asymmetry']}")
                        if metrics.get('texture_variance'):
                            st.metric("Texture Variance", f"{metrics['texture_variance']}")
                    
                    # Store metrics in session state
                    st.session_state['current_metrics'] = metrics
                    
                    st.info("💡 You can now add this image and analysis to a record!")
    
    # Upload new image for analysis
    st.subheader("Upload New Image for Analysis")
    uploaded_file = st.file_uploader(
        "Choose an image to analyze",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        key="analysis_uploader"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image for Analysis", use_column_width=True)
        
        if st.button("🔍 Analyze This Image"):
            with st.spinner("Analyzing image..."):
                metrics = analyze_image_metrics(image)
            
            if metrics:
                st.success("✅ Analysis completed!")
                
                # Display metrics in a nice format
                st.subheader("📊 Analysis Results")
                
                # Create a metrics dataframe for better display
                metrics_df = pd.DataFrame([
                    {"Metric": "Image Size", "Value": metrics.get('image_size', 'N/A')},
                    {"Metric": "Area (cm²)", "Value": f"{metrics.get('area_cm2', 'N/A')}"},
                    {"Metric": "Redness Score", "Value": f"{metrics.get('redness', 'N/A')}"},
                    {"Metric": "Border Irregularity", "Value": f"{metrics.get('border_irreg', 'N/A')}"},
                    {"Metric": "Asymmetry", "Value": f"{metrics.get('asymmetry', 'N/A')}"},
                    {"Metric": "Texture Variance", "Value": f"{metrics.get('texture_variance', 'N/A')}"},
                    {"Metric": "Saturation", "Value": f"{metrics.get('saturation', 'N/A')}"}
                ])
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Store for potential record creation
                st.session_state['current_image'] = image
                st.session_state['current_metrics'] = metrics

def show_view_images_page():
    """Display the image gallery page"""
    st.header("📊 Image Gallery")
    
    records = get_records_with_images()
    
    if records.empty:
        st.info("📸 No images found. Capture or upload some images first!")
        return
    
    # Filter by lesion if multiple lesions exist
    lesions = list_lesions()
    if len(lesions) > 1:
        selected_lesion = st.selectbox(
            "Select Lesion:",
            ["All Lesions"] + [f"{row['id']}: {row['label']} ({row['condition']})" for _, row in lesions.iterrows()]
        )
        
        if selected_lesion != "All Lesions":
            lesion_id = int(selected_lesion.split(':')[0])
            records = get_records_with_images(lesion_id)
    
    # Display images in a grid
    st.subheader(f"📸 Found {len(records)} images")
    
    # Create a grid layout
    cols = st.columns(3)
    
    for idx, (_, record) in enumerate(records.iterrows()):
        col_idx = idx % 3
        with cols[col_idx]:
            try:
                # Try to load and display the image
                if os.path.exists(record['img_path']):
                    image = Image.open(record['img_path'])
                    st.image(image, caption=f"Record #{record['id']}", use_column_width=True)
                    
                    # Display metrics
                    metrics_text = f"""
                    **Date:** {record['ts'][:10]}
                    **Itch:** {record['itch']}/10
                    **Pain:** {record['pain']}/10
                    **Stress:** {record['stress']}/10
                    """
                    
                    if record['area_cm2']:
                        metrics_text += f"**Area:** {record['area_cm2']} cm²\n"
                    if record['redness']:
                        metrics_text += f"**Redness:** {record['redness']}\n"
                    
                    st.markdown(metrics_text)
                else:
                    st.error(f"Image not found: {record['img_path']}")
                    
            except Exception as e:
                st.error(f"Error loading image: {e}")

def show_add_record_page():
    """Display the add record page"""
    st.header("📝 Add Record with Image")
    
    # Create lesion if needed
    st.subheader("1. Create or Select Lesion")
    
    lesions = list_lesions()
    
    if lesions.empty:
        st.info("No lesions found. Let's create your first one!")
        
        col1, col2 = st.columns(2)
        with col1:
            label = st.text_input("Lesion Label", placeholder="e.g., left forearm A")
        with col2:
            condition = st.selectbox("Condition Type", CONDITIONS)
        
        if st.button("➕ Create Lesion") and label:
            lesion_id = insert_lesion(label, condition)
            st.success(f"✅ Created lesion #{lesion_id}: {label} [{condition}]")
            st.rerun()
    else:
        selected_lesion = st.selectbox(
            "Select Lesion:",
            [f"{row['id']}: {row['label']} ({row['condition']})" for _, row in lesions.iterrows()]
        )
        lesion_id = int(selected_lesion.split(':')[0])
    
    # Symptom assessment
    st.subheader("2. Symptom Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        itch = st.slider("Itch Level (0-10)", 0, 10, 5)
        pain = st.slider("Pain Level (0-10)", 0, 10, 5)
    
    with col2:
        sleep = st.number_input("Sleep Hours Last Night", 0.0, 24.0, 8.0, 0.5)
        stress = st.slider("Stress Level (0-10)", 0, 10, 5)
    
    # Additional information
    st.subheader("3. Additional Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        triggers = st.text_input("Triggers (comma-separated)", placeholder="stress, sweat, fragrance")
        new_products = st.text_input("New Products Used", placeholder="new soap, lotion")
    
    with col2:
        meds_taken = st.text_input("Medications Taken", placeholder="triamcinolone, antihistamine")
        adherence = st.checkbox("Took medications as planned")
    
    notes = st.text_area("Additional Notes", placeholder="Any other observations...")
    
    # Image upload
    st.subheader("4. Add Image (Optional)")
    
    uploaded_file = st.file_uploader(
        "Upload an image for this record",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'],
        key="record_uploader"
    )
    
    image_path = ""
    metrics = {}
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image for Record", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                metrics = analyze_image_metrics(image)
            
            if metrics:
                st.success("✅ Image analysis completed!")
                
                # Display key metrics
                cols = st.columns(3)
                with cols[0]:
                    if metrics.get('area_cm2'):
                        st.metric("Area", f"{metrics['area_cm2']} cm²")
                with cols[1]:
                    if metrics.get('redness'):
                        st.metric("Redness", f"{metrics['redness']}")
                with cols[2]:
                    if metrics.get('border_irreg'):
                        st.metric("Irregularity", f"{metrics['border_irreg']}")
        
        # Save image
        if st.button("💾 Save Image"):
            image_path = save_uploaded_image(uploaded_file)
            if image_path:
                st.success("✅ Image saved!")
    
    # Submit record
    st.subheader("5. Save Record")
    
    if st.button("💾 Save Record", type="primary"):
        if 'lesion_id' in locals():
            # Insert record
            insert_record(
                lesion_id=lesion_id,
                ts=dt.datetime.now().isoformat(timespec="seconds"),
                img_path=image_path,
                itch=itch, pain=pain, sleep=sleep, stress=stress,
                triggers=triggers, new_products=new_products, meds_taken=meds_taken,
                adherence=adherence, notes=notes, metrics=metrics
            )
            
            st.success("✅ Record saved successfully!")
            st.balloons()
            
            # Clear form
            st.rerun()
        else:
            st.error("❌ Please create or select a lesion first!")

def show_data_analysis_page():
    """Display the data analysis page"""
    st.header("📈 Data Analysis")
    
    records = get_records_with_images()
    
    if records.empty:
        st.info("📊 No data available for analysis. Add some records with images first!")
        return
    
    # Basic statistics
    st.subheader("📊 Basic Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(records))
    with col2:
        avg_itch = records['itch'].mean()
        st.metric("Avg Itch Level", f"{avg_itch:.1f}/10")
    with col3:
        avg_pain = records['pain'].mean()
        st.metric("Avg Pain Level", f"{avg_pain:.1f}/10")
    with col4:
        avg_stress = records['stress'].mean()
        st.metric("Avg Stress Level", f"{avg_stress:.1f}/10")
    
    # Time series analysis
    st.subheader("📈 Trends Over Time")
    
    if len(records) > 1:
        # Convert timestamp to datetime
        records['ts'] = pd.to_datetime(records['ts'])
        records = records.sort_values('ts')
        
        # Create time series chart
        if MATPLOTLIB_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Symptom trends
            ax1.plot(records['ts'], records['itch'], 'o-', label='Itch', color=COLOR_SCHEMES['symptoms'][0])
            ax1.plot(records['ts'], records['pain'], 's-', label='Pain', color=COLOR_SCHEMES['symptoms'][1])
            ax1.plot(records['ts'], records['stress'], '^-', label='Stress', color=COLOR_SCHEMES['symptoms'][2])
            ax1.set_title('Symptom Trends Over Time')
            ax1.set_ylabel('Level (0-10)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Area trends (if available)
            if 'area_cm2' in records.columns and records['area_cm2'].notna().any():
                ax2.plot(records['ts'], records['area_cm2'], 'o-', label='Area', color=COLOR_SCHEMES['analysis'][0])
                ax2.set_title('Lesion Area Over Time')
                ax2.set_ylabel('Area (cm²)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("📊 Install matplotlib for trend visualization")
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    if len(records) > 2:
        # Create correlation matrix
        numeric_cols = ['itch', 'pain', 'stress', 'sleep']
        if 'area_cm2' in records.columns:
            numeric_cols.append('area_cm2')
        if 'redness' in records.columns:
            numeric_cols.append('redness')
        
        correlation_data = records[numeric_cols].corr()
        
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Symptom Correlations')
            st.pyplot(fig)
        else:
            st.dataframe(correlation_data, use_container_width=True)
    
    # Export data
    st.subheader("📤 Export Data")
    
    if st.button("📥 Download CSV"):
        csv = records.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="skintrack_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

def show_image_gallery():
    """Helper function to show image gallery"""
    records = get_records_with_images()
    
    if records.empty:
        st.info("📸 No images found. Upload some images first!")
        return
    
    # Display images in a grid
    cols = st.columns(3)
    
    for idx, (_, record) in enumerate(records.iterrows()):
        col_idx = idx % 3
        with cols[col_idx]:
            try:
                if os.path.exists(record['img_path']):
                    image = Image.open(record['img_path'])
                    st.image(image, caption=f"Record #{record['id']}", use_column_width=True)
                    
                    # Display basic info
                    st.markdown(f"""
                    **Date:** {record['ts'][:10]}
                    **Itch:** {record['itch']}/10
                    **Pain:** {record['pain']}/10
                    """)
                else:
                    st.error(f"Image not found")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
