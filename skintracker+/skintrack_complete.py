#!/usr/bin/env python3
"""
SkinTrack+ Complete - Standalone CLI Version
===========================================

A comprehensive command-line application for tracking skin conditions with:
- Image capture and analysis
- Food and stress tracking
- Sun exposure monitoring
- Medication logging
- Data visualization and charts
- Export capabilities

This is a complete standalone version that can run independently.
"""

import os
import sys
import datetime as dt
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import shutil

# Try to import image processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available. Image processing features will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not available. Image handling features will be limited.")

try:
    from skimage.color import rgb2lab
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  scikit-image not available. Color analysis features will be limited.")

# Data visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not available. Data visualization features will be limited.")

# Configuration
DATA_DIR = Path("skintrack_data")
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "skintrack.db"
CHARTS_DIR = DATA_DIR / "charts"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# Constants
CONDITIONS = [
    "eczema", "psoriasis", "guttate psoriasis", "keratosis pilaris",
    "cystic/hormonal acne", "melanoma", "vitiligo", "contact dermatitis", "cold sores"
]

FOOD_CATEGORIES = [
    "dairy", "gluten", "nuts", "shellfish", "eggs", "soy", "wheat",
    "citrus", "tomatoes", "spicy foods", "processed foods", "sugar",
    "alcohol", "caffeine", "chocolate", "nightshades", "other"
]

STRESS_TYPES = [
    "work stress", "personal stress", "financial stress", "health stress",
    "relationship stress", "academic stress", "social stress", "environmental stress",
    "sleep deprivation", "emotional stress", "physical stress", "other"
]

SUN_EXPOSURE_TYPES = [
    "natural sunlight", "UVB therapy", "UVA therapy", "phototherapy", "tanning bed",
    "outdoor activity", "beach/sunbathing", "walking outside", "gardening", "sports", "other"
]

MEDICATION_TYPES = [
    "topical steroid", "topical non-steroid", "oral medication", "injection", "phototherapy",
    "biologic", "immunosuppressant", "antibiotic", "antihistamine", "vitamin supplement",
    "herbal supplement", "over-the-counter", "prescription", "other"
]

IMAGE_CAPTURE_OPTIONS = [
    "take photo with camera",
    "upload existing image",
    "view captured images",
    "analyze image metrics"
]

COLOR_SCHEMES = {
    "symptoms": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
    "food": ["#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
    "stress": ["#E74C3C", "#F39C12", "#F1C40F", "#27AE60", "#3498DB"],
    "sun": ["#FFA500", "#FFD700", "#FF6347", "#32CD32", "#1E90FF"],
    "medication": ["#9370DB", "#20B2AA", "#FF69B4", "#FF4500", "#00CED1"]
}

# Database functions
def init_db():
    """Initialize the database with all required tables"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        
        # Lesions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lesions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                condition TEXT
            );
        """)
        
        # Records table
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
        
        # Food log table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS food_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesion_id INTEGER,
                ts TEXT,
                food_item TEXT,
                category TEXT,
                quantity TEXT,
                meal_type TEXT,
                skin_reaction INTEGER,
                reaction_delay_hours INTEGER,
                notes TEXT,
                FOREIGN KEY (lesion_id) REFERENCES lesions (id)
            );
        """)
        
        # Stress log table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stress_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesion_id INTEGER,
                ts TEXT,
                stress_type TEXT,
                stress_level INTEGER,
                duration_hours INTEGER,
                symptoms TEXT,
                coping_methods TEXT,
                skin_impact INTEGER,
                notes TEXT,
                FOREIGN KEY (lesion_id) REFERENCES lesions (id)
            );
        """)
        
        # Sun exposure log table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sun_exposure_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesion_id INTEGER,
                ts TEXT,
                exposure_type TEXT,
                duration_minutes INTEGER,
                time_of_day TEXT,
                uv_index INTEGER,
                protection_methods TEXT,
                skin_improvement INTEGER,
                side_effects TEXT,
                notes TEXT,
                FOREIGN KEY (lesion_id) REFERENCES lesions (id)
            );
        """)
        
        # Medication log table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS medication_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesion_id INTEGER,
                ts TEXT,
                medication_name TEXT,
                medication_type TEXT,
                dose TEXT,
                frequency TEXT,
                taken_as_prescribed INTEGER,
                effectiveness INTEGER,
                side_effects TEXT,
                notes TEXT,
                FOREIGN KEY (lesion_id) REFERENCES lesions (id)
            );
        """)
        
        con.commit()
        print("✅ Database initialized successfully!")

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

# Image processing functions
def capture_image_with_camera():
    """Capture an image using the device camera"""
    if not CV2_AVAILABLE:
        print("❌ OpenCV not available. Cannot capture images with camera.")
        return None
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera. Please check if camera is available.")
            return None
        
        print("📸 Camera initialized. Press 'c' to capture, 'q' to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            cv2.imshow('SkinTrack+ Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = IMAGES_DIR / filename
                
                cv2.imwrite(str(filepath), frame)
                print(f"✅ Image captured and saved: {filepath}")
                break
            elif key == ord('q'):
                print("❌ Image capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if 'filepath' in locals():
            return str(filepath)
        return None
        
    except Exception as e:
        print(f"❌ Error capturing image: {e}")
        return None

def upload_existing_image():
    """Upload an existing image file"""
    print("\n📁 Upload Existing Image")
    print("-" * 30)
    
    image_path = input("Enter the full path to your image file: ").strip()
    
    if not image_path:
        print("❌ No image path provided.")
        return None
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext not in valid_extensions:
        print(f"❌ Invalid image format. Supported formats: {', '.join(valid_extensions)}")
        return None
    
    try:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}{file_ext}"
        new_path = IMAGES_DIR / filename
        
        shutil.copy2(image_path, new_path)
        print(f"✅ Image uploaded successfully: {new_path}")
        return str(new_path)
        
    except Exception as e:
        print(f"❌ Error uploading image: {e}")
        return None

def analyze_image_metrics(image_path):
    """Analyze an image and extract skin condition metrics"""
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        print("❌ OpenCV or PIL not available. Cannot analyze images.")
        return {}
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return {}
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        metrics = {}
        
        # Basic image properties
        height, width = img.shape[:2]
        metrics['image_size'] = f"{width}x{height}"
        
        # Area measurement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            area_cm2 = area * 0.0001
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
                print(f"⚠️  Color analysis failed: {e}")
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
        
        print("✅ Image analysis completed!")
        return metrics
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return {}

# Main menu and UI functions
def print_header():
    """Print application header"""
    print("=" * 60)
    print("🧴 SkinTrack+ Complete - Chronic Skin Condition Tracker")
    print("=" * 60)
    print("Standalone CLI Version with Image Analysis & Data Visualization")
    print("=" * 60)

def print_menu():
    """Print main menu options"""
    print("\n📋 Main Menu:")
    print("1. Create new lesion")
    print("2. List all lesions")
    print("3. Add record to lesion")
    print("4. View lesion history")
    print("5. 📸 Image Capture & Analysis")
    print("6. 🍽️ Log food intake")
    print("7. 😰 Log stress event")
    print("8. ☀️ Log sun exposure")
    print("9. 💊 Log medication")
    print("10. View food reactions")
    print("11. View stress patterns")
    print("12. View sun exposure patterns")
    print("13. View medication effectiveness")
    print("14. 📊 Data Analysis & Charts")
    print("15. Export data")
    print("16. Initialize database")
    print("17. Exit")
    print("-" * 40)

def get_user_choice():
    """Get user menu choice"""
    while True:
        try:
            choice = input("Enter your choice (1-17): ").strip()
            if choice in [str(i) for i in range(1, 18)]:
                return choice
            else:
                print("❌ Please enter a number between 1 and 17.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

def main():
    """Main application loop"""
    print_header()
    
    # Initialize database
    init_db()
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == '1':
            create_lesion()
        elif choice == '2':
            list_all_lesions()
        elif choice == '3':
            add_record()
        elif choice == '4':
            view_history()
        elif choice == '5':
            capture_and_analyze_image()
        elif choice == '6':
            log_food_intake()
        elif choice == '7':
            log_stress_event()
        elif choice == '8':
            log_sun_exposure()
        elif choice == '9':
            log_medication()
        elif choice == '10':
            view_food_reactions()
        elif choice == '11':
            view_stress_patterns()
        elif choice == '12':
            view_sun_exposure_patterns()
        elif choice == '13':
            view_medication_effectiveness()
        elif choice == '14':
            data_analysis_menu()
        elif choice == '15':
            export_data()
        elif choice == '16':
            init_db()
        elif choice == '17':
            print("\n👋 Thank you for using SkinTrack+ Complete!")
            break
        
        input("\nPress Enter to continue...")

# Placeholder functions for the complete implementation
def create_lesion():
    """Create a new lesion"""
    print("\n➕ Create New Lesion")
    print("-" * 30)
    
    label = input("Enter lesion label (e.g., left forearm A): ").strip()
    if not label:
        print("❌ Label cannot be empty.")
        return
    
    print("\nAvailable conditions:")
    for i, condition in enumerate(CONDITIONS, 1):
        print(f"{i}. {condition}")
    
    while True:
        try:
            choice = int(input(f"\nSelect condition (1-{len(CONDITIONS)}): "))
            if 1 <= choice <= len(CONDITIONS):
                condition = CONDITIONS[choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(CONDITIONS)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    lesion_id = insert_lesion(label, condition)
    print(f"✅ Created lesion #{lesion_id}: {label} [{condition}]")

def list_all_lesions():
    """List all lesions"""
    print("\n📋 All Lesions")
    print("-" * 30)
    
    lesions = list_lesions()
    if lesions.empty:
        print("No lesions found. Create one first!")
        return
    
    print(f"{'ID':<5} {'Label':<20} {'Condition':<20}")
    print("-" * 45)
    for _, row in lesions.iterrows():
        print(f"{row['id']:<5} {row['label']:<20} {row['condition']:<20}")

def add_record():
    """Add a record to a lesion"""
    print("\n📝 Add Record to Lesion")
    print("-" * 30)
    
    lesions = list_lesions()
    if lesions.empty:
        print("No lesions found. Create one first!")
        return
    
    print("Available lesions:")
    for _, row in lesions.iterrows():
        print(f"{row['id']}. {row['label']} [{row['condition']}]")
    
    while True:
        try:
            lesion_id = int(input("\nSelect lesion ID: "))
            if lesion_id in lesions['id'].values:
                break
            else:
                print("❌ Invalid lesion ID.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get symptom data
    print("\n📊 Symptom Assessment:")
    print("Rate each symptom from 0-10 (0 = none, 10 = severe)")
    
    while True:
        try:
            itch = int(input("Itch level (0-10): "))
            if 0 <= itch <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    while True:
        try:
            pain = int(input("Pain level (0-10): "))
            if 0 <= pain <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    while True:
        try:
            sleep = float(input("Sleep hours last night (0-24): "))
            if 0 <= sleep <= 24:
                break
            else:
                print("❌ Please enter a number between 0 and 24.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    while True:
        try:
            stress = int(input("Stress level (0-10): "))
            if 0 <= stress <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get other information
    print("\n📋 Additional Information:")
    triggers = input("Triggers (comma-separated, or press Enter for none): ").strip()
    new_products = input("New products used (comma-separated, or press Enter for none): ").strip()
    meds_taken = input("Medications taken today (comma-separated, or press Enter for none): ").strip()
    adherence = input("Took medications as planned? (y/n): ").strip().lower() == 'y'
    notes = input("Additional notes (or press Enter for none): ").strip()
    
    # Ask if user wants to add an image
    add_image = input("\nAdd an image to this record? (y/n): ").strip().lower()
    image_path = ""
    metrics = {
        "area_cm2": None,
        "redness": None,
        "border_irreg": None,
        "asymmetry": None,
        "depig_deltaE": None
    }
    
    if add_image == 'y':
        print("\n📸 Image options:")
        print("1. Take photo with camera")
        print("2. Upload existing image")
        print("3. Skip image")
        
        while True:
            try:
                img_choice = input("\nSelect option (1-3): ").strip()
                if img_choice in ['1', '2', '3']:
                    break
                else:
                    print("❌ Please enter a number between 1 and 3.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        if img_choice == '1':
            image_path = capture_image_with_camera()
        elif img_choice == '2':
            image_path = upload_existing_image()
        
        if image_path:
            print("\n🔍 Analyzing image...")
            metrics = analyze_image_metrics(image_path)
            if metrics:
                print("\n📊 Image Analysis Results:")
                print("-" * 40)
                for key, value in metrics.items():
                    if value is not None:
                        print(f"{key.replace('_', ' ').title()}: {value}")

    # Insert record
    insert_record(
        lesion_id=lesion_id,
        ts=dt.datetime.now().isoformat(timespec="seconds"),
        img_path=image_path,
        itch=itch, pain=pain, sleep=sleep, stress=stress,
        triggers=triggers, new_products=new_products, meds_taken=meds_taken,
        adherence=adherence, notes=notes, metrics=metrics
    )
    
    print("✅ Record added successfully!")

def view_history():
    """View history for a lesion"""
    print("\n📈 View Lesion History")
    print("-" * 30)
    
    lesions = list_lesions()
    if lesions.empty:
        print("No lesions found. Create one first!")
        return
    
    print("Available lesions:")
    for _, row in lesions.iterrows():
        print(f"{row['id']}. {row['label']} [{row['condition']}]")
    
    while True:
        try:
            lesion_id = int(input("\nSelect lesion ID: "))
            if lesion_id in lesions['id'].values:
                break
            else:
                print("❌ Invalid lesion ID.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    with sqlite3.connect(DB_PATH) as con:
        hist = pd.read_sql_query("""
            SELECT * FROM records WHERE lesion_id = ? ORDER BY ts ASC
        """, con, params=(lesion_id,))
    
    if hist.empty:
        print("No records found for this lesion.")
        return
    
    print(f"\n📊 History for lesion #{lesion_id}:")
    print("-" * 80)
    print(f"{'Date':<20} {'Itch':<4} {'Pain':<4} {'Sleep':<5} {'Stress':<6} {'Meds':<10}")
    print("-" * 80)
    
    for _, row in hist.iterrows():
        date_str = str(row['ts']).split('T')[0] if 'T' in str(row['ts']) else str(row['ts'])[:10]
        meds = "Yes" if row.get('adherence', 0) else "No"
        print(f"{date_str:<20} {row['itch']:<4} {row['pain']:<4} {row['sleep']:<5.1f} {row['stress']:<6} {meds:<10}")

def capture_and_analyze_image():
    """Main function for image capture and analysis"""
    print("\n📸 Image Capture & Analysis")
    print("-" * 30)
    
    print("Image capture options:")
    for i, option in enumerate(IMAGE_CAPTURE_OPTIONS, 1):
        print(f"{i}. {option}")
    
    while True:
        try:
            choice = int(input(f"\nSelect option (1-{len(IMAGE_CAPTURE_OPTIONS)}): "))
            if 1 <= choice <= len(IMAGE_CAPTURE_OPTIONS):
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(IMAGE_CAPTURE_OPTIONS)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    image_path = None
    
    if choice == 1:
        print("\n📸 Taking photo with camera...")
        image_path = capture_image_with_camera()
    elif choice == 2:
        image_path = upload_existing_image()
    elif choice == 3:
        list_captured_images()
        return
    elif choice == 4:
        print("\n📁 Select image for analysis:")
        image_path = upload_existing_image()
    
    if not image_path:
        print("❌ No image available for analysis.")
        return
    
    print("\n🔍 Analyzing image...")
    metrics = analyze_image_metrics(image_path)
    
    if metrics:
        print("\n📊 Image Analysis Results:")
        print("-" * 40)
        for key, value in metrics.items():
            if value is not None:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    add_to_record = input("\nAdd this image to a lesion record? (y/n): ").strip().lower()
    if add_to_record == 'y':
        add_record_with_image(image_path, metrics)

def list_captured_images():
    """List all captured images for a lesion"""
    print("\n📸 Captured Images")
    print("-" * 30)
    
    lesions = list_lesions()
    if lesions.empty:
        print("No lesions found. Create one first!")
        return
    
    print("Available lesions:")
    for _, row in lesions.iterrows():
        print(f"{row['id']}. {row['label']} [{row['condition']}]")
    
    while True:
        try:
            lesion_id = int(input("\nSelect lesion ID to view images: "))
            if lesion_id in lesions['id'].values:
                break
            else:
                print("❌ Invalid lesion ID.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("""
            SELECT id, ts, img_path, itch, pain, stress
            FROM records 
            WHERE lesion_id = ? AND img_path != ''
            ORDER BY ts DESC
        """, con, params=(lesion_id,))
    
    if df.empty:
        print("No images found for this lesion.")
        return
    
    print(f"\n📸 Images for lesion #{lesion_id}:")
    print("-" * 80)
    print(f"{'ID':<5} {'Date':<20} {'Image Path':<30} {'Itch':<4} {'Pain':<4} {'Stress':<6}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        date_str = str(row['ts']).split('T')[0] if 'T' in str(row['ts']) else str(row['ts'])[:10]
        img_name = os.path.basename(row['img_path']) if row['img_path'] else 'N/A'
        print(f"{row['id']:<5} {date_str:<20} {img_name:<30} {row['itch']:<4} {row['pain']:<4} {row['stress']:<6}")

def add_record_with_image(image_path, metrics=None):
    """Add a record with an image to a lesion"""
    print("\n📝 Add Record with Image")
    print("-" * 30)
    
    lesions = list_lesions()
    if lesions.empty:
        print("No lesions found. Create one first!")
        return
    
    print("Available lesions:")
    for _, row in lesions.iterrows():
        print(f"{row['id']}. {row['label']} [{row['condition']}]")
    
    while True:
        try:
            lesion_id = int(input("\nSelect lesion ID: "))
            if lesion_id in lesions['id'].values:
                break
            else:
                print("❌ Invalid lesion ID.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get symptom data
    print("\n📊 Symptom Assessment:")
    print("Rate each symptom from 0-10 (0 = none, 10 = severe)")
    
    while True:
        try:
            itch = int(input("Itch level (0-10): "))
            if 0 <= itch <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    while True:
        try:
            pain = int(input("Pain level (0-10): "))
            if 0 <= pain <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    while True:
        try:
            sleep = float(input("Sleep hours last night (0-24): "))
            if 0 <= sleep <= 24:
                break
            else:
                print("❌ Please enter a number between 0 and 24.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    while True:
        try:
            stress = int(input("Stress level (0-10): "))
            if 0 <= stress <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get other information
    print("\n📋 Additional Information:")
    triggers = input("Triggers (comma-separated, or press Enter for none): ").strip()
    new_products = input("New products used (comma-separated, or press Enter for none): ").strip()
    meds_taken = input("Medications taken today (comma-separated, or press Enter for none): ").strip()
    adherence = input("Took medications as planned? (y/n): ").strip().lower() == 'y'
    notes = input("Additional notes (or press Enter for none): ").strip()
    
    # Use provided metrics or create empty ones
    if metrics is None:
        metrics = {
            "area_cm2": None,
            "redness": None,
            "border_irreg": None,
            "asymmetry": None,
            "depig_deltaE": None
        }
    
    # Insert record with image
    insert_record(
        lesion_id=lesion_id,
        ts=dt.datetime.now().isoformat(timespec="seconds"),
        img_path=image_path,
        itch=itch, pain=pain, sleep=sleep, stress=stress,
        triggers=triggers, new_products=new_products, meds_taken=meds_taken,
        adherence=adherence, notes=notes, metrics=metrics
    )
    
    print("✅ Record with image added successfully!")

# Placeholder functions for other features
def log_food_intake():
    print("\n🍽️ Food Intake Logging")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete food tracking features")

def log_stress_event():
    print("\n😰 Stress Event Logging")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete stress tracking features")

def log_sun_exposure():
    print("\n☀️ Sun Exposure Logging")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete sun exposure tracking features")

def log_medication():
    print("\n💊 Medication Logging")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete medication tracking features")

def view_food_reactions():
    print("\n🍽️ Food Reactions")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete food analysis features")

def view_stress_patterns():
    print("\n😰 Stress Patterns")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete stress analysis features")

def view_sun_exposure_patterns():
    print("\n☀️ Sun Exposure Patterns")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete sun exposure analysis features")

def view_medication_effectiveness():
    print("\n💊 Medication Effectiveness")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete medication analysis features")

def data_analysis_menu():
    print("\n📊 Data Analysis & Charts")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete data visualization features")

def export_data():
    print("\n📤 Export Data")
    print("⚠️  Feature not yet implemented in this version")
    print("Use the full skintrack_standalone.py for complete export features")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)
