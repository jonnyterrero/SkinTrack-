#!/usr/bin/env python3
"""
SkinTrack+ - Chronic Skin Condition Tracker (Standalone Python Version)
=======================================================================
This is a standalone version that can run directly with: python skintrack_standalone.py

It provides a simple command-line interface for the core functionality.
For the full web interface, use: streamlit run skintrack_app.py
"""

import io
import os
import base64
import datetime as dt
import sqlite3
from pathlib import Path
from typing import Optional, Tuple
import sys

import numpy as np
import pandas as pd

# Try to import OpenCV, but provide fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available. Image processing features will be limited.")

# Try to import other image processing libraries
try:
    from skimage.color import rgb2lab, deltaE_ciede2000
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  scikit-image not available. Color analysis features will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow not available. Image handling features will be limited.")

# Data visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    # Set style for better-looking charts
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not available. Data visualization features will be limited.")

# Optional ML imports
try:
    import torch
    import torchvision.transforms as T
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ---------------------------
# Config & setup
# ---------------------------
DATA_DIR = Path("skintrack_data")
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "skintrack.db"
MODEL_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CONDITIONS = [
    "eczema",
    "psoriasis",
    "guttate psoriasis",
    "keratosis pilaris",
    "cystic/hormonal acne",
    "melanoma",
    "vitiligo",
    "contact dermatitis",
    "cold sores",
]

TRIGGER_SUGGESTIONS = [
    "stress", "sweat/exercise", "fragrance", "detergent",
    "cosmetics", "weather - cold/dry", "weather - hot/humid",
    "pollen", "dust mites", "pet dander", "new products",
]

# Food categories and common triggers
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

STRESS_SYMPTOMS = [
    "increased itching", "flare-ups", "redness", "swelling", "pain",
    "sleep problems", "anxiety", "depression", "fatigue", "irritability",
    "concentration issues", "other"
]

# Sun exposure tracking
SUN_EXPOSURE_TYPES = [
    "natural sunlight", "UVB therapy", "UVA therapy", "phototherapy", "tanning bed",
    "outdoor activity", "beach/sunbathing", "walking outside", "gardening", "sports",
    "other"
]

SUN_PROTECTION_METHODS = [
    "sunscreen", "protective clothing", "hat", "sunglasses", "shade", "umbrella",
    "long sleeves", "pants", "no protection", "other"
]

# Medication tracking
MEDICATION_TYPES = [
    "topical steroid", "topical non-steroid", "oral medication", "injection", "phototherapy",
    "biologic", "immunosuppressant", "antibiotic", "antihistamine", "vitamin supplement",
    "herbal supplement", "over-the-counter", "prescription", "other"
]

MEDICATION_FREQUENCIES = [
    "once daily", "twice daily", "three times daily", "as needed", "weekly",
    "bi-weekly", "monthly", "before meals", "after meals", "at bedtime",
    "other"
]

# Data visualization and analysis options
CHART_TYPES = [
    "line chart", "bar chart", "pie chart", "scatter plot", "heatmap", "box plot",
    "histogram", "area chart", "radar chart", "trend analysis"
]

ANALYSIS_TYPES = [
    "symptom trends", "food reactions", "stress patterns", "sun exposure effectiveness",
    "medication performance", "trigger analysis", "treatment correlation", "overall summary"
]

TIME_PERIODS = [
    "last 7 days", "last 30 days", "last 90 days", "last 6 months", "last year", "all time"
]

# Color schemes for different data types
COLOR_SCHEMES = {
    "symptoms": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
    "food": ["#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
    "stress": ["#E74C3C", "#F39C12", "#F1C40F", "#27AE60", "#3498DB"],
    "sun": ["#FFA500", "#FFD700", "#FF6347", "#32CD32", "#1E90FF"],
    "medication": ["#9370DB", "#20B2AA", "#FF69B4", "#FF4500", "#00CED1"]
}

# Image capture and processing options
IMAGE_CAPTURE_OPTIONS = [
    "take photo with camera",
    "upload existing image",
    "view captured images",
    "analyze image metrics"
]

IMAGE_ANALYSIS_METRICS = [
    "area measurement",
    "redness analysis", 
    "border irregularity",
    "asymmetry analysis",
    "color analysis",
    "texture analysis"
]

# ---------------------------
# DB helpers
# ---------------------------
def init_db():
    """Initialize the database with required tables"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lesions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                condition TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS records(
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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS med_schedule(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lesion_id INTEGER,
                name TEXT,
                dose TEXT,
                morning INTEGER,
                afternoon INTEGER,
                evening INTEGER,
                notes TEXT
            );
        """)
        # New table for food logging
        cur.execute("""
            CREATE TABLE IF NOT EXISTS food_log(
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
        # New table for detailed stress tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stress_log(
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
        # New table for sun exposure tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sun_exposure_log(
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
        # New table for medication logging
        cur.execute("""
            CREATE TABLE IF NOT EXISTS medication_log(
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
            INSERT INTO records(
                lesion_id, ts, img_path, itch, pain, sleep, stress, triggers, new_products,
                meds_taken, adherence, notes, area_cm2, redness, border_irreg, asymmetry, depig_deltaE
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            lesion_id, ts, img_path, int(itch), int(pain), float(sleep), int(stress),
            triggers, new_products, meds_taken, int(bool(adherence)), notes,
            metrics.get("area_cm2"), metrics.get("redness"), metrics.get("border_irreg"),
            metrics.get("asymmetry"), metrics.get("depig_deltaE")
        ))
        con.commit()

def lesion_history(lesion_id):
    """Get history for a specific lesion"""
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query("""
            SELECT * FROM records WHERE lesion_id=? ORDER BY ts ASC
        """, con, params=(lesion_id,))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def med_schedule_for(lesion_id):
    """Get medication schedule for a specific lesion"""
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("SELECT * FROM med_schedule WHERE lesion_id=? ORDER BY id ASC", con, params=(lesion_id,))

def upsert_med_schedule(lesion_id, name, dose, morning, afternoon, evening, notes):
    """Add or update medication schedule"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO med_schedule(lesion_id, name, dose, morning, afternoon, evening, notes)
            VALUES(?,?,?,?,?,?,?)
        """, (lesion_id, name, dose, int(bool(morning)), int(bool(afternoon)), int(bool(evening)), notes))
        con.commit()

# Food logging functions
def insert_food_log(lesion_id, ts, food_item, category, quantity, meal_type, skin_reaction, reaction_delay_hours, notes):
    """Insert a new food log entry"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO food_log(lesion_id, ts, food_item, category, quantity, meal_type, 
                                skin_reaction, reaction_delay_hours, notes)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (lesion_id, ts, food_item, category, quantity, meal_type, 
              skin_reaction, reaction_delay_hours, notes))
        con.commit()

def get_food_history(lesion_id, days=30):
    """Get food history for a specific lesion"""
    with sqlite3.connect(DB_PATH) as con:
        cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
        df = pd.read_sql_query("""
            SELECT * FROM food_log 
            WHERE lesion_id = ? AND ts >= ?
            ORDER BY ts DESC
        """, con, params=(lesion_id, cutoff_date))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def get_food_reactions(lesion_id):
    """Get food items that caused skin reactions"""
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("""
            SELECT food_item, category, COUNT(*) as reaction_count, 
                   AVG(skin_reaction) as avg_reaction_level,
                   AVG(reaction_delay_hours) as avg_delay
            FROM food_log 
            WHERE lesion_id = ? AND skin_reaction > 0
            GROUP BY food_item, category
            ORDER BY reaction_count DESC, avg_reaction_level DESC
        """, con, params=(lesion_id,))

# Stress tracking functions
def insert_stress_log(lesion_id, ts, stress_type, stress_level, duration_hours, symptoms, coping_methods, skin_impact, notes):
    """Insert a new stress log entry"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO stress_log(lesion_id, ts, stress_type, stress_level, duration_hours,
                                  symptoms, coping_methods, skin_impact, notes)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (lesion_id, ts, stress_type, stress_level, duration_hours,
              symptoms, coping_methods, skin_impact, notes))
        con.commit()

def get_stress_history(lesion_id, days=30):
    """Get stress history for a specific lesion"""
    with sqlite3.connect(DB_PATH) as con:
        cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
        df = pd.read_sql_query("""
            SELECT * FROM stress_log 
            WHERE lesion_id = ? AND ts >= ?
            ORDER BY ts DESC
        """, con, params=(lesion_id, cutoff_date))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def get_stress_patterns(lesion_id):
    """Get stress patterns and their impact on skin"""
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("""
            SELECT stress_type, COUNT(*) as occurrence_count,
                   AVG(stress_level) as avg_stress_level,
                   AVG(skin_impact) as avg_skin_impact,
                   AVG(duration_hours) as avg_duration
            FROM stress_log 
            WHERE lesion_id = ?
            GROUP BY stress_type
            ORDER BY occurrence_count DESC, avg_skin_impact DESC
        """, con, params=(lesion_id,))

# Sun exposure tracking functions
def insert_sun_exposure_log(lesion_id, ts, exposure_type, duration_minutes, time_of_day, uv_index, protection_methods, skin_improvement, side_effects, notes):
    """Insert a new sun exposure log entry"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO sun_exposure_log(lesion_id, ts, exposure_type, duration_minutes, time_of_day,
                                        uv_index, protection_methods, skin_improvement, side_effects, notes)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (lesion_id, ts, exposure_type, duration_minutes, time_of_day,
              uv_index, protection_methods, skin_improvement, side_effects, notes))
        con.commit()

def get_sun_exposure_history(lesion_id, days=30):
    """Get sun exposure history for a specific lesion"""
    with sqlite3.connect(DB_PATH) as con:
        cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
        df = pd.read_sql_query("""
            SELECT * FROM sun_exposure_log 
            WHERE lesion_id = ? AND ts >= ?
            ORDER BY ts DESC
        """, con, params=(lesion_id, cutoff_date))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def get_sun_exposure_patterns(lesion_id):
    """Get sun exposure patterns and their effectiveness"""
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("""
            SELECT exposure_type, COUNT(*) as exposure_count,
                   AVG(duration_minutes) as avg_duration,
                   AVG(skin_improvement) as avg_improvement,
                   AVG(uv_index) as avg_uv_index
            FROM sun_exposure_log 
            WHERE lesion_id = ?
            GROUP BY exposure_type
            ORDER BY exposure_count DESC, avg_improvement DESC
        """, con, params=(lesion_id,))

# Medication logging functions
def insert_medication_log(lesion_id, ts, medication_name, medication_type, dose, frequency, taken_as_prescribed, effectiveness, side_effects, notes):
    """Insert a new medication log entry"""
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO medication_log(lesion_id, ts, medication_name, medication_type, dose, frequency,
                                      taken_as_prescribed, effectiveness, side_effects, notes)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (lesion_id, ts, medication_name, medication_type, dose, frequency,
              taken_as_prescribed, effectiveness, side_effects, notes))
        con.commit()

def get_medication_history(lesion_id, days=30):
    """Get medication history for a specific lesion"""
    with sqlite3.connect(DB_PATH) as con:
        cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
        df = pd.read_sql_query("""
            SELECT * FROM medication_log 
            WHERE lesion_id = ? AND ts >= ?
            ORDER BY ts DESC
        """, con, params=(lesion_id, cutoff_date))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def get_medication_effectiveness(lesion_id):
    """Get medication effectiveness analysis"""
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query("""
            SELECT medication_name, medication_type, COUNT(*) as usage_count,
                   AVG(effectiveness) as avg_effectiveness,
                   AVG(taken_as_prescribed) as avg_adherence,
                   COUNT(CASE WHEN side_effects != '' THEN 1 END) as side_effect_count
            FROM medication_log 
            WHERE lesion_id = ?
            GROUP BY medication_name, medication_type
            ORDER BY avg_effectiveness DESC, usage_count DESC
        """, con, params=(lesion_id,))

# Data analysis and visualization functions
def get_symptom_statistics(lesion_id, days=30):
    """Get comprehensive symptom statistics"""
    with sqlite3.connect(DB_PATH) as con:
        cutoff_date = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
        df = pd.read_sql_query("""
            SELECT itch, pain, sleep, stress, adherence,
                   area_cm2, redness, border_irreg, asymmetry, depig_deltaE
            FROM records 
            WHERE lesion_id = ? AND ts >= ?
        """, con, params=(lesion_id, cutoff_date))
    
    if df.empty:
        return None
    
    stats = {
        'itch': {
            'mean': df['itch'].mean(),
            'median': df['itch'].median(),
            'std': df['itch'].std(),
            'min': df['itch'].min(),
            'max': df['itch'].max(),
            'count': len(df)
        },
        'pain': {
            'mean': df['pain'].mean(),
            'median': df['pain'].median(),
            'std': df['pain'].std(),
            'min': df['pain'].min(),
            'max': df['pain'].max(),
            'count': len(df)
        },
        'sleep': {
            'mean': df['sleep'].mean(),
            'median': df['sleep'].median(),
            'std': df['sleep'].std(),
            'min': df['sleep'].min(),
            'max': df['sleep'].max(),
            'count': len(df)
        },
        'stress': {
            'mean': df['stress'].mean(),
            'median': df['stress'].median(),
            'std': df['stress'].std(),
            'min': df['stress'].min(),
            'max': df['stress'].max(),
            'count': len(df)
        }
    }
    return stats

def get_overall_summary(lesion_id, days=30):
    """Get overall summary statistics for all data types"""
    summary = {}
    
    # Symptom statistics
    symptom_stats = get_symptom_statistics(lesion_id, days)
    if symptom_stats:
        summary['symptoms'] = symptom_stats
    
    # Food statistics
    food_data = get_food_history(lesion_id, days)
    if not food_data.empty:
        summary['food'] = {
            'total_entries': len(food_data),
            'avg_reaction': food_data['skin_reaction'].mean(),
            'reactions_count': len(food_data[food_data['skin_reaction'] > 0]),
            'top_categories': food_data['category'].value_counts().head(5).to_dict()
        }
    
    # Stress statistics
    stress_data = get_stress_history(lesion_id, days)
    if not stress_data.empty:
        summary['stress'] = {
            'total_events': len(stress_data),
            'avg_stress_level': stress_data['stress_level'].mean(),
            'avg_skin_impact': stress_data['skin_impact'].mean(),
            'top_stress_types': stress_data['stress_type'].value_counts().head(5).to_dict()
        }
    
    # Sun exposure statistics
    sun_data = get_sun_exposure_history(lesion_id, days)
    if not sun_data.empty:
        summary['sun_exposure'] = {
            'total_sessions': len(sun_data),
            'avg_duration': sun_data['duration_minutes'].mean(),
            'avg_improvement': sun_data['skin_improvement'].mean(),
            'avg_uv_index': sun_data['uv_index'].mean(),
            'top_exposure_types': sun_data['exposure_type'].value_counts().head(5).to_dict()
        }
    
    # Medication statistics
    med_data = get_medication_history(lesion_id, days)
    if not med_data.empty:
        summary['medication'] = {
            'total_entries': len(med_data),
            'avg_effectiveness': med_data['effectiveness'].mean(),
            'avg_adherence': med_data['taken_as_prescribed'].mean(),
            'top_medications': med_data['medication_name'].value_counts().head(5).to_dict()
        }
    
    return summary

def create_symptom_trend_chart(lesion_id, days=30, save_path=None):
    """Create a line chart showing symptom trends over time"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available for chart creation")
        return
    
    hist = lesion_history(lesion_id)
    if hist.empty:
        print("❌ No data available for charting")
        return
    
    # Filter by time period
    cutoff_date = dt.datetime.now() - dt.timedelta(days=days)
    hist['ts'] = pd.to_datetime(hist['ts'])
    hist = hist[hist['ts'] >= cutoff_date]
    
    if hist.empty:
        print("❌ No data in the specified time period")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot symptom trends
    ax.plot(hist['ts'], hist['itch'], 'o-', label='Itch Level', color=COLOR_SCHEMES['symptoms'][0], linewidth=2)
    ax.plot(hist['ts'], hist['pain'], 's-', label='Pain Level', color=COLOR_SCHEMES['symptoms'][1], linewidth=2)
    ax.plot(hist['ts'], hist['stress'], '^-', label='Stress Level', color=COLOR_SCHEMES['symptoms'][2], linewidth=2)
    
    # Format the chart
    ax.set_title(f'Symptom Trends - Lesion #{lesion_id} (Last {days} days)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Level (0-10)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days//7)))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_food_reaction_chart(lesion_id, days=30, save_path=None):
    """Create a bar chart showing food reactions"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available for chart creation")
        return
    
    reactions = get_food_reactions(lesion_id)
    if reactions.empty:
        print("❌ No food reaction data available")
        return
    
    # Get recent food data for time filtering
    food_data = get_food_history(lesion_id, days)
    if food_data.empty:
        print("❌ No food data in the specified time period")
        return
    
    # Filter reactions to only include foods from the time period
    recent_foods = food_data['food_item'].unique()
    reactions = reactions[reactions['food_item'].isin(recent_foods)]
    
    if reactions.empty:
        print("❌ No food reactions in the specified time period")
        return
    
    # Take top 10 foods by reaction count
    top_reactions = reactions.head(10)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bar chart of reaction counts
    bars1 = ax1.bar(range(len(top_reactions)), top_reactions['reaction_count'], 
                    color=COLOR_SCHEMES['food'][:len(top_reactions)])
    ax1.set_title(f'Top Food Reactions - Lesion #{lesion_id} (Last {days} days)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Reactions', fontsize=12)
    ax1.set_xticks(range(len(top_reactions)))
    ax1.set_xticklabels(top_reactions['food_item'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars1, top_reactions['reaction_count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(int(count)), ha='center', va='bottom')
    
    # Bar chart of average reaction levels
    bars2 = ax2.bar(range(len(top_reactions)), top_reactions['avg_reaction_level'], 
                    color=COLOR_SCHEMES['food'][len(top_reactions):len(top_reactions)*2])
    ax2.set_title('Average Reaction Levels', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Reaction Level (0-10)', fontsize=12)
    ax2.set_xticks(range(len(top_reactions)))
    ax2.set_xticklabels(top_reactions['food_item'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, level in zip(bars2, top_reactions['avg_reaction_level']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{level:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_stress_pattern_chart(lesion_id, days=30, save_path=None):
    """Create charts showing stress patterns"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available for chart creation")
        return
    
    patterns = get_stress_patterns(lesion_id)
    if patterns.empty:
        print("❌ No stress pattern data available")
        return
    
    # Get recent stress data for time filtering
    stress_data = get_stress_history(lesion_id, days)
    if stress_data.empty:
        print("❌ No stress data in the specified time period")
        return
    
    # Filter patterns to only include stress types from the time period
    recent_stress_types = stress_data['stress_type'].unique()
    patterns = patterns[patterns['stress_type'].isin(recent_stress_types)]
    
    if patterns.empty:
        print("❌ No stress patterns in the specified time period")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pie chart of stress types
    top_stress = patterns.head(8)  # Top 8 stress types
    ax1.pie(top_stress['occurrence_count'], labels=top_stress['stress_type'], 
            autopct='%1.1f%%', colors=COLOR_SCHEMES['stress'][:len(top_stress)])
    ax1.set_title('Stress Type Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart of stress levels
    bars = ax2.bar(range(len(top_stress)), top_stress['avg_stress_level'], 
                   color=COLOR_SCHEMES['stress'][len(top_stress):len(top_stress)*2])
    ax2.set_title('Average Stress Levels by Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Stress Level (0-10)', fontsize=12)
    ax2.set_xticks(range(len(top_stress)))
    ax2.set_xticklabels(top_stress['stress_type'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, level in zip(bars, top_stress['avg_stress_level']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{level:.1f}', ha='center', va='bottom')
    
    # Bar chart of skin impact
    bars = ax3.bar(range(len(top_stress)), top_stress['avg_skin_impact'], 
                   color=COLOR_SCHEMES['stress'][len(top_stress)*2:len(top_stress)*3])
    ax3.set_title('Average Skin Impact by Stress Type', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average Skin Impact (0-10)', fontsize=12)
    ax3.set_xticks(range(len(top_stress)))
    ax3.set_xticklabels(top_stress['stress_type'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, impact in zip(bars, top_stress['avg_skin_impact']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{impact:.1f}', ha='center', va='bottom')
    
    # Scatter plot of stress level vs skin impact
    ax4.scatter(top_stress['avg_stress_level'], top_stress['avg_skin_impact'], 
               s=100, c=COLOR_SCHEMES['stress'][:len(top_stress)], alpha=0.7)
    ax4.set_title('Stress Level vs Skin Impact Correlation', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Average Stress Level', fontsize=12)
    ax4.set_ylabel('Average Skin Impact', fontsize=12)
    
    # Add labels for each point
    for i, stress_type in enumerate(top_stress['stress_type']):
        ax4.annotate(stress_type, (top_stress.iloc[i]['avg_stress_level'], 
                                  top_stress.iloc[i]['avg_skin_impact']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_sun_exposure_chart(lesion_id, days=30, save_path=None):
    """Create charts showing sun exposure patterns"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available for chart creation")
        return
    
    patterns = get_sun_exposure_patterns(lesion_id)
    if patterns.empty:
        print("❌ No sun exposure pattern data available")
        return
    
    # Get recent sun exposure data for time filtering
    sun_data = get_sun_exposure_history(lesion_id, days)
    if sun_data.empty:
        print("❌ No sun exposure data in the specified time period")
        return
    
    # Filter patterns to only include exposure types from the time period
    recent_exposure_types = sun_data['exposure_type'].unique()
    patterns = patterns[patterns['exposure_type'].isin(recent_exposure_types)]
    
    if patterns.empty:
        print("❌ No sun exposure patterns in the specified time period")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pie chart of exposure types
    top_exposure = patterns.head(8)  # Top 8 exposure types
    ax1.pie(top_exposure['exposure_count'], labels=top_exposure['exposure_type'], 
            autopct='%1.1f%%', colors=COLOR_SCHEMES['sun'][:len(top_exposure)])
    ax1.set_title('Sun Exposure Type Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart of average improvement
    bars = ax2.bar(range(len(top_exposure)), top_exposure['avg_improvement'], 
                   color=COLOR_SCHEMES['sun'][len(top_exposure):len(top_exposure)*2])
    ax2.set_title('Average Skin Improvement by Exposure Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Improvement (0-10)', fontsize=12)
    ax2.set_xticks(range(len(top_exposure)))
    ax2.set_xticklabels(top_exposure['exposure_type'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, improvement in zip(bars, top_exposure['avg_improvement']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{improvement:.1f}', ha='center', va='bottom')
    
    # Bar chart of average duration
    bars = ax3.bar(range(len(top_exposure)), top_exposure['avg_duration'], 
                   color=COLOR_SCHEMES['sun'][len(top_exposure)*2:len(top_exposure)*3])
    ax3.set_title('Average Duration by Exposure Type', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average Duration (minutes)', fontsize=12)
    ax3.set_xticks(range(len(top_exposure)))
    ax3.set_xticklabels(top_exposure['exposure_type'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, duration in zip(bars, top_exposure['avg_duration']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{duration:.0f}', ha='center', va='bottom')
    
    # Scatter plot of duration vs improvement
    ax4.scatter(top_exposure['avg_duration'], top_exposure['avg_improvement'], 
               s=100, c=COLOR_SCHEMES['sun'][:len(top_exposure)], alpha=0.7)
    ax4.set_title('Duration vs Improvement Correlation', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Average Duration (minutes)', fontsize=12)
    ax4.set_ylabel('Average Improvement', fontsize=12)
    
    # Add labels for each point
    for i, exposure_type in enumerate(top_exposure['exposure_type']):
        ax4.annotate(exposure_type, (top_exposure.iloc[i]['avg_duration'], 
                                    top_exposure.iloc[i]['avg_improvement']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_medication_effectiveness_chart(lesion_id, days=30, save_path=None):
    """Create charts showing medication effectiveness"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available for chart creation")
        return
    
    effectiveness = get_medication_effectiveness(lesion_id)
    if effectiveness.empty:
        print("❌ No medication effectiveness data available")
        return
    
    # Get recent medication data for time filtering
    med_data = get_medication_history(lesion_id, days)
    if med_data.empty:
        print("❌ No medication data in the specified time period")
        return
    
    # Filter effectiveness to only include medications from the time period
    recent_medications = med_data['medication_name'].unique()
    effectiveness = effectiveness[effectiveness['medication_name'].isin(recent_medications)]
    
    if effectiveness.empty:
        print("❌ No medication effectiveness in the specified time period")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bar chart of effectiveness
    top_meds = effectiveness.head(8)  # Top 8 medications
    bars = ax1.bar(range(len(top_meds)), top_meds['avg_effectiveness'], 
                   color=COLOR_SCHEMES['medication'][:len(top_meds)])
    ax1.set_title('Medication Effectiveness', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Effectiveness (0-10)', fontsize=12)
    ax1.set_xticks(range(len(top_meds)))
    ax1.set_xticklabels(top_meds['medication_name'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, eff in zip(bars, top_meds['avg_effectiveness']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{eff:.1f}', ha='center', va='bottom')
    
    # Bar chart of adherence
    bars = ax2.bar(range(len(top_meds)), top_meds['avg_adherence'] * 100, 
                   color=COLOR_SCHEMES['medication'][len(top_meds):len(top_meds)*2])
    ax2.set_title('Medication Adherence Rate', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Adherence Rate (%)', fontsize=12)
    ax2.set_xticks(range(len(top_meds)))
    ax2.set_xticklabels(top_meds['medication_name'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, adherence in zip(bars, top_meds['avg_adherence'] * 100):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{adherence:.0f}%', ha='center', va='bottom')
    
    # Bar chart of usage count
    bars = ax3.bar(range(len(top_meds)), top_meds['usage_count'], 
                   color=COLOR_SCHEMES['medication'][len(top_meds)*2:len(top_meds)*3])
    ax3.set_title('Medication Usage Frequency', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Times Used', fontsize=12)
    ax3.set_xticks(range(len(top_meds)))
    ax3.set_xticklabels(top_meds['medication_name'], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, top_meds['usage_count']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(int(count)), ha='center', va='bottom')
    
    # Scatter plot of effectiveness vs adherence
    ax4.scatter(top_meds['avg_effectiveness'], top_meds['avg_adherence'] * 100, 
               s=100, c=COLOR_SCHEMES['medication'][:len(top_meds)], alpha=0.7)
    ax4.set_title('Effectiveness vs Adherence Correlation', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Average Effectiveness', fontsize=12)
    ax4.set_ylabel('Adherence Rate (%)', fontsize=12)
    
    # Add labels for each point
    for i, med_name in enumerate(top_meds['medication_name']):
        ax4.annotate(med_name, (top_meds.iloc[i]['avg_effectiveness'], 
                               top_meds.iloc[i]['avg_adherence'] * 100),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_overall_summary_chart(lesion_id, days=30, save_path=None):
    """Create a comprehensive summary chart"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available for chart creation")
        return
    
    summary = get_overall_summary(lesion_id, days)
    if not summary:
        print("❌ No data available for summary chart")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall statistics
    stats_data = []
    labels = []
    
    if 'symptoms' in summary:
        stats_data.extend([
            summary['symptoms']['itch']['mean'],
            summary['symptoms']['pain']['mean'],
            summary['symptoms']['stress']['mean']
        ])
        labels.extend(['Avg Itch', 'Avg Pain', 'Avg Stress'])
    
    if 'food' in summary:
        stats_data.append(summary['food']['avg_reaction'])
        labels.append('Avg Food Reaction')
    
    if 'stress' in summary:
        stats_data.append(summary['stress']['avg_skin_impact'])
        labels.append('Avg Stress Impact')
    
    if 'sun_exposure' in summary:
        stats_data.append(summary['sun_exposure']['avg_improvement'])
        labels.append('Avg Sun Improvement')
    
    if 'medication' in summary:
        stats_data.append(summary['medication']['avg_effectiveness'])
        labels.append('Avg Med Effectiveness')
    
    # Radar chart of key metrics
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats_data += stats_data[:1]  # Close the plot
    angles += angles[:1]
    
    ax1.plot(angles, stats_data, 'o-', linewidth=2, color=COLOR_SCHEMES['symptoms'][0])
    ax1.fill(angles, stats_data, alpha=0.25, color=COLOR_SCHEMES['symptoms'][0])
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 10)
    ax1.set_title('Overall Health Metrics', fontsize=14, fontweight='bold')
    ax1.grid(True)
    
    # Data volume chart
    categories = []
    volumes = []
    colors = []
    
    if 'symptoms' in summary:
        categories.append('Symptom Records')
        volumes.append(summary['symptoms']['itch']['count'])
        colors.append(COLOR_SCHEMES['symptoms'][0])
    
    if 'food' in summary:
        categories.append('Food Entries')
        volumes.append(summary['food']['total_entries'])
        colors.append(COLOR_SCHEMES['food'][0])
    
    if 'stress' in summary:
        categories.append('Stress Events')
        volumes.append(summary['stress']['total_events'])
        colors.append(COLOR_SCHEMES['stress'][0])
    
    if 'sun_exposure' in summary:
        categories.append('Sun Sessions')
        volumes.append(summary['sun_exposure']['total_sessions'])
        colors.append(COLOR_SCHEMES['sun'][0])
    
    if 'medication' in summary:
        categories.append('Medication Entries')
        volumes.append(summary['medication']['total_entries'])
        colors.append(COLOR_SCHEMES['medication'][0])
    
    bars = ax2.bar(categories, volumes, color=colors)
    ax2.set_title('Data Volume by Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Entries', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, volume in zip(bars, volumes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(int(volume)), ha='center', va='bottom')
    
    # Trend indicators
    trend_data = []
    trend_labels = []
    
    if 'symptoms' in summary:
        trend_data.append(summary['symptoms']['itch']['mean'])
        trend_labels.append('Itch Level')
    
    if 'food' in summary:
        trend_data.append(summary['food']['avg_reaction'])
        trend_labels.append('Food Reactions')
    
    if 'stress' in summary:
        trend_data.append(summary['stress']['avg_skin_impact'])
        trend_labels.append('Stress Impact')
    
    if 'sun_exposure' in summary:
        trend_data.append(summary['sun_exposure']['avg_improvement'])
        trend_labels.append('Sun Improvement')
    
    if 'medication' in summary:
        trend_data.append(summary['medication']['avg_effectiveness'])
        trend_labels.append('Med Effectiveness')
    
    bars = ax3.bar(trend_labels, trend_data, color=COLOR_SCHEMES['symptoms'][:len(trend_data)])
    ax3.set_title('Key Performance Indicators', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average Score (0-10)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, trend_data):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # Summary text
    ax4.axis('off')
    summary_text = f"Lesion #{lesion_id} Summary\nLast {days} days\n\n"
    
    if 'symptoms' in summary:
        summary_text += f"📊 Symptom Records: {summary['symptoms']['itch']['count']}\n"
        summary_text += f"   Avg Itch: {summary['symptoms']['itch']['mean']:.1f}/10\n"
        summary_text += f"   Avg Pain: {summary['symptoms']['pain']['mean']:.1f}/10\n"
        summary_text += f"   Avg Stress: {summary['symptoms']['stress']['mean']:.1f}/10\n\n"
    
    if 'food' in summary:
        summary_text += f"🍽️ Food Entries: {summary['food']['total_entries']}\n"
        summary_text += f"   Reactions: {summary['food']['reactions_count']}\n"
        summary_text += f"   Avg Reaction: {summary['food']['avg_reaction']:.1f}/10\n\n"
    
    if 'stress' in summary:
        summary_text += f"😰 Stress Events: {summary['stress']['total_events']}\n"
        summary_text += f"   Avg Impact: {summary['stress']['avg_skin_impact']:.1f}/10\n\n"
    
    if 'sun_exposure' in summary:
        summary_text += f"☀️ Sun Sessions: {summary['sun_exposure']['total_sessions']}\n"
        summary_text += f"   Avg Improvement: {summary['sun_exposure']['avg_improvement']:.1f}/10\n\n"
    
    if 'medication' in summary:
        summary_text += f"💊 Medication Entries: {summary['medication']['total_entries']}\n"
        summary_text += f"   Avg Effectiveness: {summary['medication']['avg_effectiveness']:.1f}/10\n"
        summary_text += f"   Adherence Rate: {summary['medication']['avg_adherence']*100:.0f}%\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

# ---------------------------
# Simple CLI Interface
# ---------------------------
def print_header():
    """Print application header"""
    print("=" * 60)
    print("🧴 SkinTrack+ - Chronic Skin Condition Tracker")
    print("=" * 60)
    print("Standalone Python Version")
    print("For full web interface: streamlit run skintrack_app.py")
    print("=" * 60)

def print_menu():
    """Print main menu options"""
    print("\n📋 Main Menu:")
    print("1. Create new lesion")
    print("2. List all lesions")
    print("3. Add record to lesion")
    print("4. View lesion history")
    print("5. Add medication schedule")
    print("6. Log food intake")
    print("7. Log stress event")
    print("8. View food reactions")
    print("9. View stress patterns")
    print("10. Log sun exposure")
    print("11. Log medication")
    print("12. View sun exposure patterns")
    print("13. View medication effectiveness")
    print("14. 📊 Data Analysis & Charts")
    print("15. 📸 Image Capture & Analysis")
    print("16. Export data")
    print("17. Initialize database")
    print("18. Exit")
    print("-" * 40)

def get_user_choice():
    """Get user menu choice"""
    while True:
        try:
            choice = input("Enter your choice (1-18): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']:
                return choice
            else:
                print("❌ Please enter a number between 1 and 18.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

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
            # Analyze the image
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
    
    hist = lesion_history(lesion_id)
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

def add_medication():
    """Add medication schedule"""
    print("\n💊 Add Medication Schedule")
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
    
    name = input("Medication name (e.g., triamcinolone 0.1%): ").strip()
    if not name:
        print("❌ Medication name cannot be empty.")
        return
    
    dose = input("Dose/frequency (e.g., thin layer BID): ").strip()
    
    print("\nWhen to take:")
    morning = input("Morning? (y/n): ").strip().lower() == 'y'
    afternoon = input("Afternoon? (y/n): ").strip().lower() == 'y'
    evening = input("Evening? (y/n): ").strip().lower() == 'y'
    
    notes = input("Additional notes (or press Enter for none): ").strip()
    
    upsert_med_schedule(lesion_id, name, dose, morning, afternoon, evening, notes)
    print("✅ Medication schedule added successfully!")

def log_food_intake():
    """Log food intake and track skin reactions"""
    print("\n🍽️ Log Food Intake")
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
    
    # Get food details
    food_item = input("Food item consumed: ").strip()
    if not food_item:
        print("❌ Food item cannot be empty.")
        return
    
    print("\nFood categories:")
    for i, category in enumerate(FOOD_CATEGORIES, 1):
        print(f"{i}. {category}")
    
    while True:
        try:
            choice = int(input(f"\nSelect category (1-{len(FOOD_CATEGORIES)}): "))
            if 1 <= choice <= len(FOOD_CATEGORIES):
                category = FOOD_CATEGORIES[choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(FOOD_CATEGORIES)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    quantity = input("Quantity consumed (e.g., 1 cup, 2 slices): ").strip()
    meal_type = input("Meal type (breakfast/lunch/dinner/snack): ").strip().lower()
    
    # Check for skin reaction
    print("\n📊 Skin Reaction Assessment:")
    print("Did this food cause any skin reaction? (0 = none, 10 = severe)")
    
    while True:
        try:
            skin_reaction = int(input("Skin reaction level (0-10): "))
            if 0 <= skin_reaction <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    reaction_delay_hours = 0
    if skin_reaction > 0:
        while True:
            try:
                reaction_delay_hours = int(input("How many hours after eating did the reaction occur? "))
                if reaction_delay_hours >= 0:
                    break
                else:
                    print("❌ Please enter a non-negative number.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    notes = input("Additional notes (or press Enter for none): ").strip()
    
    # Insert food log
    insert_food_log(
        lesion_id=lesion_id,
        ts=dt.datetime.now().isoformat(timespec="seconds"),
        food_item=food_item,
        category=category,
        quantity=quantity,
        meal_type=meal_type,
        skin_reaction=skin_reaction,
        reaction_delay_hours=reaction_delay_hours,
        notes=notes
    )
    
    print("✅ Food intake logged successfully!")
    if skin_reaction > 0:
        print(f"⚠️ Skin reaction recorded: Level {skin_reaction} after {reaction_delay_hours} hours")

def log_stress_event():
    """Log stress events and their impact on skin condition"""
    print("\n😰 Log Stress Event")
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
    
    # Get stress type
    print("\nStress types:")
    for i, stress_type in enumerate(STRESS_TYPES, 1):
        print(f"{i}. {stress_type}")
    
    while True:
        try:
            choice = int(input(f"\nSelect stress type (1-{len(STRESS_TYPES)}): "))
            if 1 <= choice <= len(STRESS_TYPES):
                stress_type = STRESS_TYPES[choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(STRESS_TYPES)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get stress level
    print("\n📊 Stress Assessment:")
    print("Rate your stress level from 0-10 (0 = none, 10 = extreme)")
    
    while True:
        try:
            stress_level = int(input("Stress level (0-10): "))
            if 0 <= stress_level <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get duration
    while True:
        try:
            duration_hours = float(input("How long did this stress last? (hours): "))
            if duration_hours >= 0:
                break
            else:
                print("❌ Please enter a non-negative number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get symptoms
    print("\nStress symptoms (comma-separated):")
    for i, symptom in enumerate(STRESS_SYMPTOMS, 1):
        print(f"{i}. {symptom}")
    print("Or type your own symptoms separated by commas")
    
    symptoms = input("Symptoms experienced: ").strip()
    
    # Get coping methods
    coping_methods = input("Coping methods used (or press Enter for none): ").strip()
    
    # Get skin impact
    print("\n📊 Skin Impact Assessment:")
    print("How did this stress affect your skin condition? (0 = no effect, 10 = severe worsening)")
    
    while True:
        try:
            skin_impact = int(input("Skin impact level (0-10): "))
            if 0 <= skin_impact <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    notes = input("Additional notes (or press Enter for none): ").strip()
    
    # Insert stress log
    insert_stress_log(
        lesion_id=lesion_id,
        ts=dt.datetime.now().isoformat(timespec="seconds"),
        stress_type=stress_type,
        stress_level=stress_level,
        duration_hours=duration_hours,
        symptoms=symptoms,
        coping_methods=coping_methods,
        skin_impact=skin_impact,
        notes=notes
    )
    
    print("✅ Stress event logged successfully!")
    if skin_impact > 0:
        print(f"⚠️ Skin impact recorded: Level {skin_impact}")

def view_food_reactions():
    """View food reactions and patterns"""
    print("\n🍽️ Food Reactions Analysis")
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
    
    reactions = get_food_reactions(lesion_id)
    if reactions.empty:
        print("No food reactions recorded for this lesion.")
        return
    
    print(f"\n📊 Food Reactions for lesion #{lesion_id}:")
    print("-" * 80)
    print(f"{'Food Item':<20} {'Category':<15} {'Reactions':<10} {'Avg Level':<10} {'Avg Delay':<10}")
    print("-" * 80)
    
    for _, row in reactions.iterrows():
        print(f"{row['food_item']:<20} {row['category']:<15} {row['reaction_count']:<10} "
              f"{row['avg_reaction_level']:<10.1f} {row['avg_delay']:<10.1f}")
    
    # Show recent food history
    print(f"\n📋 Recent Food History (last 30 days):")
    print("-" * 60)
    food_history = get_food_history(lesion_id, days=30)
    
    if not food_history.empty:
        print(f"{'Date':<12} {'Food':<20} {'Reaction':<10} {'Notes':<20}")
        print("-" * 60)
        for _, row in food_history.head(10).iterrows():  # Show last 10 entries
            date_str = str(row['ts']).split('T')[0] if 'T' in str(row['ts']) else str(row['ts'])[:10]
            reaction = f"Level {row['skin_reaction']}" if row['skin_reaction'] > 0 else "None"
            notes = row['notes'][:18] + "..." if len(row['notes']) > 20 else row['notes']
            print(f"{date_str:<12} {row['food_item']:<20} {reaction:<10} {notes:<20}")
    else:
        print("No food entries in the last 30 days.")

def view_stress_patterns():
    """View stress patterns and their impact on skin"""
    print("\n😰 Stress Patterns Analysis")
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
    
    patterns = get_stress_patterns(lesion_id)
    if patterns.empty:
        print("No stress events recorded for this lesion.")
        return
    
    print(f"\n📊 Stress Patterns for lesion #{lesion_id}:")
    print("-" * 90)
    print(f"{'Stress Type':<20} {'Occurrences':<12} {'Avg Level':<10} {'Skin Impact':<12} {'Avg Duration':<12}")
    print("-" * 90)
    
    for _, row in patterns.iterrows():
        print(f"{row['stress_type']:<20} {row['occurrence_count']:<12} "
              f"{row['avg_stress_level']:<10.1f} {row['avg_skin_impact']:<12.1f} "
              f"{row['avg_duration']:<12.1f}")
    
    # Show recent stress history
    print(f"\n📋 Recent Stress History (last 30 days):")
    print("-" * 70)
    stress_history = get_stress_history(lesion_id, days=30)
    
    if not stress_history.empty:
        print(f"{'Date':<12} {'Type':<20} {'Level':<6} {'Impact':<8} {'Notes':<20}")
        print("-" * 70)
        for _, row in stress_history.head(10).iterrows():  # Show last 10 entries
            date_str = str(row['ts']).split('T')[0] if 'T' in str(row['ts']) else str(row['ts'])[:10]
            notes = row['notes'][:18] + "..." if len(row['notes']) > 20 else row['notes']
            print(f"{date_str:<12} {row['stress_type']:<20} {row['stress_level']:<6} "
                  f"{row['skin_impact']:<8} {notes:<20}")
    else:
        print("No stress events in the last 30 days.")

def log_sun_exposure():
    """Log sun exposure events and their impact on skin condition"""
    print("\n☀️ Log Sun Exposure")
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
    
    # Get exposure type
    print("\nSun exposure types:")
    for i, exposure_type in enumerate(SUN_EXPOSURE_TYPES, 1):
        print(f"{i}. {exposure_type}")
    
    while True:
        try:
            choice = int(input(f"\nSelect exposure type (1-{len(SUN_EXPOSURE_TYPES)}): "))
            if 1 <= choice <= len(SUN_EXPOSURE_TYPES):
                exposure_type = SUN_EXPOSURE_TYPES[choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(SUN_EXPOSURE_TYPES)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get duration
    while True:
        try:
            duration_minutes = int(input("How long was the exposure? (minutes): "))
            if duration_minutes >= 0:
                break
            else:
                print("❌ Please enter a non-negative number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get time of day
    print("\nTime of day:")
    print("1. Morning (before 12 PM)")
    print("2. Afternoon (after 12 PM)")
    print("3. Evening (after 6 PM)")
    
    while True:
        try:
            time_of_day_choice = int(input("\nSelect time of day (1-3): "))
            if 1 <= time_of_day_choice <= 3:
                time_of_day = ["Morning", "Afternoon", "Evening"][time_of_day_choice - 1]
                break
            else:
                print("❌ Please enter a number between 1 and 3.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get UV index
    while True:
        try:
            uv_index = int(input("What was the UV index during exposure? (0-11): "))
            if 0 <= uv_index <= 11:
                break
            else:
                print("❌ Please enter a number between 0 and 11.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get protection methods
    print("\nSun protection methods used (comma-separated, or press Enter for none):")
    protection_methods = input("e.g., sunscreen, hat, sunglasses, shade: ").strip()
    
    # Get skin improvement
    print("\n📊 Skin Improvement Assessment:")
    print("How did this sun exposure affect your skin condition? (0 = no effect, 10 = severe improvement)")
    
    while True:
        try:
            skin_improvement = int(input("Skin improvement level (0-10): "))
            if 0 <= skin_improvement <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get side effects
    print("\nSide effects (if any, comma-separated, or press Enter for none):")
    side_effects = input("e.g., sunburn, dry skin, tanning: ").strip()
    
    notes = input("Additional notes (or press Enter for none): ").strip()
    
    # Insert sun exposure log
    insert_sun_exposure_log(
        lesion_id=lesion_id,
        ts=dt.datetime.now().isoformat(timespec="seconds"),
        exposure_type=exposure_type,
        duration_minutes=duration_minutes,
        time_of_day=time_of_day,
        uv_index=uv_index,
        protection_methods=protection_methods,
        skin_improvement=skin_improvement,
        side_effects=side_effects,
        notes=notes
    )
    
    print("✅ Sun exposure logged successfully!")
    if skin_improvement > 0:
        print(f"⚠️ Skin improvement recorded: Level {skin_improvement}")

def view_sun_exposure_patterns():
    """View sun exposure patterns and their effectiveness"""
    print("\n☀️ Sun Exposure Patterns Analysis")
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
    
    patterns = get_sun_exposure_patterns(lesion_id)
    if patterns.empty:
        print("No sun exposure events recorded for this lesion.")
        return
    
    print(f"\n📊 Sun Exposure Patterns for lesion #{lesion_id}:")
    print("-" * 90)
    print(f"{'Exposure Type':<20} {'Occurrences':<12} {'Avg Duration':<10} {'Avg Improvement':<12} {'Avg UV Index':<12}")
    print("-" * 90)
    
    for _, row in patterns.iterrows():
        print(f"{row['exposure_type']:<20} {row['exposure_count']:<12} "
              f"{row['avg_duration']:<10.1f} {row['avg_improvement']:<12.1f} "
              f"{row['avg_uv_index']:<12.1f}")
    
    # Show recent sun exposure history
    print(f"\n📋 Recent Sun Exposure History (last 30 days):")
    print("-" * 70)
    sun_exposure_history = get_sun_exposure_history(lesion_id, days=30)
    
    if not sun_exposure_history.empty:
        print(f"{'Date':<12} {'Type':<20} {'Duration':<10} {'Time':<8} {'UV':<6} {'Improvement':<12} {'Notes':<20}")
        print("-" * 70)
        for _, row in sun_exposure_history.head(10).iterrows():  # Show last 10 entries
            date_str = str(row['ts']).split('T')[0] if 'T' in str(row['ts']) else str(row['ts'])[:10]
            notes = row['notes'][:18] + "..." if len(row['notes']) > 20 else row['notes']
            print(f"{date_str:<12} {row['exposure_type']:<20} {row['duration_minutes']:<10} "
                  f"{row['time_of_day']:<8} {row['uv_index']:<6} {row['skin_improvement']:<12} {notes:<20}")
    else:
        print("No sun exposure events in the last 30 days.")

def log_medication():
    """Log medication taken and its effectiveness"""
    print("\n💊 Log Medication")
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
    
    name = input("Medication name (e.g., triamcinolone 0.1%): ").strip()
    if not name:
        print("❌ Medication name cannot be empty.")
        return
    
    print("\nMedication types:")
    for i, med_type in enumerate(MEDICATION_TYPES, 1):
        print(f"{i}. {med_type}")
    
    while True:
        try:
            med_type_choice = int(input(f"\nSelect medication type (1-{len(MEDICATION_TYPES)}): "))
            if 1 <= med_type_choice <= len(MEDICATION_TYPES):
                medication_type = MEDICATION_TYPES[med_type_choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(MEDICATION_TYPES)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    dose = input("Dose/frequency (e.g., thin layer BID): ").strip()
    
    print("\nWhen to take:")
    morning = input("Morning? (y/n): ").strip().lower() == 'y'
    afternoon = input("Afternoon? (y/n): ").strip().lower() == 'y'
    evening = input("Evening? (y/n): ").strip().lower() == 'y'
    
    notes = input("Additional notes (or press Enter for none): ").strip()
    
    # Get effectiveness
    print("\n📊 Medication Effectiveness Assessment:")
    print("Rate the effectiveness of this medication (0 = ineffective, 10 = highly effective)")
    
    while True:
        try:
            effectiveness = int(input("Effectiveness (0-10): "))
            if 0 <= effectiveness <= 10:
                break
            else:
                print("❌ Please enter a number between 0 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get adherence
    adherence = input("Was this medication taken as prescribed? (y/n): ").strip().lower() == 'y'
    
    # Get side effects
    print("\nSide effects (if any, comma-separated, or press Enter for none):")
    side_effects = input("e.g., sunburn, dry skin, tanning: ").strip()
    
    # Insert medication log
    insert_medication_log(
        lesion_id=lesion_id,
        ts=dt.datetime.now().isoformat(timespec="seconds"),
        medication_name=name,
        medication_type=medication_type,
        dose=dose,
        frequency=f"{dose} {MEDICATION_FREQUENCIES[0]}", # Default to once daily for simplicity
        taken_as_prescribed=int(bool(adherence)),
        effectiveness=effectiveness,
        side_effects=side_effects,
        notes=notes
    )
    
    print("✅ Medication logged successfully!")
    if effectiveness > 0:
        print(f"⚠️ Medication effectiveness recorded: Level {effectiveness}")

def view_medication_effectiveness():
    """View medication effectiveness analysis"""
    print("\n📊 Medication Effectiveness Analysis")
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
    
    effectiveness_data = get_medication_effectiveness(lesion_id)
    if effectiveness_data.empty:
        print("No medication effectiveness data found for this lesion.")
        return
    
    print(f"\n📊 Medication Effectiveness for lesion #{lesion_id}:")
    print("-" * 80)
    print(f"{'Medication Name':<20} {'Type':<15} {'Usage Count':<12} {'Avg Effectiveness':<15} {'Avg Adherence':<15} {'Side Effects':<15}")
    print("-" * 80)
    
    for _, row in effectiveness_data.iterrows():
        print(f"{row['medication_name']:<20} {row['medication_type']:<15} {row['usage_count']:<12} "
              f"{row['avg_effectiveness']:<15.1f} {row['avg_adherence']:<15.1f} {row['side_effect_count']:<15}")
    
    # Show recent medication history
    print(f"\n📋 Recent Medication History (last 30 days):")
    print("-" * 60)
    medication_history = get_medication_history(lesion_id, days=30)
    
    if not medication_history.empty:
        print(f"{'Date':<12} {'Name':<20} {'Type':<15} {'Dose':<10} {'Freq':<10} {'Effectiveness':<15} {'Adherence':<15} {'Notes':<20}")
        print("-" * 60)
        for _, row in medication_history.head(10).iterrows():  # Show last 10 entries
            date_str = str(row['ts']).split('T')[0] if 'T' in str(row['ts']) else str(row['ts'])[:10]
            notes = row['notes'][:18] + "..." if len(row['notes']) > 20 else row['notes']
            print(f"{date_str:<12} {row['medication_name']:<20} {row['medication_type']:<15} {row['dose']:<10} "
                  f"{row['frequency']:<10} {row['effectiveness']:<15} {row['taken_as_prescribed']:<15} {notes:<20}")
    else:
        print("No medication entries in the last 30 days.")

def export_data():
    """Export data to CSV"""
    print("\n📤 Export Data")
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
            lesion_id = int(input("\nSelect lesion ID to export: "))
            if lesion_id in lesions['id'].values:
                break
            else:
                print("❌ Invalid lesion ID.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get lesion info
    lesion_info = lesions[lesions['id'] == lesion_id].iloc[0]
    
    print("\nExport options:")
    print("1. Lesion records only")
    print("2. Lesion records + food data")
    print("3. Lesion records + stress data")
    print("4. Lesion records + sun exposure data")
    print("5. Lesion records + medication data")
    print("6. All data (records + food + stress + sun + medication)")
    
    while True:
        try:
            export_choice = input("\nSelect export type (1-6): ").strip()
            if export_choice in ['1', '2', '3', '4', '5', '6']:
                break
            else:
                print("❌ Please enter a number between 1 and 6.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Create export filename
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"skintrack_lesion{lesion_id}_{timestamp}"
    
    # Export lesion records
    hist = lesion_history(lesion_id)
    if not hist.empty:
        records_filename = f"{base_filename}_records.csv"
        hist.to_csv(records_filename, index=False)
        print(f"✅ Records exported to: {records_filename}")
        print(f"   Records: {len(hist)}")
    else:
        print("⚠️ No lesion records found.")
    
    # Export food data if requested
    if export_choice in ['2', '4', '6']:
        food_hist = get_food_history(lesion_id, days=365)  # Last year
        if not food_hist.empty:
            food_filename = f"{base_filename}_food.csv"
            food_hist.to_csv(food_filename, index=False)
            print(f"✅ Food data exported to: {food_filename}")
            print(f"   Food entries: {len(food_hist)}")
        else:
            print("⚠️ No food data found.")
    
    # Export stress data if requested
    if export_choice in ['3', '6']:
        stress_hist = get_stress_history(lesion_id, days=365)  # Last year
        if not stress_hist.empty:
            stress_filename = f"{base_filename}_stress.csv"
            stress_hist.to_csv(stress_filename, index=False)
            print(f"✅ Stress data exported to: {stress_filename}")
            print(f"   Stress events: {len(stress_hist)}")
        else:
            print("⚠️ No stress data found.")
    
    # Export sun exposure data if requested
    if export_choice in ['4', '6']:
        sun_exposure_hist = get_sun_exposure_history(lesion_id, days=365)  # Last year
        if not sun_exposure_hist.empty:
            sun_exposure_filename = f"{base_filename}_sun_exposure.csv"
            sun_exposure_hist.to_csv(sun_exposure_filename, index=False)
            print(f"✅ Sun exposure data exported to: {sun_exposure_filename}")
            print(f"   Sun exposure events: {len(sun_exposure_hist)}")
        else:
            print("⚠️ No sun exposure data found.")
    
    # Export medication data if requested
    if export_choice in ['5', '6']:
        medication_hist = get_medication_history(lesion_id, days=365)  # Last year
        if not medication_hist.empty:
            medication_filename = f"{base_filename}_medication.csv"
            medication_hist.to_csv(medication_filename, index=False)
            print(f"✅ Medication data exported to: {medication_filename}")
            print(f"   Medication entries: {len(medication_hist)}")
        else:
            print("⚠️ No medication data found.")
    
    print(f"\n📋 Lesion: {lesion_info['label']} [{lesion_info['condition']}]")

# Image capture and processing functions
def capture_image_with_camera():
    """Capture an image using the device camera"""
    if not CV2_AVAILABLE:
        print("❌ OpenCV not available. Cannot capture images with camera.")
        return None
    
    try:
        # Initialize camera
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
            
            # Display the frame
            cv2.imshow('SkinTrack+ Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture the image
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
    
    # Get image path from user
    image_path = input("Enter the full path to your image file: ").strip()
    
    if not image_path:
        print("❌ No image path provided.")
        return None
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    # Check if it's a valid image file
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext not in valid_extensions:
        print(f"❌ Invalid image format. Supported formats: {', '.join(valid_extensions)}")
        return None
    
    try:
        # Copy image to our images directory
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}{file_ext}"
        new_path = IMAGES_DIR / filename
        
        # Copy the file
        import shutil
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
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return {}
        
        # Convert BGR to RGB for analysis
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        metrics = {}
        
        # Basic image properties
        height, width = img.shape[:2]
        metrics['image_size'] = f"{width}x{height}"
        
        # Area measurement (simplified - assumes lesion takes up significant portion)
        # This is a basic implementation - more sophisticated segmentation would be needed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assumed to be the lesion)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            # Convert to cm² (assuming 1 pixel ≈ 0.01 cm² for typical smartphone camera)
            area_cm2 = area * 0.0001  # Rough approximation
            metrics['area_cm2'] = round(area_cm2, 2)
        else:
            metrics['area_cm2'] = None
        
        # Redness analysis
        if SKIMAGE_AVAILABLE:
            try:
                # Convert to LAB color space for better color analysis
                img_lab = rgb2lab(img_rgb)
                
                # Calculate redness (using a* channel in LAB space)
                a_channel = img_lab[:, :, 1]
                redness_score = np.mean(a_channel)
                metrics['redness'] = round(redness_score, 2)
                
                # Color analysis
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
        
        # Border irregularity (simplified)
        if contours:
            perimeter = cv2.arcLength(largest_contour, True)
            if area > 0:
                # Circularity = 4π * area / perimeter²
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                irregularity = 1 - circularity  # Higher = more irregular
                metrics['border_irreg'] = round(irregularity, 3)
            else:
                metrics['border_irreg'] = None
        else:
            metrics['border_irreg'] = None
        
        # Asymmetry analysis (simplified)
        if contours:
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            # Asymmetry score based on how far from 1.0 (perfect symmetry)
            asymmetry = abs(1 - aspect_ratio)
            metrics['asymmetry'] = round(asymmetry, 3)
        else:
            metrics['asymmetry'] = None
        
        # Texture analysis (simplified)
        if len(img.shape) == 3:
            # Calculate standard deviation of gray values as texture measure
            gray_std = np.std(gray)
            metrics['texture_variance'] = round(gray_std, 2)
        else:
            metrics['texture_variance'] = None
        
        print("✅ Image analysis completed!")
        return metrics
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return {}

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
    
    # Get records with images for this lesion
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
    
    if choice == 1:  # Take photo with camera
        print("\n📸 Taking photo with camera...")
        image_path = capture_image_with_camera()
    elif choice == 2:  # Upload existing image
        image_path = upload_existing_image()
    elif choice == 3:  # View captured images
        list_captured_images()
        return
    elif choice == 4:  # Analyze image metrics
        # For analysis, we need an existing image
        print("\n📁 Select image for analysis:")
        image_path = upload_existing_image()
    
    if not image_path:
        print("❌ No image available for analysis.")
        return
    
    # Analyze the image
    print("\n🔍 Analyzing image...")
    metrics = analyze_image_metrics(image_path)
    
    if metrics:
        print("\n📊 Image Analysis Results:")
        print("-" * 40)
        for key, value in metrics.items():
            if value is not None:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Ask if user wants to add this to a lesion record
    add_to_record = input("\nAdd this image to a lesion record? (y/n): ").strip().lower()
    if add_to_record == 'y':
        add_record_with_image(image_path, metrics)

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

def data_analysis_menu():
    """Data analysis and visualization menu"""
    print("\n📊 Data Analysis & Charts")
    print("-" * 30)
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib/seaborn not available for chart creation")
        print("Please install matplotlib and seaborn for data visualization:")
        print("pip install matplotlib seaborn")
        return
    
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
    
    print("\nAnalysis options:")
    print("1. Symptom trends (line chart)")
    print("2. Food reactions (bar charts)")
    print("3. Stress patterns (pie & bar charts)")
    print("4. Sun exposure effectiveness (pie & bar charts)")
    print("5. Medication effectiveness (bar charts)")
    print("6. Overall summary (comprehensive dashboard)")
    print("7. Statistical summary (numerical data)")
    
    while True:
        try:
            analysis_choice = input("\nSelect analysis type (1-7): ").strip()
            if analysis_choice in ['1', '2', '3', '4', '5', '6', '7']:
                break
            else:
                print("❌ Please enter a number between 1 and 7.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    print("\nTime period:")
    for i, period in enumerate(TIME_PERIODS, 1):
        print(f"{i}. {period}")
    
    while True:
        try:
            period_choice = int(input(f"\nSelect time period (1-{len(TIME_PERIODS)}): "))
            if 1 <= period_choice <= len(TIME_PERIODS):
                period = TIME_PERIODS[period_choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(TIME_PERIODS)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Convert time period to days
    days_map = {
        "last 7 days": 7,
        "last 30 days": 30,
        "last 90 days": 90,
        "last 6 months": 180,
        "last year": 365,
        "all time": 3650  # 10 years for "all time"
    }
    days = days_map[period]
    
    # Ask if user wants to save charts
    save_charts = input("\nSave charts to files? (y/n): ").strip().lower() == 'y'
    
    if save_charts:
        # Create charts directory
        charts_dir = DATA_DIR / "charts"
        charts_dir.mkdir(exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"lesion{lesion_id}_{timestamp}"
    
    print(f"\n📊 Generating analysis for lesion #{lesion_id} ({period})...")
    
    if analysis_choice == '1':
        # Symptom trends
        print("📈 Creating symptom trends chart...")
        if save_charts:
            save_path = charts_dir / f"{base_filename}_symptom_trends.png"
            create_symptom_trend_chart(lesion_id, days, save_path)
        else:
            create_symptom_trend_chart(lesion_id, days)
    
    elif analysis_choice == '2':
        # Food reactions
        print("🍽️ Creating food reaction charts...")
        if save_charts:
            save_path = charts_dir / f"{base_filename}_food_reactions.png"
            create_food_reaction_chart(lesion_id, days, save_path)
        else:
            create_food_reaction_chart(lesion_id, days)
    
    elif analysis_choice == '3':
        # Stress patterns
        print("😰 Creating stress pattern charts...")
        if save_charts:
            save_path = charts_dir / f"{base_filename}_stress_patterns.png"
            create_stress_pattern_chart(lesion_id, days, save_path)
        else:
            create_stress_pattern_chart(lesion_id, days)
    
    elif analysis_choice == '4':
        # Sun exposure
        print("☀️ Creating sun exposure charts...")
        if save_charts:
            save_path = charts_dir / f"{base_filename}_sun_exposure.png"
            create_sun_exposure_chart(lesion_id, days, save_path)
        else:
            create_sun_exposure_chart(lesion_id, days)
    
    elif analysis_choice == '5':
        # Medication effectiveness
        print("💊 Creating medication effectiveness charts...")
        if save_charts:
            save_path = charts_dir / f"{base_filename}_medication_effectiveness.png"
            create_medication_effectiveness_chart(lesion_id, days, save_path)
        else:
            create_medication_effectiveness_chart(lesion_id, days)
    
    elif analysis_choice == '6':
        # Overall summary
        print("📊 Creating comprehensive summary dashboard...")
        if save_charts:
            save_path = charts_dir / f"{base_filename}_overall_summary.png"
            create_overall_summary_chart(lesion_id, days, save_path)
        else:
            create_overall_summary_chart(lesion_id, days)
    
    elif analysis_choice == '7':
        # Statistical summary
        print("📈 Generating statistical summary...")
        display_statistical_summary(lesion_id, days)
    
    print("✅ Analysis complete!")

def display_statistical_summary(lesion_id, days=30):
    """Display comprehensive statistical summary"""
    print(f"\n📊 Statistical Summary - Lesion #{lesion_id} (Last {days} days)")
    print("=" * 80)
    
    summary = get_overall_summary(lesion_id, days)
    if not summary:
        print("❌ No data available for statistical analysis")
        return
    
    # Symptom Statistics
    if 'symptoms' in summary:
        print("\n📈 SYMPTOM STATISTICS:")
        print("-" * 40)
        symptoms = summary['symptoms']
        print(f"Total Records: {symptoms['itch']['count']}")
        print(f"Average Itch Level: {symptoms['itch']['mean']:.2f}/10 (Min: {symptoms['itch']['min']}, Max: {symptoms['itch']['max']})")
        print(f"Average Pain Level: {symptoms['pain']['mean']:.2f}/10 (Min: {symptoms['pain']['min']}, Max: {symptoms['pain']['max']})")
        print(f"Average Stress Level: {symptoms['stress']['mean']:.2f}/10 (Min: {symptoms['stress']['min']}, Max: {symptoms['stress']['max']})")
        print(f"Average Sleep Hours: {symptoms['sleep']['mean']:.2f} hours (Min: {symptoms['sleep']['min']}, Max: {symptoms['sleep']['max']})")
    
    # Food Statistics
    if 'food' in summary:
        print("\n🍽️ FOOD STATISTICS:")
        print("-" * 40)
        food = summary['food']
        print(f"Total Food Entries: {food['total_entries']}")
        print(f"Food Reactions: {food['reactions_count']} ({food['reactions_count']/food['total_entries']*100:.1f}% of entries)")
        print(f"Average Reaction Level: {food['avg_reaction']:.2f}/10")
        print("Top Food Categories:")
        for category, count in food['top_categories'].items():
            print(f"  • {category}: {count} entries")
    
    # Stress Statistics
    if 'stress' in summary:
        print("\n😰 STRESS STATISTICS:")
        print("-" * 40)
        stress = summary['stress']
        print(f"Total Stress Events: {stress['total_events']}")
        print(f"Average Stress Level: {stress['avg_stress_level']:.2f}/10")
        print(f"Average Skin Impact: {stress['avg_skin_impact']:.2f}/10")
        print("Top Stress Types:")
        for stress_type, count in stress['top_stress_types'].items():
            print(f"  • {stress_type}: {count} events")
    
    # Sun Exposure Statistics
    if 'sun_exposure' in summary:
        print("\n☀️ SUN EXPOSURE STATISTICS:")
        print("-" * 40)
        sun = summary['sun_exposure']
        print(f"Total Sun Sessions: {sun['total_sessions']}")
        print(f"Average Duration: {sun['avg_duration']:.1f} minutes")
        print(f"Average Skin Improvement: {sun['avg_improvement']:.2f}/10")
        print(f"Average UV Index: {sun['avg_uv_index']:.1f}")
        print("Top Exposure Types:")
        for exposure_type, count in sun['top_exposure_types'].items():
            print(f"  • {exposure_type}: {count} sessions")
    
    # Medication Statistics
    if 'medication' in summary:
        print("\n💊 MEDICATION STATISTICS:")
        print("-" * 40)
        med = summary['medication']
        print(f"Total Medication Entries: {med['total_entries']}")
        print(f"Average Effectiveness: {med['avg_effectiveness']:.2f}/10")
        print(f"Adherence Rate: {med['avg_adherence']*100:.1f}%")
        print("Top Medications:")
        for med_name, count in med['top_medications'].items():
            print(f"  • {med_name}: {count} entries")
    
    # Overall Insights
    print("\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    if 'symptoms' in summary:
        itch_avg = summary['symptoms']['itch']['mean']
        if itch_avg > 7:
            print("⚠️  High average itch level - consider reviewing triggers and treatments")
        elif itch_avg < 3:
            print("✅ Low average itch level - good symptom control")
    
    if 'food' in summary and 'stress' in summary:
        food_reaction_rate = summary['food']['reactions_count'] / summary['food']['total_entries']
        if food_reaction_rate > 0.3:
            print("⚠️  High food reaction rate - consider dietary review")
        elif food_reaction_rate < 0.1:
            print("✅ Low food reaction rate - good dietary control")
    
    if 'stress' in summary:
        stress_impact = summary['stress']['avg_skin_impact']
        if stress_impact > 6:
            print("⚠️  High stress impact on skin - consider stress management strategies")
        elif stress_impact < 3:
            print("✅ Low stress impact on skin - good stress management")
    
    if 'sun_exposure' in summary:
        sun_improvement = summary['sun_exposure']['avg_improvement']
        if sun_improvement > 6:
            print("✅ Good sun exposure effectiveness - continue current regimen")
        elif sun_improvement < 3:
            print("⚠️  Low sun exposure effectiveness - consider adjusting duration or timing")
    
    if 'medication' in summary:
        med_effectiveness = summary['medication']['avg_effectiveness']
        med_adherence = summary['medication']['avg_adherence']
        if med_effectiveness > 7 and med_adherence > 0.8:
            print("✅ High medication effectiveness and adherence - excellent treatment compliance")
        elif med_effectiveness < 5:
            print("⚠️  Low medication effectiveness - consider discussing with healthcare provider")
        elif med_adherence < 0.7:
            print("⚠️  Low medication adherence - consider reminder strategies")
    
    print("\n" + "=" * 80)

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
            add_medication()
        elif choice == '6':
            log_food_intake()
        elif choice == '7':
            log_stress_event()
        elif choice == '8':
            view_food_reactions()
        elif choice == '9':
            view_stress_patterns()
        elif choice == '10':
            log_sun_exposure()
        elif choice == '11':
            log_medication()
        elif choice == '12':
            view_sun_exposure_patterns()
        elif choice == '13':
            view_medication_effectiveness()
        elif choice == '14':
            data_analysis_menu()
        elif choice == '15':
            capture_and_analyze_image()
        elif choice == '16':
            export_data()
        elif choice == '17':
            init_db()
        elif choice == '18':
            print("\n👋 Thank you for using SkinTrack+!")
            print("For the full web interface with image analysis, run:")
            print("  streamlit run skintrack_app.py")
            break
        
        input("\nPress Enter to continue...")

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
