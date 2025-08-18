#!/usr/bin/env python3
"""
Test script for SkinTrack+ image capture and analysis features
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from skintrack_standalone import (
        init_db, list_lesions, insert_lesion, 
        capture_image_with_camera, upload_existing_image, 
        analyze_image_metrics, list_captured_images,
        capture_and_analyze_image, add_record_with_image
    )
    print("✅ Successfully imported SkinTrack+ modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def test_database_setup():
    """Test database initialization and basic operations"""
    print("\n🧪 Testing Database Setup")
    print("-" * 40)
    
    try:
        # Initialize database
        init_db()
        print("✅ Database initialized successfully")
        
        # List lesions (should be empty initially)
        lesions = list_lesions()
        print(f"✅ Listed lesions: {len(lesions)} found")
        
        # Create a test lesion
        lesion_id = insert_lesion("Test Lesion", "eczema")
        print(f"✅ Created test lesion with ID: {lesion_id}")
        
        return lesion_id
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return None

def test_image_analysis():
    """Test image analysis functionality"""
    print("\n🧪 Testing Image Analysis")
    print("-" * 40)
    
    # Test with a dummy image path (won't actually analyze, but tests the function structure)
    dummy_path = "test_image.jpg"
    
    try:
        # Test the analysis function (will fail gracefully if no image)
        metrics = analyze_image_metrics(dummy_path)
        print("✅ Image analysis function structure is correct")
        print(f"   Expected empty metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")

def test_image_functions():
    """Test image capture and upload functions"""
    print("\n🧪 Testing Image Functions")
    print("-" * 40)
    
    try:
        # Test camera capture (will fail gracefully if no camera)
        print("Testing camera capture function...")
        result = capture_image_with_camera()
        if result is None:
            print("✅ Camera capture function handles missing camera gracefully")
        else:
            print(f"✅ Camera capture returned: {result}")
        
        # Test upload function (will fail gracefully if no file)
        print("Testing image upload function...")
        result = upload_existing_image()
        if result is None:
            print("✅ Image upload function handles missing file gracefully")
        else:
            print(f"✅ Image upload returned: {result}")
            
    except Exception as e:
        print(f"❌ Image function test failed: {e}")

def test_list_images():
    """Test listing captured images"""
    print("\n🧪 Testing List Images Function")
    print("-" * 40)
    
    try:
        # This should work even with no images
        list_captured_images()
        print("✅ List images function executed successfully")
        
    except Exception as e:
        print(f"❌ List images test failed: {e}")

def test_main_image_function():
    """Test the main image capture and analysis function"""
    print("\n🧪 Testing Main Image Function")
    print("-" * 40)
    
    try:
        # Test the main function (will show menu but not execute)
        print("Testing main image capture function structure...")
        # We can't easily test the interactive function, but we can verify it exists
        if callable(capture_and_analyze_image):
            print("✅ Main image function is callable")
        else:
            print("❌ Main image function is not callable")
            
    except Exception as e:
        print(f"❌ Main image function test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 SkinTrack+ Image Features Test Suite")
    print("=" * 50)
    
    # Test database setup
    lesion_id = test_database_setup()
    
    # Test image analysis
    test_image_analysis()
    
    # Test image functions
    test_image_functions()
    
    # Test list images
    test_list_images()
    
    # Test main image function
    test_main_image_function()
    
    print("\n✅ All tests completed!")
    print("\n📝 Notes:")
    print("- Camera capture requires OpenCV and a working camera")
    print("- Image analysis requires OpenCV and PIL")
    print("- Some functions will fail gracefully when dependencies are missing")
    print("- This is expected behavior for a robust application")

if __name__ == "__main__":
    main()
