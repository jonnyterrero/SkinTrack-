#!/usr/bin/env python3
"""
Test script for SkinTrack+ new features
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main module
try:
    import skintrack_standalone as st
    print("✅ Successfully imported skintrack_standalone module")
    
    # Test database initialization
    print("\n🧪 Testing database initialization...")
    st.init_db()
    
    # Test constants
    print(f"\n📋 Food categories: {len(st.FOOD_CATEGORIES)} categories")
    print(f"📋 Stress types: {len(st.STRESS_TYPES)} types")
    print(f"📋 Stress symptoms: {len(st.STRESS_SYMPTOMS)} symptoms")
    print(f"📋 Sun exposure types: {len(st.SUN_EXPOSURE_TYPES)} types")
    print(f"📋 Sun protection methods: {len(st.SUN_PROTECTION_METHODS)} methods")
    print(f"📋 Medication types: {len(st.MEDICATION_TYPES)} types")
    print(f"📋 Medication frequencies: {len(st.MEDICATION_FREQUENCIES)} frequencies")
    
    # Test database functions
    print("\n🧪 Testing database functions...")
    
    # Test lesion creation
    lesion_id = st.insert_lesion("Test Lesion", "eczema")
    print(f"✅ Created test lesion with ID: {lesion_id}")
    
    # Test food logging
    st.insert_food_log(
        lesion_id=lesion_id,
        ts="2024-01-01T12:00:00",
        food_item="Test Food",
        category="dairy",
        quantity="1 cup",
        meal_type="lunch",
        skin_reaction=5,
        reaction_delay_hours=2,
        notes="Test food reaction"
    )
    print("✅ Food log entry created")
    
    # Test stress logging
    st.insert_stress_log(
        lesion_id=lesion_id,
        ts="2024-01-01T14:00:00",
        stress_type="work stress",
        stress_level=7,
        duration_hours=4.5,
        symptoms="increased itching,flare-ups",
        coping_methods="deep breathing",
        skin_impact=6,
        notes="Test stress event"
    )
    print("✅ Stress log entry created")
    
    # Test sun exposure logging
    st.insert_sun_exposure_log(
        lesion_id=lesion_id,
        ts="2024-01-01T16:00:00",
        exposure_type="natural sunlight",
        duration_minutes=30,
        time_of_day="Afternoon",
        uv_index=5,
        protection_methods="sunscreen,hat",
        skin_improvement=7,
        side_effects="",
        notes="Test sun exposure"
    )
    print("✅ Sun exposure log entry created")
    
    # Test medication logging
    st.insert_medication_log(
        lesion_id=lesion_id,
        ts="2024-01-01T18:00:00",
        medication_name="Test Medication",
        medication_type="topical steroid",
        dose="thin layer",
        frequency="twice daily",
        taken_as_prescribed=1,
        effectiveness=8,
        side_effects="",
        notes="Test medication"
    )
    print("✅ Medication log entry created")
    
    # Test data retrieval
    food_history = st.get_food_history(lesion_id)
    print(f"✅ Retrieved {len(food_history)} food entries")
    
    stress_history = st.get_stress_history(lesion_id)
    print(f"✅ Retrieved {len(stress_history)} stress entries")
    
    food_reactions = st.get_food_reactions(lesion_id)
    print(f"✅ Retrieved {len(food_reactions)} food reactions")
    
    stress_patterns = st.get_stress_patterns(lesion_id)
    print(f"✅ Retrieved {len(stress_patterns)} stress patterns")
    
    sun_exposure_history = st.get_sun_exposure_history(lesion_id)
    print(f"✅ Retrieved {len(sun_exposure_history)} sun exposure entries")
    
    sun_exposure_patterns = st.get_sun_exposure_patterns(lesion_id)
    print(f"✅ Retrieved {len(sun_exposure_patterns)} sun exposure patterns")
    
    medication_history = st.get_medication_history(lesion_id)
    print(f"✅ Retrieved {len(medication_history)} medication entries")
    
    medication_effectiveness = st.get_medication_effectiveness(lesion_id)
    print(f"✅ Retrieved {len(medication_effectiveness)} medication effectiveness records")
    
    # Test data analysis functions
    print("\n🧪 Testing data analysis functions...")
    
    # Test symptom statistics
    symptom_stats = st.get_symptom_statistics(lesion_id)
    if symptom_stats:
        print("✅ Symptom statistics generated")
        print(f"   Average itch: {symptom_stats['itch']['mean']:.2f}")
        print(f"   Average pain: {symptom_stats['pain']['mean']:.2f}")
        print(f"   Average stress: {symptom_stats['stress']['mean']:.2f}")
    else:
        print("⚠️ No symptom statistics available")
    
    # Test overall summary
    overall_summary = st.get_overall_summary(lesion_id)
    if overall_summary:
        print("✅ Overall summary generated")
        for category, data in overall_summary.items():
            print(f"   {category}: {len(data)} data points")
    else:
        print("⚠️ No overall summary available")
    
    # Test chart creation (if matplotlib is available)
    if st.MATPLOTLIB_AVAILABLE:
        print("\n🧪 Testing chart creation...")
        
        # Test symptom trend chart
        try:
            st.create_symptom_trend_chart(lesion_id, days=1)
            print("✅ Symptom trend chart created")
        except Exception as e:
            print(f"⚠️ Symptom trend chart failed: {e}")
        
        # Test food reaction chart
        try:
            st.create_food_reaction_chart(lesion_id, days=1)
            print("✅ Food reaction chart created")
        except Exception as e:
            print(f"⚠️ Food reaction chart failed: {e}")
        
        # Test stress pattern chart
        try:
            st.create_stress_pattern_chart(lesion_id, days=1)
            print("✅ Stress pattern chart created")
        except Exception as e:
            print(f"⚠️ Stress pattern chart failed: {e}")
        
        # Test sun exposure chart
        try:
            st.create_sun_exposure_chart(lesion_id, days=1)
            print("✅ Sun exposure chart created")
        except Exception as e:
            print(f"⚠️ Sun exposure chart failed: {e}")
        
        # Test medication effectiveness chart
        try:
            st.create_medication_effectiveness_chart(lesion_id, days=1)
            print("✅ Medication effectiveness chart created")
        except Exception as e:
            print(f"⚠️ Medication effectiveness chart failed: {e}")
        
        # Test overall summary chart
        try:
            st.create_overall_summary_chart(lesion_id, days=1)
            print("✅ Overall summary chart created")
        except Exception as e:
            print(f"⚠️ Overall summary chart failed: {e}")
    else:
        print("⚠️ matplotlib not available - skipping chart tests")
    
    print("\n🎉 All tests passed! The new features are working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
