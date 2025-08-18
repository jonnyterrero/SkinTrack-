# SkinTrack+ - Food Logging, Stress Tracking, Sun Exposure, and Medication Features

## Overview

This document describes the comprehensive tracking features added to SkinTrack+, which help users identify triggers and patterns related to their chronic skin conditions, including food reactions, stress impacts, sun exposure therapy, and medication effectiveness.

## New Features

### 1. Food Logging System

The food logging system allows users to track their food intake and correlate it with skin reactions.

#### Features:
- **Food Categories**: Pre-defined categories including dairy, gluten, nuts, shellfish, eggs, soy, wheat, citrus, tomatoes, spicy foods, processed foods, sugar, alcohol, caffeine, chocolate, nightshades, and other
- **Reaction Tracking**: Rate skin reactions from 0-10 (0 = none, 10 = severe)
- **Delay Tracking**: Record how many hours after eating the reaction occurred
- **Meal Type**: Categorize by breakfast, lunch, dinner, or snack
- **Quantity Tracking**: Record amount consumed (e.g., "1 cup", "2 slices")

#### Database Schema:
```sql
CREATE TABLE food_log(
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
```

### 2. Enhanced Stress Tracking System

The stress tracking system provides detailed monitoring of stress events and their impact on skin conditions.

#### Features:
- **Stress Types**: Categorized stress types including work stress, personal stress, financial stress, health stress, relationship stress, academic stress, social stress, environmental stress, sleep deprivation, emotional stress, physical stress, and other
- **Stress Levels**: Rate stress from 0-10 (0 = none, 10 = extreme)
- **Duration Tracking**: Record how long the stress lasted in hours
- **Symptom Tracking**: Record stress symptoms like increased itching, flare-ups, redness, swelling, pain, sleep problems, anxiety, depression, fatigue, irritability, concentration issues
- **Coping Methods**: Track what coping strategies were used
- **Skin Impact**: Rate how the stress affected skin condition (0-10)

#### Database Schema:
```sql
CREATE TABLE stress_log(
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
```

### 3. Sun Exposure Tracking System

The sun exposure tracking system monitors UV therapy and natural sunlight exposure, which is a known treatment for many skin conditions like psoriasis and eczema.

#### Features:
- **Exposure Types**: Natural sunlight, UVB therapy, UVA therapy, phototherapy, tanning bed, outdoor activity, beach/sunbathing, walking outside, gardening, sports, and other
- **Duration Tracking**: Record exposure time in minutes
- **Time of Day**: Morning, afternoon, or evening exposure
- **UV Index**: Track UV intensity (0-11 scale)
- **Protection Methods**: Sunscreen, protective clothing, hat, sunglasses, shade, umbrella, long sleeves, pants, no protection
- **Skin Improvement**: Rate improvement from 0-10 (0 = no effect, 10 = severe improvement)
- **Side Effects**: Track any adverse effects like sunburn, dry skin, tanning

#### Database Schema:
```sql
CREATE TABLE sun_exposure_log(
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
```

### 4. Medication Logging System

The medication logging system tracks medication usage, effectiveness, and adherence for comprehensive treatment management.

#### Features:
- **Medication Types**: Topical steroid, topical non-steroid, oral medication, injection, phototherapy, biologic, immunosuppressant, antibiotic, antihistamine, vitamin supplement, herbal supplement, over-the-counter, prescription, and other
- **Dosage Tracking**: Record dose and frequency information
- **Effectiveness Assessment**: Rate effectiveness from 0-10 (0 = ineffective, 10 = highly effective)
- **Adherence Tracking**: Monitor if medication was taken as prescribed
- **Side Effects**: Record any medication side effects
- **Frequency Options**: Once daily, twice daily, three times daily, as needed, weekly, bi-weekly, monthly, before meals, after meals, at bedtime

#### Database Schema:
```sql
CREATE TABLE medication_log(
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
```

## Usage Instructions

### Running the Application

```bash
python skintrack_standalone.py
```

### Menu Options

The main menu now includes these comprehensive options:

1. **Create new lesion** - Create a new skin condition to track
2. **List all lesions** - View all tracked skin conditions
3. **Add record to lesion** - Add symptom and treatment records
4. **View lesion history** - View historical data for a lesion
5. **Add medication schedule** - Set up medication reminders
6. **Log food intake** - Track food consumption and reactions
7. **Log stress event** - Record stress events and skin impact
8. **View food reactions** - Analyze food-related skin reactions
9. **View stress patterns** - Analyze stress patterns and skin impact
10. **Log sun exposure** - **NEW**: Track UV therapy and sunlight exposure
11. **Log medication** - **NEW**: Record medication usage and effectiveness
12. **View sun exposure patterns** - **NEW**: Analyze sun exposure effectiveness
13. **View medication effectiveness** - **NEW**: Analyze medication performance
14. **Export data** - **ENHANCED**: Export all data types
15. **Initialize database** - Set up the database
16. **Exit** - Close the application

### Food Logging Workflow

1. Select "6. Log food intake" from the main menu
2. Choose the lesion/condition to track
3. Enter the food item consumed
4. Select the food category from the predefined list
5. Enter quantity and meal type
6. Rate any skin reaction (0-10)
7. If there was a reaction, record the delay time
8. Add any additional notes

### Stress Tracking Workflow

1. Select "7. Log stress event" from the main menu
2. Choose the lesion/condition to track
3. Select the type of stress from the predefined list
4. Rate the stress level (0-10)
5. Enter how long the stress lasted
6. List any symptoms experienced
7. Record coping methods used
8. Rate the impact on skin condition (0-10)
9. Add any additional notes

### Sun Exposure Tracking Workflow

1. Select "10. Log sun exposure" from the main menu
2. Choose the lesion/condition to track
3. Select the exposure type (natural sunlight, UVB therapy, etc.)
4. Enter duration in minutes
5. Select time of day (morning, afternoon, evening)
6. Enter UV index (0-11)
7. List protection methods used
8. Rate skin improvement (0-10)
9. Record any side effects
10. Add any additional notes

### Medication Logging Workflow

1. Select "11. Log medication" from the main menu
2. Choose the lesion/condition to track
3. Enter medication name
4. Select medication type from predefined list
5. Enter dose and frequency information
6. Rate effectiveness (0-10)
7. Indicate if taken as prescribed
8. Record any side effects
9. Add any additional notes

### Data Analysis

#### Food Reactions Analysis
- View foods that caused skin reactions
- See average reaction levels and delay times
- Track reaction frequency by food item
- Review recent food history

#### Stress Patterns Analysis
- Identify most common stress types
- Track stress frequency and duration
- Correlate stress with skin impact
- Review recent stress history

#### Sun Exposure Patterns Analysis
- Identify most effective exposure types
- Track optimal duration and timing
- Monitor UV index effectiveness
- Review sun protection methods

#### Medication Effectiveness Analysis
- Track medication performance
- Monitor adherence rates
- Identify most effective treatments
- Track side effect patterns

### Data Export

The export feature now supports multiple export types:

1. **Lesion records only** - Traditional symptom and treatment data
2. **Lesion records + food data** - Include food logging data
3. **Lesion records + stress data** - Include stress tracking data
4. **Lesion records + sun exposure data** - Include sun exposure data
5. **Lesion records + medication data** - Include medication data
6. **All data** - Complete dataset including all tracking types

## Benefits

### For Patients:
- **Identify Triggers**: Discover which foods, stress types, or environmental factors worsen skin conditions
- **Track Treatment Effectiveness**: Monitor sun exposure and medication effectiveness
- **Optimize Therapy**: Find optimal timing and duration for UV therapy
- **Improve Adherence**: Track medication compliance and effectiveness
- **Personalized Care**: Understand individual response to different treatments

### For Healthcare Providers:
- **Comprehensive Data**: Get detailed patient history including all lifestyle factors
- **Treatment Optimization**: Make evidence-based decisions about UV therapy and medication
- **Patient Education**: Help patients understand their condition and treatment options
- **Adherence Monitoring**: Track medication compliance and effectiveness
- **Therapy Planning**: Optimize phototherapy and medication regimens

## Technical Implementation

### Database Functions

#### Food Logging:
- `insert_food_log()` - Add new food entry
- `get_food_history()` - Retrieve food history
- `get_food_reactions()` - Analyze food reactions

#### Stress Tracking:
- `insert_stress_log()` - Add new stress entry
- `get_stress_history()` - Retrieve stress history
- `get_stress_patterns()` - Analyze stress patterns

#### Sun Exposure Tracking:
- `insert_sun_exposure_log()` - Add new sun exposure entry
- `get_sun_exposure_history()` - Retrieve sun exposure history
- `get_sun_exposure_patterns()` - Analyze sun exposure patterns

#### Medication Logging:
- `insert_medication_log()` - Add new medication entry
- `get_medication_history()` - Retrieve medication history
- `get_medication_effectiveness()` - Analyze medication effectiveness

### Data Validation

- All numeric inputs are validated for appropriate ranges
- Required fields are checked for completeness
- Foreign key relationships ensure data integrity
- Timestamps are automatically generated

### Error Handling

- Graceful handling of missing data
- User-friendly error messages
- Input validation with helpful feedback
- Database connection error handling

## Future Enhancements

Potential future improvements could include:

1. **Machine Learning Analysis**: Automatically identify trigger patterns and optimal treatment regimens
2. **Mobile App**: Smartphone interface for easier logging
3. **Integration**: Connect with health apps, weather services (for UV index), and medical devices
4. **Notifications**: Reminders for medication, UV therapy sessions, and logging
5. **Sharing**: Secure sharing with healthcare providers
6. **Analytics**: Advanced statistical analysis and reporting
7. **Weather Integration**: Automatic UV index tracking
8. **Treatment Recommendations**: AI-powered treatment suggestions based on patterns

## Support

For technical support or feature requests, please refer to the main SkinTrack+ documentation or contact the development team.

---

*This comprehensive feature set transforms SkinTrack+ into a complete lifestyle and treatment monitoring system, providing valuable insights for both patients and healthcare providers in managing chronic skin conditions through evidence-based tracking of triggers, treatments, and outcomes.*
