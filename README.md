# SkinTrack+

**A comprehensive Progressive Web App (PWA) for tracking and analyzing chronic skin conditions.**

Code for a skintracker project. Aimed at helping people with chronic skin conditions and illnesses.

*Automatically synced with your [v0.app](https://v0.app) deployments*

[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com/jonnyterreros-projects/v0-web-app-to-mobile)
[![Built with v0](https://img.shields.io/badge/Built%20with-v0.app-black?style=for-the-badge)](https://v0.app/chat/projects/9biUawphA3D)

## 📱 Overview

SkinTrack+ is a modern, feature-rich web application designed to help individuals track, monitor, and analyze their chronic skin conditions over time. Built with Next.js and React, this Progressive Web App (PWA) provides a seamless, mobile-friendly experience that can be installed on any device, working both online and offline.

The application empowers users to maintain detailed records of their skin health, track symptoms, capture and analyze images, visualize progress trends, and export data for healthcare providers—all in one intuitive platform.

## ✨ Key Features

### 📸 Image Capture & Analysis

- **Camera Integration**: Capture photos directly from your device's camera
- **Image Upload**: Upload existing images from your device's gallery
- **Automated Image Analysis**: Advanced analysis tools that measure:
  - **Area Measurement**: Calculate the size of skin lesions in cm²
  - **Redness Analysis**: Measure inflammation levels using LAB color space analysis
  - **Border Irregularity**: Analyze the irregularity of lesion borders
  - **Asymmetry Detection**: Measure symmetry/asymmetry of lesions
  - **Texture Analysis**: Evaluate texture variance in the affected skin
  - **Confidence Scoring**: AI-powered confidence ratings for analysis results
- **Image Gallery**: Organize and view all captured images in one place
- **Image Management**: Easily delete or manage stored images

### 📝 Symptom Tracking

Comprehensive symptom logging system that allows you to track:

- **Condition Types**: Support for multiple chronic skin conditions including:
  - Eczema
  - Psoriasis (including guttate psoriasis)
  - Keratosis Pilaris
  - Cystic/Hormonal Acne
  - Melanoma tracking
  - Vitiligo
  - Contact Dermatitis
  - Cold Sores
- **Symptom Severity Scales**: 
  - **Itch Level**: 0-10 scale to track itching intensity
  - **Pain Level**: 0-10 scale to monitor pain/discomfort
  - **Stress Level**: Track stress correlation with flare-ups
- **Sleep Tracking**: Monitor sleep quality and its impact on skin health
- **Trigger Documentation**: Log potential triggers (foods, products, environmental factors)
- **Medication Tracking**: 
  - Record medications and treatments
  - Track medication adherence
  - Note new products or treatments
- **Custom Notes**: Add detailed notes about each entry
- **Body Area Selection**: Tag symptoms to specific body locations

### 🗺️ Body Map Visualization

- **Interactive Body Map**: Visual representation of your body with selectable areas
- **Body Area Tracking**: 15+ predefined body areas including:
  - Head & Face
  - Neck, Chest, Abdomen, Pelvis
  - Left/Right Arms and Hands
  - Left/Right Thighs and Knees
  - Left/Right Feet
- **Color-Coded Severity**: 
  - Red indicators for high severity
  - Yellow for medium severity
  - Green for low severity
- **Area-Specific Records**: View all records associated with each body area
- **Visual Progress Tracking**: See at a glance which areas are affected and how severely

### 📊 Data Analysis & Visualization

- **Trend Analysis**: Track changes over time with visual trend indicators
  - Increasing trends (⬆️ red)
  - Decreasing trends (⬇️ green)
  - Stable trends (➡️ gray)
- **Statistical Summary**: 
  - Average symptom scores across all metrics
  - Comparison of recent vs. historical data
  - Progress visualization
- **Interactive Charts**: Beautiful, responsive charts powered by Recharts
- **Data Export**: 
  - Export all data as JSON files
  - Prepare reports for healthcare providers
  - Maintain offline backups
- **Quick Stats Dashboard**: 
  - Total records count
  - Image count
  - Recent activity overview

### 👤 Profile Management

- **Personal Information**: Store and manage your profile details
- **Demographics**: Age, gender, skin type
- **Medical Information**: 
  - Current conditions
  - Active medications
  - Known allergies
- **Custom Notes**: Personal health notes and observations
- **Local Storage**: All profile data stored securely on your device

### 🔗 Integrations & Export

- **API Integration**: 
  - Generate API keys for programmatic access
  - Webhook support for external integrations
  - RESTful API endpoints
- **Data Export**: 
  - JSON export for complete data portability
  - Structured format for easy import into other systems
  - Include profile and all records
- **Data Import**: Import data from external sources or backups
- **Webhook Notifications**: Configure webhooks to send data to external services

### 🚀 Progressive Web App (PWA) Features

- **Installable App**: Install on any device (mobile, tablet, desktop) for a native app experience
- **Offline Support**: Service worker enables offline functionality
- **Auto-Updates**: Automatic updates when new versions are available
- **Fast Loading**: Optimized performance with Next.js
- **Responsive Design**: Beautiful, responsive UI that works on all screen sizes
- **Mobile-First**: Designed with mobile users in mind

## 🛠️ Technology Stack

- **Framework**: Next.js 15 (React 19)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom animations
- **UI Components**: Radix UI primitives with custom styling
- **Charts**: Recharts for data visualization
- **State Management**: React hooks with localStorage
- **PWA**: Service Workers for offline support
- **Deployment**: Vercel

## 📦 Installation & Setup

### Prerequisites

- Node.js 18+ and npm/pnpm/yarn

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/jonnyterrero/SkinTrack-.git
   cd SkinTrack-
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   pnpm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   # or
   pnpm dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Building for Production

```bash
npm run build
npm start
```

### Installing as a PWA

1. Visit the deployed application in your browser
2. Click the "INSTALL FREE!" button (desktop) or use the browser menu (mobile)
3. For mobile: Tap the menu (⋮) → "Add to Home Screen" or "Install SkinTrack+"
4. The app will now work like a native application

## 📖 How to Use

### Getting Started

1. **Create Your Profile**: Navigate to the Profile tab and fill in your information
2. **First Entry**: Start by capturing an image or logging your first symptom entry
3. **Track Regularly**: Make it a habit to log entries regularly for best results
4. **Review Trends**: Use the Data Analysis tab to see your progress over time

### Best Practices

- **Consistent Tracking**: Log entries at the same time each day for accurate trends
- **Detailed Notes**: Include context about triggers, treatments, or environmental factors
- **Regular Images**: Take photos in consistent lighting conditions for better comparison
- **Export Data**: Regularly export your data as a backup
- **Share with Providers**: Use exported data during medical appointments

## 🔄 Deployment

Your project is live at:

**[https://vercel.com/jonnyterreros-projects/v0-web-app-to-mobile](https://vercel.com/jonnyterreros-projects/v0-web-app-to-mobile)**

## 🏗️ Build Your App

Continue building your app on:

**[https://v0.app/chat/projects/9biUawphA3D](https://v0.app/chat/projects/9biUawphA3D)**

## 📝 How It Works

1. **Create and modify** your project using [v0.app](https://v0.app)
2. **Deploy your chats** from the v0 interface
3. **Changes are automatically pushed** to this repository from [v0.app](https://v0.app)
4. **Vercel deploys** the latest version from this repository

## ⚠️ Important Disclaimer

**This application is NOT a diagnostic tool.** It is designed to help you track and monitor your skin conditions, but any concerning changes or new symptoms should be evaluated by a qualified healthcare professional. Always consult with a dermatologist or healthcare provider for proper diagnosis and treatment.

## 📄 License

This project is private and proprietary.

---

**Made for Karina P with love 💜**
