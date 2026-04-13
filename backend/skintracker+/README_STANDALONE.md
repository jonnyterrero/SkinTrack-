# SkinTrack+ Standalone Python Version

## 🚀 Quick Start (No Installation Required!)

If you just want to try the application immediately:

1. **Install Python** (if not already installed):
   - Go to https://www.python.org/downloads/
   - Download Python 3.8 or higher
   - **IMPORTANT**: Check "Add Python to PATH" during installation
   - Restart your command prompt/terminal

2. **Run the application**:
   ```bash
   python skintrack_standalone.py
   ```

That's it! The standalone version requires minimal dependencies and will work with just Python installed.

## 📋 What You Get

The standalone version provides a **command-line interface** with all the core features:

- ✅ **Create and manage lesions** (skin conditions)
- ✅ **Log symptoms** (itch, pain, sleep, stress)
- ✅ **Track medications** and adherence
- ✅ **View history** and trends
- ✅ **Export data** to CSV files
- ✅ **Local database** (SQLite) - all data stays on your computer

## 🆚 Standalone vs Web Version

| Feature | Standalone (CLI) | Web Interface |
|---------|------------------|---------------|
| **Ease of use** | ✅ Simple menu system | ✅ Beautiful web interface |
| **Image analysis** | ❌ No image processing | ✅ Full image analysis |
| **Dependencies** | ✅ Minimal (2 packages) | ❌ Many packages |
| **Installation** | ✅ Just Python | ❌ Python + many libraries |
| **Data storage** | ✅ Same SQLite database | ✅ Same SQLite database |
| **Export** | ✅ CSV export | ✅ CSV + PDF export |

## 🎯 Perfect For

- **Beginners** who want to start tracking immediately
- **Users with limited system resources**
- **Quick symptom logging** without image analysis
- **Testing the application** before full installation

## 📁 File Structure

```
skintracker+/
├── skintrack_standalone.py    # 🟢 Main standalone application
├── skintrack_app.py          # 🟡 Full web interface
├── requirements_minimal.txt   # 📦 Minimal dependencies
├── requirements.txt          # 📦 Full dependencies
├── QUICK_START.bat           # 🚀 Windows quick start guide
├── install.bat               # 🔧 Windows installer
├── install.ps1               # 🔧 PowerShell installer
└── README.md                 # 📖 Full documentation
```

## 🔧 Installation Options

### Option 1: Minimal Installation (Recommended)
```bash
pip install -r requirements_minimal.txt
python skintrack_standalone.py
```

### Option 2: Full Installation (For Web Interface)
```bash
pip install -r requirements.txt
streamlit run skintrack_app.py
```

## 🎮 How to Use

1. **Start the application**:
   ```bash
   python skintrack_standalone.py
   ```

2. **Follow the menu**:
   - Create a new lesion (skin condition)
   - Add symptom records
   - View your history
   - Export your data

3. **Example session**:
   ```
   ============================================================
   🧴 SkinTrack+ - Chronic Skin Condition Tracker
   ============================================================
   Standalone Python Version
   For full web interface: streamlit run skintrack_app.py
   ============================================================
   ✅ Database initialized successfully!

   📋 Main Menu:
   1. Create new lesion
   2. List all lesions
   3. Add record to lesion
   4. View lesion history
   5. Add medication schedule
   6. Export data
   7. Initialize database
   8. Exit
   ----------------------------------------
   Enter your choice (1-8): 1
   ```

## 📊 Data Storage

All your data is stored locally in:
- **Database**: `skintrack_data/skintrack.db`
- **Images**: `skintrack_data/images/` (web version only)
- **Exports**: CSV files in the current directory

Your data is **private** and never leaves your computer.

## 🆘 Troubleshooting

### "Python was not found"
- Install Python from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation
- Restart your command prompt

### "Module not found"
- Install minimal dependencies: `pip install -r requirements_minimal.txt`
- Or install full dependencies: `pip install -r requirements.txt`

### "Permission denied"
- Run as administrator (Windows)
- Check file permissions

## 🔄 Upgrading to Full Version

If you want the full web interface with image analysis:

1. Install full dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the web version:
   ```bash
   streamlit run skintrack_app.py
   ```

3. Your data will be automatically shared between both versions!

## 📞 Support

- **Data compatibility**: Both versions use the same database
- **Feature comparison**: See table above
- **Installation help**: Run `QUICK_START.bat` on Windows

## ⚠️ Important Notes

- This is **NOT a diagnostic tool**
- Always consult healthcare professionals for medical decisions
- Data is stored locally for privacy
- Regular backups of your data are recommended

---

**Ready to start?** Just run `python skintrack_standalone.py` after installing Python!
