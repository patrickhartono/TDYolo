# TDYolo - TouchDesigner YOLO Integration

This project was created and dedicated to all the students, colleagues, and friends during my time teaching at the Department of Computational Art at Goldsmiths, University of London 

It is a token of my gratitude and farewell for now! Thank you all for everything! 


A production-ready real-time object detection system integrating YOLOv11 with TouchDesigner for artistic and interactive applications.

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-blue)
![TouchDesigner](https://img.shields.io/badge/TouchDesigner-2022.30060%2B-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)

## 🎯 Features

### Core Detection Capabilities
- **Real-time YOLO Detection** - YOLOv11 with hardware optimization (MPS/CUDA/CPU)
- **Dynamic Class Filtering** - UI-configurable object type selection
- **Detection Limiting** - Top-N detection system with confidence-based sorting
- **Coordinate Export** - Precise bounding box data (X_Center, Y_Center, Width, Height)

### TouchDesigner Integration
- **Complete UI Parameter System** - All configuration via TouchDesigner interface
- **Real-time Data Export** - Live coordinate and detection data streaming
- **Visual Output** - Indexed labels with consistent color coding
- **Performance Optimization** - Frame skipping and detection limiting

### Cross-Platform Support
- **Windows & macOS** - Full compatibility with auto-detection
- **Dynamic Environment Setup** - Automatic conda environment configuration
- **Multiple Installation Paths** - Support for various conda distributions

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10+ or macOS 10.15+
- **TouchDesigner**: 2022.30060 or later
- **Python**: 3.11+ (managed via Conda)
- **Hardware**: GPU recommended for optimal performance

### Required Software
- [TouchDesigner](https://derivative.ca/) (Commercial or Educational license)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Git (for cloning repository)

## 🚀 Installation Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/patrickhartono/TDYolo.git
cd TDYolo
```

### Step 2: Create Conda Environment

#### Option A: Create from environment.yml (Recommended)
```bash
# Create environment with custom name
conda env create -f environment.yml -n your_environment_name

# Activate environment
conda activate your_environment_name
```

#### Option B: Manual Environment Setup
```bash
# Create new environment
conda create -n tdyolo python=3.11

# Activate environment
conda activate tdyolo

# Install required packages
conda install pytorch torchvision torchaudio
pip install ultralytics opencv-python numpy
```

### Step 3: Verify Installation

```bash
# Test YOLO installation
python -c "from ultralytics import YOLO; print('YOLO installation successful')"

# Test hardware acceleration (macOS)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Test hardware acceleration (Windows/Linux)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Configure TouchDesigner Project

1. **Open TouchDesigner Project**
   - Launch TouchDesigner
   - Open `TD-Py-yolo11-TD.toe` from the project directory

2. **Configure condaParam DAT**
   - Locate the `condaParam` Table DAT in the project
   - Update the following values:

   | Row | Column 0 | Column 1 |
   |-----|----------|-----------|
   | 0 | name | value |
   | 1 | Condaenv | `your_environment_name` |
   | 2 | User | `your_system_username` |
   | 3 | Conda | 0 |

   **Example:**
   ```
   Row 1: Condaenv | tdyolo
   Row 2: User     | johnsmith
   ```

3. **Setup Environment Script**
   - Copy contents of `python-script/extCondaEnv.py`
   - Paste into the `extCondaEnv` Script DAT
   - Set Execute to "On"
   - Restart TouchDesigner to initialize environment

4. **Setup Main YOLO Script**
   - Copy contents of `python-script/main-TDYolo.py`
   - Paste into the main `script2` Script DAT
   - Set Execute to "On"

### Step 5: Verify Setup

1. **Check Console Output**
   - Look for `[ENV] ✅ Conda environment setup complete`
   - Verify no error messages appear

2. **Test Detection**
   - Ensure video input is connected
   - Check that bounding boxes appear on objects
   - Verify data appears in report and summary tables

## 🎛️ Configuration

### UI Parameters

Access custom parameters in the Script DAT properties:

#### **Draw Bounding Box** (Toggle)
- Enable/disable visual bounding box rendering
- Default: On

#### **Detection Classes** (String)
- Comma-separated list of object classes to detect
- Example: `person,car,dog`
- Leave empty for all classes

#### **Confidence Threshold** (Float: 0.0-1.0)
- Minimum confidence for detections
- Default: 0.25
- Lower = more detections, Higher = higher quality

#### **Frame Skip** (Integer: 0-10)
- Performance optimization setting
- 0 = process every frame
- Higher values = skip frames for better performance

#### **Detection Limit** (Integer: 0-100)
- Maximum number of objects to detect
- 0 = unlimited
- Limits to highest confidence detections

### condaParam DAT Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| Condaenv | Your conda environment name | `tdyolo` |
| User | Your system username | `johnsmith` |
| Conda | Enable/disable conda setup | `0` or `1` |

## 📊 Data Output

### Report Table
Real-time detection data with coordinates:

| Column | Description | Example |
|--------|-------------|---------|
| Object_Type | Detected class name | `person` |
| Confidence | Detection confidence | `0.856` |
| X_Center | Horizontal center position | `450.5` |
| Y_Center | Vertical center position | `320.2` |
| Width | Bounding box width | `120.0` |
| Height | Bounding box height | `180.0` |
| ID | Object instance number | `1` |

### Summary Table
Object count aggregation:

| Column | Description | Example |
|--------|-------------|---------|
| Object_Type | Detected class name | `person` |
| Count | Number of instances | `3` |

## 🔧 Troubleshooting

### Common Issues

#### 1. "No module named 'ultralytics'" Error
**Solution:** Conda environment not properly activated
```bash
# Verify environment is active
conda info --envs
conda activate your_environment_name

# Reinstall if necessary
pip install ultralytics
```

#### 2. Custom Parameters Not Appearing
**Solution:** TouchDesigner parameter properties issue
- Check that Script DAT Execute is "On"
- Click "Setup Parameters" to refresh
- Verify `onSetupParameters()` function exists in script

#### 3. "Conda path does not exist" Error
**Solution:** Incorrect condaParam configuration
- Verify username matches your system username
- Ensure conda environment name is correct
- Check conda installation path

#### 4. Performance Issues
**Solutions:**
- Increase Frame Skip value (1-3)
- Reduce Detection Limit (5-10 objects)
- Lower Confidence Threshold
- Ensure GPU acceleration is working

#### 5. Windows DLL Errors
**Solution:** Missing Visual C++ redistributables
```bash
# Install from Microsoft or via conda
conda install vs2019_win-64
```

### Platform-Specific Notes

#### macOS
- **MPS Acceleration**: Automatically detected on Apple Silicon Macs
- **Conda Paths**: Supports `/Users/username/miniconda3` and `/Users/username/opt/miniconda3`
- **Python Version**: Auto-detected from conda environment

#### Windows
- **CUDA Support**: Automatically detected if available
- **DLL Dependencies**: Automatically configured
- **Conda Paths**: Supports `C:/Users/username/miniconda3`
- **Path Separators**: Handles Windows path format automatically

## 📁 Project Structure

```
TDYolo/
├── README.md                   # This file
├── environment.yml             # Conda environment specification
├── yolo11n.pt                 # YOLOv11 model weights
├── TD-Py-yolo11-TD.toe        # Main TouchDesigner project
├── python-script/
│   ├── main-TDYolo.py         # Core YOLO detection script
│   ├── extCondaEnv.py         # Cross-platform environment setup
│   └── Log.md                 # Development history and documentation
├── Backup/                    # TouchDesigner project backups
└── video/
    └── example.mp4            # Sample video for testing
```

## 🎨 Usage for Artistic Applications

### Coordinate System
- **Origin**: Top-left corner (0,0)
- **X_Center**: 0 to input_width (relative to source resolution)
- **Y_Center**: 0 to input_height (relative to source resolution)
- **Width/Height**: Pixel dimensions in source resolution

### TouchDesigner Integration Examples
```python
# Access detection data in TouchDesigner
report_data = op('report')
x_center = float(report_data[1, 'X_Center'].val)  # First detection X
y_center = float(report_data[1, 'Y_Center'].val)  # First detection Y

# Use for positioning effects
op('transform1').par.tx = x_center / 1920  # Normalize to 0-1
op('transform1').par.ty = y_center / 1080
```

### Performance Recommendations
- **Real-time installations**: Detection Limit 5-10, Frame Skip 1-2
- **High-precision tracking**: Detection Limit 0, Frame Skip 0
- **Performance-critical**: Detection Limit 3-5, Frame Skip 2-3

## 🔄 Updates and Maintenance

### Updating YOLOv11 Model
```bash
# Download latest model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt

# Or use different model sizes
# yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large), yolo11x.pt (extra large)
```

### Updating Dependencies
```bash
# Update conda environment
conda env update -f environment.yml

# Update specific packages
pip install --upgrade ultralytics opencv-python
```

## 📈 Performance Optimization

### Hardware Acceleration
- **Apple Silicon**: MPS acceleration (automatic)
- **NVIDIA GPU**: CUDA acceleration (automatic)
- **CPU Only**: Still functional, reduced performance

### Memory Management
- **Model Loading**: Loaded once at startup for efficiency
- **Frame Processing**: Optimized numpy array handling
- **Detection Limiting**: Reduces processing overhead

### Real-time Performance Tips
1. **Adjust Detection Limit** based on scene complexity
2. **Use Frame Skip** for performance-critical applications
3. **Optimize Confidence Threshold** for your use case
4. **Monitor console output** for performance metrics

## 🤝 Contributing

We welcome contributions! Please see our development history in `python-script/Log.md` for detailed implementation notes.

### Development Setup
1. Fork the repository
2. Create feature branch
3. Follow existing code style
4. Test on both Windows and macOS if possible
5. Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv11 implementation
- **Derivative** for TouchDesigner platform
- **PyTorch** team for hardware acceleration frameworks
- **OpenCV** community for computer vision tools

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review `python-script/Log.md` for implementation details
3. Create an issue on GitHub with:
   - Operating system and version
   - TouchDesigner version
   - Error messages and console output
   - Steps to reproduce

---

**Project Status**: Production Ready ✅  
**Last Updated**: July 10, 2025  
**Version**: 1.0.0