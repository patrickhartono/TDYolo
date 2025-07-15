# TDYolo - TouchDesigner YOLO Integration

This project was created and dedicated to all the students, colleagues, and friends during my time teaching at the Department of Computational Art at Goldsmiths, University of London 

Farewell for now! Thank you all for everything! 

A production-ready real-time object detection system integrating YOLOv11 with TouchDesigner for artistic and interactive applications, featuring **full GPU acceleration support** for both Windows (CUDA) and macOS (Metal Performance Shaders).

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-blue)
![TouchDesigner](https://img.shields.io/badge/TouchDesigner-2022.30060%2B-orange)
![Python](https://img.shields.io/badge/Python-3.11.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![MPS](https://img.shields.io/badge/MPS-Apple%20Silicon-lightgrey)

## 🎯 Features

### Core Detection Capabilities
- **Real-time YOLO Detection** - YOLOv11 with full GPU acceleration (CUDA/MPS/CPU)
- **Dynamic Class Filtering** - UI-configurable object type selection
- **Detection Limiting** - Top-N detection system with confidence-based sorting
- **Coordinate Export** - Precise bounding box data (X_Center, Y_Center, Width, Height)

### Hardware Acceleration
- **NVIDIA GPU Support** - CUDA 11.8+ with automatic detection and optimization
- **Apple Silicon Support** - Metal Performance Shaders (MPS) for M1/M2/M3 Macs
- **Multi-GPU Support** - Automatic GPU enumeration and selection
- **CPU Fallback** - Full functionality without GPU acceleration

### TouchDesigner Integration
- **Complete UI Parameter System** - All configuration via TouchDesigner interface
- **Real-time Data Export** - Live coordinate and detection data streaming
- **Visual Output** - Indexed labels with consistent color coding
- **Performance Optimization** - Frame skipping and detection limiting

### Cross-Platform Support
- **Windows & macOS** - Full compatibility with intelligent hardware detection
- **Dynamic Environment Setup** - Automatic conda environment configuration
- **Multiple Conda Distributions** - Support for Miniconda, Anaconda, MiniForge, Mambaforge

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10+ or macOS 10.15+
- **TouchDesigner**: 2022.30060 or later
- **Python**: 3.11.10 (managed via Conda)
- **Hardware**: 
  - **For GPU Acceleration (Recommended)**: NVIDIA GPU with CUDA 11.8+ or Apple Silicon Mac
  - **Minimum**: Any system with 8GB RAM (CPU-only mode)

### Required Software
- [TouchDesigner](https://derivative.ca/) (Commercial, Non-Commercial or Educational license)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Anaconda](https://www.anaconda.com/), [MiniForge](https://github.com/conda-forge/miniforge) or [MambaForge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
- Git (for cloning repository)

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: GTX 1660 or higher, CUDA 11.8+ drivers
- **Apple Silicon**: M1/M2/M3 Macs (automatic MPS support)
- **Performance**: 10-100x faster inference with GPU acceleration

## 🚀 Installation Guide

### Step 1: Clone Repository

```bash
git clone https://github.com/patrickhartono/TDYolo.git
cd TDYolo
```

### Step 2: Hardware Detection

**Check your hardware capabilities first:**

#### Windows Users
```bash
# Check for NVIDIA GPU
nvidia-smi

# Check CUDA installation
nvcc --version
```

#### macOS Users
```bash
# Check for Apple Silicon
uname -m
# Expected output: arm64 (Apple Silicon) or x86_64 (Intel)
```

### Step 3: Create Conda Environment

Choose the installation method based on your platform and hardware:

#### Option A: Windows with NVIDIA GPU (Recommended for best performance)
```bash
# Step 1: Create environment with standard packages
conda env create -f environment-win.yml -n TDYolo

# Step 2: Activate environment
conda activate TDYolo

# Step 3: Install PyTorch with CUDA support (separate installation)
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118

# Step 4: Fix NumPy compatibility (if needed)
pip install numpy==1.26.4
```

#### Option B: macOS with Apple Silicon
```bash
# Create environment with MPS support
conda env create -f environment-mac.yml -n TDYolo

# Activate environment
conda activate TDYolo
```

#### Option C: CPU-Only Installation (Any platform)
```bash
# Create basic environment
conda create -n TDYolo python=3.11.10

# Activate environment
conda activate TDYolo

# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install ultralytics opencv-python numpy matplotlib pandas pillow
```

### Step 4: Installation Verification

**Critical verification steps:**

#### Verify PyTorch Installation
```bash
conda activate TDYolo
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

#### Verify Hardware Acceleration
```bash
# Windows (CUDA)
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# macOS (MPS)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Any platform (device detection)
python -c "import torch; print('Device:', 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')"
```

#### Verify YOLO Installation
```bash
python -c "from ultralytics import YOLO; print('YOLO installation successful')"
```

**Expected outputs:**
- **GPU (CUDA)**: `CUDA available: True, GPU count: 1`
- **Apple Silicon**: `MPS available: True`
- **CPU-only**: `CUDA available: False` (this is normal for CPU installations)

## 🎛️ Configuration (UI Parameters)

Once you’ve managed to create the Conda environment and all the verifications above have succeeded, you can open the TouchDesigner file TDYolo.toe (this may continue to change, so make sure to check the latest version).

The first time you open it, you will encounter errors because you need to set the Conda path and user according to your computer’s configuration.

So, please follow the step-by-step guide below.

### Conda Parameters

<img width="398" height="349" alt="UI-Conda" src="https://github.com/user-attachments/assets/d7812319-c5bf-4cbb-8aaf-ef8de545a1ec" />
<br><br>

| Parameter | Description | Example |
|-----------|-------------|---------|
| Condaenv | Your conda environment name | `tdyolo` |
| User | Your system username | `johnsmith` |
| Conda-Refresh | Activate Selected Conda Env | `0` or `1` (momentary button) |

Finding Your Conda Path and Username
To properly configure the Conda environment in the UI, you’ll need to know two things:

1. The path to your Conda installation

2. Your system username

**For macOS Users**

To find your Conda path:
Open the Terminal and type:

```bash
which conda
# Output:
# /Users/yourusername/miniconda3/bin/conda
```

**For Windows Users**

To find your Conda path:
Open Anaconda Prompt or Command Prompt and type:

```bat
where conda
:: Output:
:: C:\Users\YourName\anaconda3\Scripts\conda.exe
```

### Yolo Parameters
<img width="398" height="571" alt="UI-TDYolo" src="https://github.com/user-attachments/assets/1fab43e5-2cfe-45e3-a83d-a37c50a1699d" />


#### **Detection Labels** (String)
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


<img width="490" height="215" alt="Screenshot 2025-07-15 115016" src="https://github.com/user-attachments/assets/34714135-fcb3-4ffb-ba3a-bd0524e60f0e" />


### Count Table
Object count aggregation:

| Column | Description | Example |
|--------|-------------|---------|
| Object_Type | Detected class name | `person` |
| Count | Number of instances | `3` |

<img width="288" height="248" alt="Screenshot 2025-07-14 214510" src="https://github.com/user-attachments/assets/e35a2a12-d878-465a-99f4-f81ba101bcfc" />


## 🔧 Troubleshooting

### GPU Acceleration Issues

#### 1. "CUDA available: False" on Windows with NVIDIA GPU
**Root Cause:** PyTorch installed without CUDA support

**Solution:**
```bash
conda activate TDYolo

# Remove CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA-enabled PyTorch
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118

# Verify CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### 2. NumPy Compatibility Warnings
**Issue:** `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**Solution:**
```bash
pip install numpy==1.26.4
```

#### 3. "PyTorch not compiled with CUDA enabled"
**Diagnosis:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
# If output shows 2.7.1+cpu, you have CPU-only PyTorch
```

**Solution:** Follow GPU installation steps above

### Environment Setup Issues

#### 4. "No module named 'ultralytics'" Error
**Solution:** Conda environment not properly activated
```bash
# Verify environment is active
conda info --envs
conda activate TDYolo

# Reinstall if necessary
pip install ultralytics
```

#### 5. "Conda path does not exist" Error
**Solution:** Incorrect condaParam configuration
- Verify username matches your system username exactly
- Ensure conda environment name is correct (case-sensitive)
- Check conda installation path supports your distribution (Miniconda/Anaconda/MiniForge)

#### 6. Environment Creation Fails with "PackagesNotFoundError"
**Issue:** Platform-specific build strings in environment files

**Solution:** Use manual installation instead:
```bash
# Create basic environment
conda create -n TDYolo python=3.11.10

# Follow platform-specific installation steps above
```

### Performance Issues

#### 7. Low Frame Rate / Poor Performance
**Diagnostic Steps:**
```bash
# Check which device is being used
# Look for this in TouchDesigner console:
[YOLO] Using CUDA    # Best performance
[YOLO] Using MPS     # Good performance (Apple Silicon)
[YOLO] Using CPU     # Lowest performance
```

**Solutions:**
- **If using CPU unexpectedly:** Follow GPU installation steps
- **For CPU-only systems:** Increase Frame Skip (2-5), reduce Detection Limit (3-5)
- **For GPU systems:** Frame Skip (0-1), Detection Limit (0-20)

#### 8. TouchDesigner UI Issues

**Custom Parameters Not Appearing:**
- Check that Script DAT Execute is "On"
- Click "Setup Parameters" to refresh
- Verify `onSetupParameters()` function exists in script

**Detection Not Working:**
- Verify video input is connected
- Check console for error messages
- Ensure conda environment is properly loaded

### Platform-Specific Issues

#### 9. Windows DLL Errors
**Solution:** Missing Visual C++ redistributables
```bash
# Install from Microsoft or via conda
conda install vs2019_win-64
```

#### 10. macOS Permission Issues
**Solution:** Grant TouchDesigner camera/microphone permissions in System Preferences > Security & Privacy

### Platform-Specific Notes

#### Windows GPU Setup
- **CUDA Requirements**: NVIDIA GPU with CUDA 11.8+ drivers
- **Two-Step Installation**: Standard packages first, then CUDA PyTorch separately
- **Common Conda Paths**: 
  - `C:/Users/username/miniconda3`
  - `C:/Users/username/miniforge3` 
  - `C:/Users/username/anaconda3`
- **DLL Dependencies**: Automatically configured for Windows environments
- **Performance**: 30-60 FPS with proper GPU setup

#### macOS Apple Silicon
- **MPS Acceleration**: Automatically detected on M1/M2/M3 Macs
- **Single-Step Installation**: environment-mac.yml includes MPS-compatible PyTorch
- **Common Conda Paths**: 
  - `/Users/username/miniconda3`
  - `/Users/username/opt/miniconda3`
  - `/Users/username/miniforge3`
- **Performance**: 20-40 FPS with Metal Performance Shaders

#### Cross-Platform Features
- **Conda Distribution Support**: Miniconda, Anaconda, MiniForge, Mambaforge
- **Automatic Hardware Detection**: Intelligently selects optimal compute device
- **Path Handling**: Automatic Windows/Unix path format conversion
- **Python Version**: Dynamic detection from conda environment

## 📁 Project Structure

```
TDYolo/
├── README.md                   # This file
├── environment-mac.yml         # macOS conda environment (Apple Silicon optimized)
├── environment-win.yml         # Windows conda environment (CUDA compatible)
├── yolo11n.pt                 # YOLOv11 model weights
├── TD-Py-yolo11-TD.toe        # Main TouchDesigner project
├── python-script/
│   ├── main-TDYolo.py         # Core YOLO detection script
│   ├── extCondaEnv.py         # Cross-platform environment setup
│   └── Log.md                 # Development history and documentation
├── Backup/                    # TouchDesigner project backups (65+ versions)
└── video/
    └── example.mp4            # Sample video for testing
```

## 🎨 Usage for Artistic Applications

### Coordinate System
- **Origin**: Top-left corner (0,0)

- **X_Center**: Pixel coordinates in source resolution (e.g.,320 for 640px input)
- **Y_Center**: Pixel coordinates in source resolution (e.g.,320 for 640px input)
- **Width/Height**: Pixel dimensions in source resolution

### TouchDesigner Integration Examples
In this example I'm using DAT to CHOP to extract XY value from the 1st and 2nd row of the Report DAT

<img width="544" height="244" alt="Screenshot 2025-07-15 114224" src="https://github.com/user-attachments/assets/720bf38c-20e3-4c99-97a1-421cb4d0a0f2" />


### Performance Recommendations by Hardware

#### NVIDIA GPU (CUDA)
- **Real-time installations**: Detection Limit 10-20, Frame Skip 0-1
- **High-precision tracking**: Detection Limit 0, Frame Skip 0
- **Expected FPS**: 30-60 FPS
- **Confidence**: 0.25-0.5

#### Apple Silicon (MPS)
- **Real-time installations**: Detection Limit 5-15, Frame Skip 1-2
- **High-precision tracking**: Detection Limit 0-10, Frame Skip 0-1
- **Expected FPS**: 20-40 FPS
- **Confidence**: 0.3-0.5

#### CPU-Only
- **Performance-critical**: Detection Limit 3-5, Frame Skip 2-5
- **Basic detection**: Detection Limit 1-3, Frame Skip 3-5
- **Expected FPS**: 3-15 FPS
- **Confidence**: 0.4-0.6 (higher to reduce processing)

## 📈 Performance Optimization

### Hardware Acceleration Details

#### NVIDIA GPU (CUDA)
- **Requirements**: GTX 1660+ or RTX series, CUDA 11.8+ drivers
- **Performance**: 10-100x faster than CPU
- **Memory**: Automatic GPU memory management
- **Detection**: Real-time inference at 30-60 FPS

#### Apple Silicon (MPS)
- **Requirements**: M1/M2/M3 Macs with macOS 12.3+
- **Performance**: 5-20x faster than CPU
- **Memory**: Unified memory architecture optimization
- **Detection**: Real-time inference at 20-40 FPS

#### CPU Fallback
- **Compatibility**: Works on any system with 8GB+ RAM
- **Performance**: Still functional for basic use cases
- **Optimization**: Frame skipping and detection limiting essential

### Memory Management
- **Model Loading**: Single initialization at startup
- **GPU Memory**: Automatic allocation and cleanup
- **Frame Processing**: Optimized numpy/tensor operations
- **Detection Limiting**: Reduces memory footprint

### Performance Monitoring
Monitor TouchDesigner console for these indicators:
```
[YOLO] Using CUDA               # GPU acceleration active
[ENV] GPU 0: RTX 4070          # GPU model detected
```

### Real-time Performance Tips
1. **Verify GPU Usage**: Check console shows CUDA/MPS, not CPU
2. **Adjust Detection Limit** based on scene complexity and hardware
3. **Use Frame Skip** appropriately for your hardware capabilities
4. **Monitor Memory Usage** in Task Manager/Activity Monitor
5. **Optimize Confidence Threshold** to balance quality vs performance

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
- **extCondaEnv.py** developed using this tutorial (code example): https://derivative.ca/community-post/tutorial/anaconda-miniconda-managing-python-environments-and-3rd-party-libraries 

## 📞 Support

For issues and questions:

### Before Reporting Issues
1. **Check Hardware Acceleration**: Verify GPU is being used with verification commands
2. **Review Troubleshooting**: Most common issues have documented solutions above
3. **Test with Example Video**: Use included `video/example.mp4` for consistent testing

### When Reporting Issues
Include this diagnostic information:
```bash
# System information
conda activate TDYolo
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')"
python -c "import platform; print('Platform:', platform.system(), platform.release())"
```

Create an issue on GitHub with:
- Operating system and version
- TouchDesigner version
- Hardware (GPU model if applicable)
- Complete console output from diagnostic commands
- Error messages and TouchDesigner console output
- Steps to reproduce

### Quick Fixes
- **Performance Issues**: First verify GPU acceleration is working
- **Import Errors**: Check conda environment is activated
- **Path Issues**: Verify condaParam DAT configuration matches your system

---

**Project Status**: Production Ready ✅  
**GPU Acceleration**: Fully Supported (CUDA/MPS) 🚀  
**Last Updated**: January 14, 2025  
**Version**: 2.0.0 (GPU Acceleration Release)
