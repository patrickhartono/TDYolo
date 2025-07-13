import sys
import os
import site
import platform
import glob
import subprocess
import json

def get_conda_info():
    """Get conda installation info and active environment details"""
    try:
        # Try to get conda info
        result = subprocess.run(['conda', 'info', '--json'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            conda_info = json.loads(result.stdout)
            return conda_info
    except Exception as e:
        print(f"[ENV] Warning: Could not get conda info: {e}")
    return None

def find_conda_environments(username):
    """Find all possible conda environment locations"""
    system_platform = platform.system()
    possible_locations = []
    
    if system_platform == 'Windows':
        # Common Windows conda locations
        base_paths = [
            f"C:/Users/{username}/miniconda3",
            f"C:/Users/{username}/anaconda3", 
            f"C:/Users/{username}/mambaforge",
            f"C:/Users/{username}/miniforge3",
            f"C:/ProgramData/miniconda3",
            f"C:/ProgramData/anaconda3"
        ]
        
        # Check if conda info gives us custom paths
        conda_info = get_conda_info()
        if conda_info and 'envs_dirs' in conda_info:
            for env_dir in conda_info['envs_dirs']:
                if os.path.exists(env_dir):
                    base_path = os.path.dirname(env_dir)
                    if base_path not in [bp for bp in base_paths]:
                        base_paths.append(base_path)
        
        for base_path in base_paths:
            if os.path.exists(base_path):
                possible_locations.append(base_path)
                
    elif system_platform == 'Darwin':  # macOS
        # Common macOS conda locations
        base_paths = [
            f"/Users/{username}/miniconda3",
            f"/Users/{username}/opt/miniconda3",
            f"/Users/{username}/anaconda3",
            f"/Users/{username}/opt/anaconda3",
            f"/Users/{username}/mambaforge",
            f"/Users/{username}/miniforge3",
            f"/opt/miniconda3",
            f"/opt/anaconda3"
        ]
        
        # Check conda info for custom paths
        conda_info = get_conda_info()
        if conda_info and 'envs_dirs' in conda_info:
            for env_dir in conda_info['envs_dirs']:
                if os.path.exists(env_dir):
                    base_path = os.path.dirname(env_dir)
                    if base_path not in base_paths:
                        base_paths.append(base_path)
        
        for base_path in base_paths:
            if os.path.exists(base_path):
                possible_locations.append(base_path)
    
    return possible_locations

def get_python_version_from_env(conda_base):
    """Get exact Python version from conda environment"""
    system_platform = platform.system()
    
    # Try to read from pyvenv.cfg first (most reliable)
    pyvenv_cfg = os.path.join(conda_base, 'pyvenv.cfg')
    if os.path.exists(pyvenv_cfg):
        try:
            with open(pyvenv_cfg, 'r') as f:
                for line in f:
                    if line.startswith('version'):
                        version = line.split('=')[1].strip()
                        # Extract major.minor (e.g., "3.11.10" -> "3.11")
                        major_minor = '.'.join(version.split('.')[:2])
                        return f"python{major_minor}"
        except Exception as e:
            print(f"[ENV] Warning: Could not read pyvenv.cfg: {e}")
    
    # Fallback: check lib directories
    if system_platform == 'Windows':
        lib_path = os.path.join(conda_base, 'Lib')
    else:
        lib_path = os.path.join(conda_base, 'lib')
    
    if os.path.exists(lib_path):
        python_dirs = glob.glob(os.path.join(lib_path, 'python*'))
        python_dirs = [d for d in python_dirs if os.path.isdir(d)]
        if python_dirs:
            # Sort to get the highest version if multiple exist
            python_dirs.sort(reverse=True)
            python_version = os.path.basename(python_dirs[0])
            return python_version
    
    # Final fallback for expected version
    return "python3.11"

def detect_compute_device():
    """Detect available compute devices (CUDA, MPS, CPU)"""
    device_info = {
        'cuda_available': False,
        'mps_available': False,
        'device': 'cpu'
    }
    
    try:
        # Try to import torch to check device availability
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            device_info['cuda_available'] = True
            device_info['device'] = 'cuda'
            cuda_count = torch.cuda.device_count()
            print(f"[ENV] ✅ CUDA available - {cuda_count} GPU(s) detected")
            for i in range(cuda_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"[ENV]   GPU {i}: {gpu_name}")
        
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['mps_available'] = True
            device_info['device'] = 'mps'
            print(f"[ENV] ✅ MPS (Metal Performance Shaders) available")
        
        else:
            print(f"[ENV] ℹ️  Using CPU (no GPU acceleration available)")
            
    except ImportError:
        print(f"[ENV] ℹ️  PyTorch not yet loaded - device detection will be done later")
    except Exception as e:
        print(f"[ENV] Warning: Error during device detection: {e}")
    
    return device_info

def setup_windows_conda_env(conda_base, conda_env):
    """Setup conda environment for Windows"""
    print(f"[ENV] Setting up Windows conda environment...")
    print(f"[ENV] Conda base: {conda_base}")
    
    # Get Python version
    python_version = get_python_version_from_env(conda_base)
    print(f"[ENV] Detected Python version: {python_version}")
    
    # Construct paths
    conda_site_packages = os.path.join(conda_base, 'Lib', 'site-packages')
    conda_dlls = os.path.join(conda_base, 'DLLs')
    conda_library_bin = os.path.join(conda_base, 'Library', 'bin')
    conda_scripts = os.path.join(conda_base, 'Scripts')
    
    # Verify critical paths exist
    if not os.path.exists(conda_site_packages):
        raise FileNotFoundError(f"site-packages not found: {conda_site_packages}")
    
    # Add DLL directories for Windows (Python 3.8+)
    dll_dirs_added = []
    for dll_dir in [conda_dlls, conda_library_bin]:
        if os.path.exists(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
                dll_dirs_added.append(dll_dir)
                print(f"[ENV] ✅ Added DLL directory: {dll_dir}")
            except Exception as e:
                print(f"[ENV] Warning: Could not add DLL directory {dll_dir}: {e}")
    
    # Update PATH environment variable
    path_dirs = [conda_scripts, conda_library_bin, os.path.join(conda_base)]
    for path_dir in path_dirs:
        if os.path.exists(path_dir):
            current_path = os.environ.get('PATH', '')
            if path_dir not in current_path:
                os.environ['PATH'] = path_dir + os.pathsep + current_path
                print(f"[ENV] ✅ Added to PATH: {path_dir}")
    
    # Add to sys.path
    if conda_site_packages not in sys.path:
        sys.path.insert(0, conda_site_packages)
        print(f"[ENV] ✅ Added to sys.path: {conda_site_packages}")
    
    return conda_site_packages

def setup_macos_conda_env(conda_base, conda_env):
    """Setup conda environment for macOS"""
    print(f"[ENV] Setting up macOS conda environment...")
    print(f"[ENV] Conda base: {conda_base}")
    
    # Get Python version
    python_version = get_python_version_from_env(conda_base)
    print(f"[ENV] Detected Python version: {python_version}")
    
    # Construct paths
    conda_site_packages = os.path.join(conda_base, 'lib', python_version, 'site-packages')
    conda_bin = os.path.join(conda_base, 'bin')
    conda_lib = os.path.join(conda_base, 'lib')
    
    # Verify critical paths exist
    if not os.path.exists(conda_site_packages):
        raise FileNotFoundError(f"site-packages not found: {conda_site_packages}")
    
    # Update PATH environment variable
    path_dirs = [conda_bin]
    for path_dir in path_dirs:
        if os.path.exists(path_dir):
            current_path = os.environ.get('PATH', '')
            if path_dir not in current_path:
                os.environ['PATH'] = path_dir + os.pathsep + current_path
                print(f"[ENV] ✅ Added to PATH: {path_dir}")
    
    # Update library path
    if os.path.exists(conda_lib):
        dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        if conda_lib not in dyld_path:
            os.environ['DYLD_LIBRARY_PATH'] = conda_lib + os.pathsep + dyld_path
            print(f"[ENV] ✅ Added to DYLD_LIBRARY_PATH: {conda_lib}")
    
    # Add to sys.path
    if conda_site_packages not in sys.path:
        sys.path.insert(0, conda_site_packages)
        print(f"[ENV] ✅ Added to sys.path: {conda_site_packages}")
    
    # Add to PYTHONPATH
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if conda_site_packages not in current_pythonpath:
        os.environ["PYTHONPATH"] = conda_site_packages + os.pathsep + current_pythonpath
        print(f"[ENV] ✅ Added to PYTHONPATH: {conda_site_packages}")
    
    return conda_site_packages

def onStart():
    """Main function to setup conda environment for TouchDesigner"""
    print(f"[ENV] ========================================")
    print(f"[ENV] TouchDesigner Conda Environment Setup")
    print(f"[ENV] ========================================")
    
    # Get parameters from condaParam DAT
    try:
        param_dat = op('condaParam')
        if param_dat is None:
            raise Exception("condaParam DAT not found")
        
        print(f"[ENV] condaParam DAT found - rows: {param_dat.numRows}, cols: {param_dat.numCols}")
        
        # Debug DAT contents
        for row in range(min(param_dat.numRows, 5)):
            for col in range(min(param_dat.numCols, 3)):
                try:
                    cell_val = param_dat[row, col].val
                    print(f"[ENV] condaParam[{row},{col}] = '{cell_val}'")
                except:
                    print(f"[ENV] condaParam[{row},{col}] = <error>")
        
        # Extract values
        conda_env = param_dat[1,1].val if param_dat.numRows > 1 else None
        username = param_dat[2,1].val if param_dat.numRows > 2 else None
        
        print(f"[ENV] Retrieved - Username: '{username}', Environment: '{conda_env}'")
        
    except Exception as e:
        print(f"[ENV] ❌ CRITICAL ERROR: Cannot access condaParam DAT!")
        print(f"[ENV] Error: {e}")
        print(f"[ENV] Please ensure condaParam DAT exists with:")
        print(f"[ENV]   Row 1: Condaenv | your_environment_name")
        print(f"[ENV]   Row 2: User | your_username")
        return False
    
    # Validate parameters
    if not username or not conda_env or username.strip() == '' or conda_env.strip() == '':
        print(f"[ENV] ❌ ERROR: Invalid parameters from condaParam DAT!")
        print(f"[ENV] Username: '{username}', Environment: '{conda_env}'")
        return False
    
    username = username.strip()
    conda_env = conda_env.strip()
    
    # Detect platform
    system_platform = platform.system()
    print(f"[ENV] Platform: {system_platform}")
    
    if system_platform not in ['Windows', 'Darwin']:
        print(f"[ENV] ❌ Unsupported platform: {system_platform}")
        return False
    
    try:
        # Find conda installations
        print(f"[ENV] Searching for conda installations...")
        conda_locations = find_conda_environments(username)
        
        if not conda_locations:
            print(f"[ENV] ❌ No conda installations found!")
            print(f"[ENV] Please ensure conda/miniconda/miniforge is installed")
            return False
        
        print(f"[ENV] Found conda installations:")
        for loc in conda_locations:
            print(f"[ENV]   - {loc}")
        
        # Find the environment
        conda_base = None
        for location in conda_locations:
            env_path = os.path.join(location, 'envs', conda_env)
            if os.path.exists(env_path):
                conda_base = env_path
                print(f"[ENV] ✅ Found environment: {conda_base}")
                break
        
        if not conda_base:
            print(f"[ENV] ❌ Environment '{conda_env}' not found in any conda installation!")
            print(f"[ENV] Searched in:")
            for location in conda_locations:
                env_path = os.path.join(location, 'envs', conda_env)
                print(f"[ENV]   - {env_path}")
            return False
        
        # Setup environment based on platform
        if system_platform == 'Windows':
            site_packages = setup_windows_conda_env(conda_base, conda_env)
        elif system_platform == 'Darwin':
            site_packages = setup_macos_conda_env(conda_base, conda_env)
        
        print(f"[ENV] ✅ Environment setup complete!")
        print(f"[ENV] Site-packages: {site_packages}")
        
        # Detect compute devices
        print(f"[ENV] Detecting compute devices...")
        device_info = detect_compute_device()
        
        # Store device info for later use
        if hasattr(op('condaParam'), 'store'):
            op('condaParam').store('device_info', device_info)
        
        print(f"[ENV] ========================================")
        print(f"[ENV] Setup completed successfully!")
        print(f"[ENV] Ready for YOLO inference on {device_info['device']}")
        print(f"[ENV] ========================================")
        
        return True
        
    except Exception as e:
        print(f"[ENV] ❌ CRITICAL ERROR during setup: {e}")
        import traceback
        traceback.print_exc()
        return False