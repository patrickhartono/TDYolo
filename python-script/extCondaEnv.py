import sys
import os
import site
import platform
import glob

def onStart():
    # Get username and environment name from condaParam DAT
    try:
        # First, let's debug what's in condaParam DAT
        param_dat = op('condaParam')
        if param_dat is not None:
            print(f"[ENV] condaParam DAT found, rows: {param_dat.numRows}, cols: {param_dat.numCols}")
            # Show all cells to understand the structure
            for row in range(min(param_dat.numRows, 5)):  # Show first 5 rows
                for col in range(min(param_dat.numCols, 5)):  # Show first 5 cols
                    try:
                        cell_val = param_dat[row, col].val
                        print(f"[ENV] condaParam[{row},{col}] = '{cell_val}'")
                    except:
                        print(f"[ENV] condaParam[{row},{col}] = <error>")
        
        # Get values from condaParam DAT - NO hardcoded defaults
        username = None
        conda_env = None
        
        # Based on condaParam DAT structure:
        # Conda environment should be from Condaenv row (row 1, col 1)
        # Username should be from User row (row 2, col 1)
        try:
            conda_env = op('condaParam')[1,1].val if op('condaParam') is not None else None
            username = op('condaParam')[2,1].val if op('condaParam') is not None else None
            print(f"[ENV] Reading from condaParam - Username: '{username}', Environment: '{conda_env}'")
        except Exception as e:
            print(f"[ENV] Error reading condaParam: {e}")
            conda_env = None
            username = None
            
        # Validate that we have valid values from condaParam DAT
        if not username or not conda_env or username == 'value' or username.strip() == '' or conda_env.strip() == '':
            print(f"[ENV] ❌ ERROR: Invalid or missing values from condaParam DAT!")
            print(f"[ENV] Please ensure condaParam DAT contains:")
            print(f"[ENV] - Valid username in User row (should be your system username)")
            print(f"[ENV] - Valid conda environment name in Condaenv row")
            print(f"[ENV] Current values: username='{username}', conda_env='{conda_env}'")
            print(f"[ENV] Expected structure:")
            print(f"[ENV]   Row 1: Condaenv | your_conda_env")
            print(f"[ENV]   Row 2: User | your_username")
            print(f"[ENV] Script cannot continue without proper configuration.")
            return
        
        # Clean up values (remove whitespace)
        username = username.strip()
        conda_env = conda_env.strip()
        
        print(f"[ENV] ✅ Valid values from condaParam - Username: '{username}', Environment: '{conda_env}'")
        print(f"[ENV] Detected platform: {platform.system()}")
        
    except Exception as e:
        # No fallback to hardcoded values - require proper configuration
        print(f"[ENV] ❌ CRITICAL ERROR: Cannot access condaParam DAT!")
        print(f"[ENV] Error details: {e}")
        print(f"[ENV] Please ensure:")
        print(f"[ENV] 1. condaParam DAT exists in your TouchDesigner project")
        print(f"[ENV] 2. condaParam DAT contains valid username and environment name")
        print(f"[ENV] 3. Check console debug output above for DAT structure")
        print(f"[ENV] Script cannot continue without proper configuration.")
        return
    
    # Cross-platform conda environment setup
    system_platform = platform.system()
    
    if system_platform == 'Windows':
        print(f"[ENV] Setting up Windows conda environment...")
        
        # Windows conda paths
        conda_base = f"C:/Users/{username}/miniconda3/envs/{conda_env}"
        conda_site_packages = f"{conda_base}/Lib/site-packages"
        conda_dlls = f"{conda_base}/DLLs"
        conda_library_bin = f"{conda_base}/Library/bin"
        
        print(f"[ENV] Windows conda base: {conda_base}")
        
        # Find Python version dynamically
        python_version = None
        lib_path = f"{conda_base}/Lib"
        if os.path.exists(lib_path):
            python_dirs = glob.glob(f"{lib_path}/python*")
            if python_dirs:
                python_version = os.path.basename(python_dirs[0])
                print(f"[ENV] Detected Python version: {python_version}")
        
        # Verify paths exist
        if not os.path.exists(conda_site_packages):
            print(f"[ENV] ❌ Windows conda path does not exist: {conda_site_packages}")
            print(f"[ENV] Please check if:")
            print(f"[ENV] 1. Username '{username}' is correct")
            print(f"[ENV] 2. Environment '{conda_env}' exists")
            print(f"[ENV] 3. Conda is installed in default location")
            return
        
        # Add DLL directories for Windows
        try:
            if os.path.exists(conda_dlls):
                os.add_dll_directory(conda_dlls)
                print(f"[ENV] Added DLL directory: {conda_dlls}")
            if os.path.exists(conda_library_bin):
                os.add_dll_directory(conda_library_bin)
                print(f"[ENV] Added DLL directory: {conda_library_bin}")
        except Exception as e:
            print(f"[ENV] Warning: Could not add DLL directories: {e}")
        
        # Add to sys.path
        if conda_site_packages not in sys.path:
            sys.path.insert(0, conda_site_packages)
            print(f"[ENV] ✅ Added to sys.path: {conda_site_packages}")
    
    elif system_platform == 'Darwin':  # macOS
        print(f"[ENV] Setting up macOS conda environment...")
        
        # Try multiple common conda installation paths
        possible_bases = [
            f"/Users/{username}/miniconda3/envs/{conda_env}",
            f"/Users/{username}/opt/miniconda3/envs/{conda_env}",
            f"/Users/{username}/anaconda3/envs/{conda_env}",
            f"/Users/{username}/opt/anaconda3/envs/{conda_env}"
        ]
        
        conda_base = None
        for base in possible_bases:
            if os.path.exists(base):
                conda_base = base
                print(f"[ENV] Found conda installation: {conda_base}")
                break
        
        if not conda_base:
            print(f"[ENV] ❌ Could not find conda environment '{conda_env}'")
            print(f"[ENV] Searched in:")
            for base in possible_bases:
                print(f"[ENV]   - {base}")
            return
        
        # Find Python version dynamically
        python_version = None
        lib_path = f"{conda_base}/lib"
        if os.path.exists(lib_path):
            python_dirs = glob.glob(f"{lib_path}/python*")
            if python_dirs:
                python_version = os.path.basename(python_dirs[0])
                print(f"[ENV] Detected Python version: {python_version}")
        
        # Construct site-packages path
        if python_version:
            conda_site_packages = f"{conda_base}/lib/{python_version}/site-packages"
        else:
            conda_site_packages = f"{conda_base}/lib/python3.11/site-packages"  # fallback
        
        if not os.path.exists(conda_site_packages):
            print(f"[ENV] ❌ macOS conda path does not exist: {conda_site_packages}")
            print(f"[ENV] Please check if:")
            print(f"[ENV] 1. Username '{username}' is correct")
            print(f"[ENV] 2. Environment '{conda_env}' exists")
            print(f"[ENV] 3. Python version is correct")
            return
        
        # Add conda paths to PATH environment variable
        conda_bin = f"{conda_base}/bin"
        conda_lib = f"{conda_base}/lib"
        
        if os.path.exists(conda_lib):
            os.environ['PATH'] = conda_lib + os.pathsep + os.environ.get('PATH', '')
            print(f"[ENV] Added to PATH: {conda_lib}")
        
        if os.path.exists(conda_bin):
            os.environ['PATH'] = conda_bin + os.pathsep + os.environ.get('PATH', '')
            print(f"[ENV] Added to PATH: {conda_bin}")
        
        # Add to sys.path
        if conda_site_packages not in sys.path:
            sys.path.insert(0, conda_site_packages)
            print(f"[ENV] ✅ Added to sys.path: {conda_site_packages}")
        
        # Add to PYTHONPATH
        os.environ["PYTHONPATH"] = conda_site_packages + ":" + os.environ.get("PYTHONPATH", "")
    
    else:
        print(f"[ENV] ❌ Unsupported platform: {system_platform}")
        print(f"[ENV] This script currently supports Windows and macOS only.")
        return
    
    print(f"[ENV] ✅ Conda environment setup complete for {system_platform}!")

    return
