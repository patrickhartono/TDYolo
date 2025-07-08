import sys
import os
import site

def onStart():
    # Get username and environment name from UI components
    try:
        # First, let's debug what's in parameter1 DAT
        param_dat = op('parameter1')
        if param_dat is not None:
            print(f"[ENV] parameter1 DAT found, rows: {param_dat.numRows}, cols: {param_dat.numCols}")
            # Show all cells to understand the structure
            for row in range(min(param_dat.numRows, 5)):  # Show first 5 rows
                for col in range(min(param_dat.numCols, 5)):  # Show first 5 cols
                    try:
                        cell_val = param_dat[row, col].val
                        print(f"[ENV] parameter1[{row},{col}] = '{cell_val}'")
                    except:
                        print(f"[ENV] parameter1[{row},{col}] = <error>")
        
        # Get values from UI - NO hardcoded defaults
        username = None
        conda_env = None
        
        # Based on DAT structure, get username and environment from correct cells
        # Username should be from User row (row 2, col 1)
        # Environment should be from Condaenv row (row 3, col 1)
        try:
            username = op('parameter1')[2,1].val if op('parameter1') is not None else None
            conda_env = op('parameter1')[3,1].val if op('parameter1') is not None else None
            print(f"[ENV] Reading from correct cells - Username: '{username}', Environment: '{conda_env}'")
        except:
            # If that fails, try the old positions as fallback
            try:
                username = op('parameter1')[1,0].val if op('parameter1') is not None else None
                conda_env = op('parameter1')[1,1].val if op('parameter1') is not None else None
                print(f"[ENV] Fallback attempt - Username: '{username}', Environment: '{conda_env}'")
            except:
                pass
            
        # Validate that we have valid values from UI
        if not username or not conda_env or username == 'value' or username.strip() == '' or conda_env.strip() == '':
            print(f"[ENV] ❌ ERROR: Invalid or missing values from UI!")
            print(f"[ENV] Please ensure parameter1 DAT contains:")
            print(f"[ENV] - Valid username in User row (should be your macOS username)")
            print(f"[ENV] - Valid conda environment name in Condaenv row")
            print(f"[ENV] Current values: username='{username}', conda_env='{conda_env}'")
            print(f"[ENV] Expected structure:")
            print(f"[ENV]   Row 2: User | your_username")
            print(f"[ENV]   Row 3: Condaenv | your_conda_env")
            print(f"[ENV] Script cannot continue without proper UI configuration.")
            return
        
        # Clean up values (remove whitespace)
        username = username.strip()
        conda_env = conda_env.strip()
        
        print(f"[ENV] ✅ Valid values from UI - Username: '{username}', Environment: '{conda_env}'")
        
    except Exception as e:
        # No fallback to hardcoded values - require UI configuration
        print(f"[ENV] ❌ CRITICAL ERROR: Cannot access UI parameters!")
        print(f"[ENV] Error details: {e}")
        print(f"[ENV] Please ensure:")
        print(f"[ENV] 1. parameter1 DAT exists in your TouchDesigner project")
        print(f"[ENV] 2. parameter1 DAT contains valid username and environment name")
        print(f"[ENV] 3. Check console debug output above for DAT structure")
        print(f"[ENV] Script cannot continue without proper UI configuration.")
        return
    
    # Path to the Conda site-packages directory on macOS
    conda_path = f"/Users/{username}/miniconda3/envs/{conda_env}/lib/python3.11/site-packages"
    
    print(f"[ENV] Using conda path: {conda_path}")
    
    # Verify that the conda environment path exists
    if not os.path.exists(conda_path):
        print(f"[ENV] Warning: Conda path does not exist: {conda_path}")
        print(f"[ENV] Please check if:")
        print(f"[ENV] 1. Username '{username}' is correct")
        print(f"[ENV] 2. Environment '{conda_env}' exists")
        print(f"[ENV] 3. Python version is 3.11 (adjust path if different)")
        return
    else:
        print(f"[ENV] Conda path verified: {conda_path}")

    # Add the path to sys.path if it's not already included
    if conda_path not in sys.path:
        sys.path.insert(0, conda_path)

    # Optionally add it to PYTHONPATH as well (safer)
    os.environ["PYTHONPATH"] = conda_path + ":" + os.environ.get("PYTHONPATH", "")

    return
