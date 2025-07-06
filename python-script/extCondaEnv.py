import sys
import os
import site

def onStart():
    # Change this to your environment name
    conda_env = "yolo11-TD"  # <- adjust this to match your Conda environment name

    # Path to the Conda site-packages directory on macOS
    conda_path = f"/Users/patrickhartono/miniconda3/envs/{conda_env}/lib/python3.11/site-packages"

    # Add the path to sys.path if it's not already included
    if conda_path not in sys.path:
        sys.path.insert(0, conda_path)

    # Optionally add it to PYTHONPATH as well (safer)
    os.environ["PYTHONPATH"] = conda_path + ":" + os.environ.get("PYTHONPATH", "")

    return
