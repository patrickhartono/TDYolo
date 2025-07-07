# TouchDesigner YOLO Script
# Copy this entire file content into your Script DAT in TouchDesigner
# Make sure DAT Execute is set to "On"

# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2
from ultralytics import YOLO
import torch

# --- Global Initialization ---

def get_optimal_device():
    """Checks for available hardware and returns the optimal torch device."""
    if torch.backends.mps.is_available():
        print("[YOLO] Using Metal Performance Shaders (MPS) for M4 Pro optimization")
        return 'mps'
    elif torch.cuda.is_available():
        print("[YOLO] Using CUDA")
        return 'cuda'
    else:
        print("[YOLO] Using CPU")
        return 'cpu'

# Load YOLO model once with the optimal device
try:
    device = get_optimal_device()
    model = YOLO('yolo11n.pt', task='detect')
    model.to(device)
    print("[YOLO] Model yolo11n.pt loaded successfully.")
except Exception as e:
    print(f"[YOLO] Error loading model: {e}")
    model = None
    device = 'cpu' # Fallback to CPU if model loading fails

# Global frame counter for frame skipping
frame_counter = 0

# --- TouchDesigner Callbacks ---

def onSetupParameters(scriptOp):
    """Sets up the custom parameters for the YOLO script in TouchDesigner."""
    page = scriptOp.appendCustomPage('YOLO')
    page.appendToggle('Drawbox', label='Draw Bounding Box').default = True
    page.appendStr('Classes', label='Detection Classes').default = ''
    p_conf = page.appendFloat('Confidence', label='Confidence Threshold')
    p_conf.default = 0.25
    p_conf.min = 0.0
    p_conf.max = 1.0
    p_skip = page.appendInt('Frameskip', label='Frame Skip (0=process all)')
    p_skip.default = 0
    p_skip.min = 0
    p_skip.max = 10

def onCook(scriptOp):
    """Main callback function that processes video frames."""
    global frame_counter
    
    if not scriptOp.inputs or scriptOp.inputs[0] is None or model is None:
        return

    # --- 1. Get Parameters & Handle Frame Skipping ---
    params = get_parameters(scriptOp)
    frame_counter += 1
    if params['frame_skip'] > 0 and (frame_counter % (params['frame_skip'] + 1) != 0):
        # If skipping, just pass the input frame through without processing
        scriptOp.copyNumpyArray(scriptOp.inputs[0].numpyArray())
        return

    # --- 2. Pre-process Frame ---
    frame = scriptOp.inputs[0].numpyArray()
    if frame is None:
        return
    bgr_frame = convert_to_bgr(frame)

    # --- 3. Run Detection ---
    class_filter = parse_class_filter(params['classes_str'], model)
    detections = run_detection(bgr_frame, params['confidence'], class_filter, device, model)
    
    # --- 4. Update DAT Tables ---
    update_report_table(op('report'), detections, model.names)
    update_summary_table(op('summary'), detections, model.names)

    # --- 5. Prepare Output Frame ---
    output_frame = process_output_frame(bgr_frame, detections, params['draw_box'])
    
    # --- 6. Send to TouchDesigner ---
    scriptOp.copyNumpyArray(output_frame)

# --- Helper Functions ---

def get_parameters(scriptOp):
    """Extracts and returns script parameters from the TouchDesigner UI."""
    params = {}
    try:
        params['draw_box'] = scriptOp.par.Drawbox.eval()
        params['confidence'] = scriptOp.par.Confidence.eval()
        params['classes_str'] = scriptOp.par.Classes.val.strip()
        params['frame_skip'] = scriptOp.par.Frameskip.eval()
    except AttributeError:
        # Fallback to defaults if parameters don't exist
        params['draw_box'] = True
        params['confidence'] = 0.25
        params['classes_str'] = ''
        params['frame_skip'] = 0
    return params

def convert_to_bgr(td_frame):
    """Converts a TouchDesigner RGBA frame to a BGR frame for OpenCV."""
    return cv2.cvtColor(np.clip(td_frame * 255, 0, 255).astype(np.uint8, copy=False), cv2.COLOR_RGBA2BGR)

def parse_class_filter(classes_str, yolo_model):
    """Parses a comma-separated string of class names into a list of class indices."""
    if not classes_str:
        return None
    
    class_names = [name.strip().lower() for name in classes_str.split(',') if name.strip()]
    if not class_names:
        return None
        
    yolo_names = {v.lower(): k for k, v in yolo_model.names.items()} # name:index map
    class_indices = [yolo_names[name] for name in class_names if name in yolo_names]
    
    if not class_indices:
        print(f"[YOLO] Warning: No valid classes found for: {class_names}")
        print(f"[YOLO] Available classes: {list(yolo_model.names.values())[:10]}...")
        return None
        
    return class_indices

def run_detection(bgr_frame, confidence, class_filter, device, model):
    """Runs YOLO object detection on a single frame."""
    with torch.no_grad():
        results = model.predict(
            source=bgr_frame,
            conf=confidence,
            classes=class_filter,
            verbose=False,
            device=device,
            half=True if device == 'mps' else False,
            imgsz=640
        )
    return results[0] # Return the detections for the first image

def update_report_table(report_dat, detections, model_names):
    """Clears and updates the 'report' DAT with detailed detection info."""
    if report_dat is None:
        return
    report_dat.clear()
    report_dat.appendRow(['Object_Type', 'Confidence', 'ID'])
    
    if not detections.boxes:
        report_dat.appendRow(['none', '0.000', '0'])
        return

    object_counters = {}
    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = model_names[class_id]
        confidence_val = float(box.conf[0])
        
        object_counters[class_name] = object_counters.get(class_name, 0) + 1
        obj_id = object_counters[class_name]
        
        report_dat.appendRow([class_name, f'{confidence_val:.3f}', str(obj_id)])

def update_summary_table(summary_dat, detections, model_names):
    """Clears and updates the 'summary' DAT with object counts."""
    if summary_dat is None:
        return
    summary_dat.clear()
    summary_dat.appendRow(['Object_Type', 'Count'])

    if not detections.boxes:
        summary_dat.appendRow(['none', '0'])
        return
        
    object_counts = {}
    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = model_names[class_id]
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
    for class_name, count in object_counts.items():
        summary_dat.appendRow([class_name, str(count)])

def process_output_frame(bgr_frame, detections, draw_box):
    """Draws bounding boxes if enabled and converts the frame to RGBA for TouchDesigner."""
    rendered_frame = bgr_frame
    if draw_box and detections.boxes:
        rendered_frame = detections.plot()  # Use YOLO's built-in plotting

    # Convert to RGBA for TouchDesigner and flip vertically
    rgba_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGBA)
    return cv2.flip(rgba_frame, 0)
