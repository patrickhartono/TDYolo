# TouchDesigner YOLO Script
# Copy this entire file content into your Script DAT in TouchDesigner
# Make sure DAT Execute is set to "On"

# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2
from ultralytics import YOLO
import torch

# Check if MPS (Metal Performance Shaders) is available on M4 Pro
def get_optimal_device():
    if torch.backends.mps.is_available():
        print("[YOLO] Using Metal Performance Shaders (MPS) for M4 Pro optimization")
        return 'mps'
    elif torch.cuda.is_available():
        print("[YOLO] Using CUDA")
        return 'cuda'
    else:
        print("[YOLO] Using CPU")
        return 'cpu'

# Load YOLO model once with optimal device
device = get_optimal_device()
model = YOLO('yolo11n.pt', task='detect')
model.to(device)  # Move model to optimal device

def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage('YOLO')
    
    # Toggle to enable/disable bounding box drawing
    p = page.appendToggle('Drawbox', label='Draw Bounding Box')
    p[0].default = True
    
    # String parameter for class filtering (comma separated)
    p = page.appendStr('Classes', label='Detection Classes')
    p[0].default = ''  # Empty by default - detect all classes
    
    # Confidence threshold
    p = page.appendFloat('Confidence', label='Confidence Threshold')
    p[0].default = 0.25  # Lowered from 0.5 for better detection
    p[0].min = 0.0
    p[0].max = 1.0
    
    # Frame skip for performance optimization
    p = page.appendInt('Frameskip', label='Frame Skip (0=process all)')
    p[0].default = 0  # 0 = process every frame, 1 = skip 1 frame, etc.
    p[0].min = 0
    p[0].max = 10
    
    return

# Global frame counter for frame skipping optimization
frame_counter = 0

# Remove the onPulse function since we no longer need it
# Class filtering is now handled directly in onCook

def onCook(scriptOp):
    global frame_counter
    
    # Ensure input is connected
    if not scriptOp.inputs or scriptOp.inputs[0] is None:
        return
    
    # Get frame skip parameter for performance optimization
    try:
        frame_skip = scriptOp.par.Frameskip.eval() if hasattr(scriptOp.par, 'Frameskip') else 0
    except:
        frame_skip = 0
    
    # Frame skipping logic for better performance
    frame_counter += 1
    skip_detection = frame_skip > 0 and (frame_counter % (frame_skip + 1) != 0)
    
    # Check if parameters exist, if not use defaults
    try:
        drawBox = scriptOp.par.Drawbox.eval() if hasattr(scriptOp.par, 'Drawbox') else True
        # print(f'[DEBUG] DrawBox: {drawBox}')
    except:
        drawBox = True
        # print('[DEBUG] DrawBox: default (True)')
        
    try:
        confidence = scriptOp.par.Confidence.eval() if hasattr(scriptOp.par, 'Confidence') else 0.25
        # print(f'[DEBUG] Confidence: {confidence}')
    except:
        confidence = 0.25
        # print('[DEBUG] Confidence: default (0.25)')
        
    try:
        # Try different ways to access the parameter
        if hasattr(scriptOp.par, 'Classes'):
            classes_str_raw = scriptOp.par.Classes.val  # Use .val instead of .eval()
            classes_str = classes_str_raw.strip() if classes_str_raw is not None else ''
        else:
            classes_str = ''
        # print(f'[DEBUG] Classes from UI: "{classes_str}"')
    except Exception as e:
        classes_str = ''
        # print(f'[DEBUG] Classes: error accessing parameter: {e}')
    
    frame = scriptOp.inputs[0].numpyArray()
    if frame is None:
        return

    # Convert RGBA float[0–1] to uint8, then to BGR for OpenCV/YOLO
    # Optimized conversion with explicit dtype to reduce memory allocation
    bgr = cv2.cvtColor(np.clip(frame * 255, 0, 255).astype(np.uint8, copy=False), cv2.COLOR_RGBA2BGR)

    # Parse class filter - determine what to detect
    class_filter = None  # None means detect all classes
    if classes_str:  # If there's text in the Classes field
        # Convert class names to indices (YOLO class mapping)
        class_names = [name.strip() for name in classes_str.split(',') if name.strip()]
        if class_names:
            # Get YOLO class names and find indices
            yolo_names = model.names  # Dict of {index: class_name}
            class_indices = []
            for class_name in class_names:
                for idx, yolo_name in yolo_names.items():
                    if yolo_name.lower() == class_name.lower():
                        class_indices.append(idx)
                        break
            if class_indices:
                class_filter = class_indices
                # print(f'[YOLO] Detecting only: {class_names} -> indices: {class_indices}')
            else:
                print(f'[YOLO] Warning: No valid classes found for: {class_names}')
                print(f'[YOLO] Available classes: {list(yolo_names.values())[:10]}...') # Show first 10
    
    if class_filter is None:
        # print('[YOLO] Detecting all objects (no filter)')
        pass

    # Initialize with original image
    rendered = bgr
    
    # Skip detection if frame skipping is enabled for this frame
    if skip_detection:
        # print(f'[PERF] Skipping detection for frame {frame_counter} (frame_skip={frame_skip})')
        pass
    else:
        # Run YOLO detection with MPS optimization and appropriate filtering
        with torch.no_grad():  # Disable gradient computation for inference speedup
            results = model.predict(
                source=bgr, 
                conf=confidence, 
                classes=class_filter, 
                verbose=False,
                device=device,  # Explicitly use optimal device
                half=True if device == 'mps' else False,  # Use half precision on MPS for speed
                imgsz=640  # Explicit image size for consistent performance
            )
        
        det = results[0]
        # print(f'[YOLO] Found {len(det.boxes)} detections')

        # Show detailed detection information
        if len(det.boxes) > 0:
            for i, box in enumerate(det.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                # print(f'  Detection {i+1}: {class_name} ({confidence:.2f})')

        # OUTPUT TO DAT TABLE named "report"
        try:
            # Find the report table DAT
            report_table = op('report')
            if report_table is not None:
                # Clear existing data
                report_table.clear()
                
                # Set column headers first
                report_table.appendRow(['Object_Type', 'Confidence', 'ID'])
                
                # Count objects by type and collect their confidences
                total_detections = len(det.boxes)
                if total_detections > 0:
                    # Dictionary to store object counts for ID assignment
                    object_counters = {}
                    
                    # Add rows with ID per object type
                    for i, box in enumerate(det.boxes):
                        class_id = int(box.cls[0])
                        confidence_val = float(box.conf[0])
                        class_name = model.names[class_id]
                        
                        # Increment counter for this object type
                        if class_name not in object_counters:
                            object_counters[class_name] = 0
                        object_counters[class_name] += 1
                        
                        # Add row: [object_name, confidence, id_within_type]
                        report_table.appendRow([class_name, f'{confidence_val:.3f}', str(object_counters[class_name])])
                else:
                    # If no detections, add empty row
                    report_table.appendRow(['none', '0.000', '0'])
                    
        except Exception as e:
            print(f'[TABLE] Error updating report table: {e}')

        # OUTPUT TO SUMMARY TABLE named "summary"
        try:
            # Find the summary table DAT
            summary_table = op('summary')
            if summary_table is not None:
                # Clear existing data
                summary_table.clear()
                
                # Set column headers
                summary_table.appendRow(['Object_Type', 'Count'])
                
                # Count objects by type
                if total_detections > 0:
                    # Dictionary to store object counts
                    object_counts = {}
                    
                    for i, box in enumerate(det.boxes):
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        if class_name not in object_counts:
                            object_counts[class_name] = 0
                        object_counts[class_name] += 1
                    
                    # Add summary rows
                    for class_name, count in object_counts.items():
                        summary_table.appendRow([class_name, str(count)])
                else:
                    # If no detections
                    summary_table.appendRow(['none', '0'])
                    
        except Exception as e:
            print(f'[TABLE] Error updating summary table: {e}')

        # Only draw bounding boxes if there are detections AND drawBox is enabled
        if drawBox and len(det.boxes) > 0:
            rendered = det.plot()      # BGR image with bounding boxes
        # If no detections or drawBox is False, use original image (already set above)

    # Convert to RGBA for TouchDesigner and flip vertically for correct orientation
    # Optimized memory handling with explicit copy=False where safe
    rgba = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGBA)
    rgba = cv2.flip(rgba, 0)  # Vertical flip to fix YOLO text orientation

    # Final output with optimized array handling
    scriptOp.copyNumpyArray(rgba)  # Remove redundant .astype(np.uint8) as it's already uint8
    return
