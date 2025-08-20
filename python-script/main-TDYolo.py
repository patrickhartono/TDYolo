# TouchDesigner YOLO Script
# Copy this entire file content into your Script DAT in TouchDesigner
# Make sure DAT Execute is set to "On"

# me - this DAT
# scriptOp - the OP which is cooking

import numpy as np
import cv2
from ultralytics import YOLO
import torch

# Define a list of colors in BGR format
# Red, Green, Blue, Purple
CLASS_COLORS_PALETTE = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 0, 255)   # Purple
]

# Map each class name to a consistent color
class_color_map = {}

# Check if MPS (Metal Performance Shaders) is available on M4 Pro
def get_optimal_device():
    try:
        if torch.backends.mps.is_available():
            print("[YOLO] Using Metal Performance Shaders (MPS) for M4 Pro optimization")
            # Optimize Metal GPU memory allocation
            try:
                torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                print("[YOLO] Metal GPU memory pool optimized (80% allocation)")
            except Exception as e:
                print(f"[YOLO] Warning: Could not optimize Metal memory pool: {e}")
            return 'mps'
        elif torch.cuda.is_available():
            print("[YOLO] Using CUDA")
            return 'cuda'
        else:
            print("[YOLO] Using CPU")
            return 'cpu'
    except Exception as e:
        print(f"[YOLO] Error during device detection: {e}. Falling back to CPU")
        return 'cpu'

# Load YOLO model once with optimal device
device = get_optimal_device()
model = YOLO('yolo11n.pt', task='detect')
model.to(device)  # Move model to optimal device

# Model compilation for PyTorch 2.0+ performance boost
try:
    if hasattr(torch, 'compile') and device in ['mps', 'cuda']:
        print("[YOLO] Compiling model for optimized inference...")
        model.model = torch.compile(model.model, mode='max-autotune')
        print("[YOLO] Model compilation complete - expect 10-15% speedup")
except Exception as e:
    print(f"[YOLO] Model compilation failed (continuing with normal mode): {e}")

# Populate class_color_map
for i, class_name in enumerate(model.names.values()):
    class_color_map[class_name] = CLASS_COLORS_PALETTE[i % len(CLASS_COLORS_PALETTE)]

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
    p[0].normMin = 0.0
    p[0].normMax = 1.0
    
    # Frame skip for performance optimization
    p = page.appendInt('Frameskip', label='Frame Skip (0=process all)')
    p[0].default = 0  # 0 = process every frame, 1 = skip 1 frame, etc.
    p[0].normMin = 0
    p[0].normMax = 10
    
    # Detection limit
    p = page.appendInt('Detectionlimit', label='Detection Limit (0=unlimited)')
    p[0].default = 0  # 0 = unlimited detection
    p[0].normMin = 0
    p[0].normMax = 100
    
    return

# Global frame counter for frame skipping optimization
frame_counter = 0
last_detection_count = 0  # Track detection density for dynamic resolution
performance_stats = {'avg_inference_time': 0.0, 'frame_count': 0}  # Performance monitoring

# Remove the onPulse function since we no longer need it
# Class filtering is now handled directly in onCook

def onCook(scriptOp):
    global frame_counter, last_detection_count, performance_stats
    import time
    start_time = time.time()
    
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
        detection_limit = scriptOp.par.Detectionlimit.eval() if hasattr(scriptOp.par, 'Detectionlimit') else 0
        # print(f'[DEBUG] Detection Limit: {detection_limit}')
    except:
        detection_limit = 0
        # print('[DEBUG] Detection Limit: default (0)')
        
    try:
        # Get classes from parameter1 DAT using expression op('parameter1')[1, 1].val
        classes_str_raw = op('parameter1')[1, 1].val if op('parameter1') is not None else ''
        classes_str = classes_str_raw.strip() if classes_str_raw is not None else ''
        # print(f'[DEBUG] Classes from parameter1[1,1]: "{classes_str}"')  # Disabled debug
    except Exception as e:
        classes_str = ''
        # print(f'[DEBUG] Classes: error accessing parameter1[1,1]: {e}')  # Disabled debug
    
    frame = scriptOp.inputs[0].numpyArray()
    if frame is None:
        return

    # Convert RGBA float[0–1] to uint8, then to BGR for OpenCV/YOLO
    # Optimized conversion with explicit dtype to reduce memory allocation
    bgr = cv2.cvtColor(np.clip(frame * 255, 0, 255).astype(np.uint8, copy=False), cv2.COLOR_RGBA2BGR)

    # Parse class filter - determine what to detect
    class_filter = None  # None means detect all classes
    if classes_str:  # If there's text in the Classes field
        # print(f'[DEBUG] Processing classes string: "{classes_str}"')  # Disabled debug
        # Convert class names to indices (YOLO class mapping)
        class_names = [name.strip() for name in classes_str.split(',') if name.strip()]
        # print(f'[DEBUG] Parsed class names: {class_names}')  # Disabled debug
        if class_names:
            # Get YOLO class names and find indices
            yolo_names = model.names  # Dict of {index: class_name}
            # print(f'[DEBUG] Available YOLO classes: {list(yolo_names.values())}')  # Disabled debug
            class_indices = []
            for class_name in class_names:
                for idx, yolo_name in yolo_names.items():
                    if yolo_name.lower() == class_name.lower():
                        class_indices.append(idx)
                        # print(f'[DEBUG] Found match: "{class_name}" -> index {idx}')  # Disabled debug
                        break
                else:
                    print(f'[YOLO] Warning: No match found for: "{class_name}"')  # Keep important warnings
            if class_indices:
                class_filter = class_indices
                # print(f'[YOLO] Detecting only: {class_names} -> indices: {class_indices}')  # Disabled debug
            else:
                print(f'[YOLO] Warning: No valid classes found for: {class_names}')
                print(f'[YOLO] Available classes: {list(yolo_names.values())[:10]}...') # Show first 10
    else:
        # print('[DEBUG] No classes string provided, detecting all objects')  # Disabled debug
        pass
    
    if class_filter is None:
        # print('[YOLO] Detecting all objects (no filter)')  # Disabled debug
        pass

    # Initialize with original image
    rendered = bgr
    
    # Skip detection if frame skipping is enabled for this frame
    if skip_detection:
        # print(f'[PERF] Skipping detection for frame {frame_counter} (frame_skip={frame_skip})')
        pass
    else:
        # Dynamic resolution based on detection density for performance optimization
        dynamic_imgsz = 640  # Default resolution
        if last_detection_count <= 2:  # Few objects = lower resolution for speed
            dynamic_imgsz = 416
        elif last_detection_count >= 8:  # Many objects = higher resolution for accuracy
            dynamic_imgsz = 832
        
        # Run YOLO detection with MPS optimization and appropriate filtering
        # print(f'[DEBUG] Running YOLO with class_filter: {class_filter}')  # Disabled debug
        with torch.no_grad():  # Disable gradient computation for inference speedup
            results = model.predict(
                source=bgr, 
                conf=confidence, 
                classes=class_filter, 
                verbose=False,
                device=device,  # Explicitly use optimal device
                half=True if device == 'mps' else False,  # Use half precision on MPS for speed
                imgsz=dynamic_imgsz  # Dynamic image size for performance optimization
            )
        
        det = results[0]
        current_detection_count = len(det.boxes)
        last_detection_count = current_detection_count  # Update for next frame
        # print(f'[YOLO] Found {current_detection_count} detections (imgsz: {dynamic_imgsz})')

        # Apply detection limit - sort by confidence and take top N
        if len(det.boxes) > 0 and detection_limit > 0:
            # Sort boxes by confidence (descending) and take top N
            confidences = det.boxes.conf.cpu().numpy()
            sorted_indices = np.argsort(confidences)[::-1]  # Sort descending
            
            # Limit to top N detections
            limit_indices = sorted_indices[:detection_limit]
            
            # Fix negative stride issue by making a copy
            limit_indices = limit_indices.copy()
            
            # Create new detection result with limited boxes
            det.boxes = det.boxes[limit_indices]
            # print(f'[YOLO] Limited to top {len(det.boxes)} detections (limit: {detection_limit})')
        
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
                report_table.appendRow(['Object_Type', 'Confidence', 'X_Center', 'Y_Center', 'Width', 'Height', 'ID'])
                
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
                        
                        # Calculate bounding box coordinates
                        x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]
                        x_center = (x1 + x2) / 2.0
                        y_center = (y1 + y2) / 2.0
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Increment counter for this object type
                        if class_name not in object_counters:
                            object_counters[class_name] = 0
                        object_counters[class_name] += 1
                        
                        # Add row: [object_name, confidence, x_center, y_center, width, height, id_within_type]
                        report_table.appendRow([
                            class_name, 
                            f'{confidence_val:.3f}', 
                            f'{x_center:.1f}', 
                            f'{y_center:.1f}', 
                            f'{width:.1f}', 
                            f'{height:.1f}', 
                            str(object_counters[class_name])
                        ])
                else:
                    # If no detections, add empty row
                    report_table.appendRow(['none', '0.000', '0.0', '0.0', '0.0', '0.0', '0'])
                    
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

        # Custom drawing logic with indexed labels
        if drawBox and len(det.boxes) > 0:
            # Create a copy of the image to draw on
            rendered = bgr.copy()
            
            # Re-use the object counting logic for unique IDs in labels
            label_counters = {}

            for box in det.boxes:
                # Get detection data
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence_val = float(box.conf[0])

                # Increment counter for this class to get a unique ID
                label_counters[class_name] = label_counters.get(class_name, 0) + 1
                obj_id = label_counters[class_name]

                # Create the custom label
                label = f'{class_name} {obj_id}: {confidence_val:.2f}'

                # --- Draw bounding box ---
                current_class_color = class_color_map.get(class_name, (255, 255, 255)) # Default to white if class not in map
                cv2.rectangle(rendered, (x1, y1), (x2, y2), current_class_color, 2)

                # --- Draw label background ---
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y1 = max(y1, label_size[1] + 10)
                cv2.rectangle(rendered, (x1, label_y1 - label_size[1] - 10), (x1 + label_size[0], label_y1 - base_line), current_class_color, cv2.FILLED)

                # --- Draw label text ---
                cv2.putText(rendered, label, (x1, label_y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text

    # Metal GPU memory cleanup every 30 frames to prevent fragmentation
    if device == 'mps' and frame_counter % 30 == 0:
        try:
            torch.mps.empty_cache()
            # print(f"[YOLO] Metal GPU memory cache cleared (frame {frame_counter})")
        except Exception as e:
            print(f"[YOLO] Warning: Could not clear Metal cache: {e}")
    
    # Performance monitoring and stats
    end_time = time.time()
    frame_time = end_time - start_time
    performance_stats['frame_count'] += 1
    performance_stats['avg_inference_time'] = (
        (performance_stats['avg_inference_time'] * (performance_stats['frame_count'] - 1) + frame_time) 
        / performance_stats['frame_count']
    )
    
    # Log performance stats every 100 frames
    if frame_counter % 100 == 0 and frame_counter > 0:
        avg_fps = 1.0 / performance_stats['avg_inference_time'] if performance_stats['avg_inference_time'] > 0 else 0
        print(f"[PERF] Frame {frame_counter}: Avg FPS: {avg_fps:.1f}, Avg inference: {performance_stats['avg_inference_time']*1000:.1f}ms")
    
    # Convert to RGBA for TouchDesigner and flip vertically for correct orientation
    # Optimized memory handling with explicit copy=False where safe
    rgba = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGBA)
    rgba = cv2.flip(rgba, 0)  # Vertical flip to fix YOLO text orientation

    # Final output with optimized array handling
    scriptOp.copyNumpyArray(rgba)  # Remove redundant .astype(np.uint8) as it's already uint8
    return
