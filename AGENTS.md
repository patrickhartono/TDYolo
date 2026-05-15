<claude-mem-context>
# Memory Context

# [TDYolo] recent context, 2026-05-15 9:31am GMT+7

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (17,477t read) | 359,845t work | 95% savings

### May 14, 2026
S252 User confirmed cleanup is done and is proceeding to save the project manually (May 14 at 5:36 PM)
S253 Implement YOLO detection label overlay in TouchDesigner showing "person 2: 0.9" format — class name, per-class track ID, and confidence score displayed as colored badges above each bounding box (May 14 at 5:38 PM)
1214 5:49p 🔵 TDYolo_v2 outTOP Wiring Map Confirmed
1215 5:50p 🟣 Label Overlay Network Fully Wired and Configured Inside TDYolo_v2
1216 " 🔵 label_updater CHOP Reference Resolves to None with Relative Path
1217 " 🔴 label_updater CHOP Reference Fixed: Absolute Path Required
1218 " 🟣 label_updater Script Written and Loaded into chopexecuteDAT
1219 5:51p 🟣 label_updater Working: Labels Populating Correctly with Class, ID, and Confidence
1220 " 🟣 External Label Composite Nodes Created at /project1 Level
1221 " 🔵 TDYolo_v2 outputConnector Map: out8 (Labels) is Index 5
1222 " 🟣 Full Label Overlay Pipeline Wired End-to-End at /project1
1223 " 🔴 Critical Bug: textTOP bgalpha Fills Entire 640×640 Canvas, Not Just Text Badge Area
1224 5:52p 🔴 bgalpha=0 Fix: Only 2104 Text Glyph Pixels Non-Transparent on 640×640 Canvas
1225 " ⚖️ Architecture Pivot: Per-Slot transformTOP Added Despite Original Plan Prohibition
1226 " 🔴 labels_merge Wiring Corrupted: Only Odd-Numbered Label Slots Connected
1227 5:53p 🔴 labels_merge Rewired: All 12 transform_label_N Slots Connected Correctly
1228 " 🟣 Label Updater v2 Working: Correct "person 1: 0.89" Format with Per-Slot Transform Positioning
1229 5:54p 🔵 transform_label_N Parameter Confirmation: tunit=fraction, outputresolution=custom 640×640
1230 " 🔴 fillmode=nativeres on transformTOP Ineffective: textTOP Still 640×640 with bgalpha=1
1231 " 🔴 transform_label Tiling Parameters Causing Badge to Repeat Across Canvas
1232 " 🔴 Badge Transparency Fixed: fillmode=fill + Explicit sx/sy Ratios Produces Correct 160×24 Badge on Transparent Canvas
1233 5:55p 🟣 Label Overlay Position Calculation Verified: tx/ty Match Expected Values Exactly
1234 " 🟣 Badge Position Accuracy Confirmed: Sub-Pixel Alignment to Expected Canvas Coordinates
S254 Fix TouchDesigner YOLO detection overlay: label badges were rendering as large solid colored rectangular blocks instead of tight-fitting badges around text (May 14 at 5:55 PM)
1235 6:03p 🔵 Text Color Change Producing Block Box Visual Artifact
1236 6:05p 🔵 Visual Confirmation of Block Box CSS Bug in TDYolo Project
1237 6:06p ⚖️ Plan Updated: Dynamic Badge Width via textTOP Canvas Resize in TDYolo
S255 Fix invisible text on label_N Text TOP operators in TDYolo_v2 TouchDesigner component (May 14 at 6:08 PM)
1238 6:10p 🔵 Text Visibility Bug: Background Alpha Hiding Text Color
1239 " 🔵 label_0 Text Invisible: White Font Swallowed by Full-Red Background in TouchDesigner
1240 " 🔴 label_0 Text Now Visible After Resetting positionx/positiony to 0
1241 6:11p ✅ Attempted Safeguard Patch to label_updater to Always Reset Text Position
S256 Fix invisible text labels on TDYolo_v2 component, then resize badges to be less obtrusive (May 14 at 6:11 PM)
1242 6:13p ✅ Label Badge Size Reduced: Font 16→11, BADGE_H 24→16, CHAR_W 8→6
S257 Fix invisible and upside-down text labels in TDYolo_v2 TouchDesigner YOLO overlay component, then resize and reposition badges correctly (May 14 at 6:13 PM)
1243 6:15p 🔵 TDYolo_v2 Pipeline Topology: All Transforms Rotated 180° — Consistent Flip Architecture
1244 6:16p 🔴 Upside-Down Text Fixed: Removed 180° Rotation from transform_labels, Rewrote Position Math
S258 TDYolo_v2 TouchDesigner YOLO detection pipeline — commit session changes including bbox-only output, label overlay system, and compositor refactor (May 14 at 6:17 PM)
1245 6:20p 🔵 User Working in TouchDesigner Project with MCP Connected
1246 " 🔵 TDYolo Repository State: Branch Ahead, Many Untracked Backup Snapshots
1247 " 🟣 TDYolo_v2: bbox-only Output, Label Overlay System, Retired Internal overlay_shader
S259 TDYolo_v2 session wrap-up — verifying claude-mem captured all observations before closing session (May 14 at 6:20 PM)
S270 TDYolo v2-Fix.15 — Finalise and release session started, MCP confirmed live (May 14 at 6:22 PM)
### May 15, 2026
1466 8:12a 🔵 TouchDesigner YOLO Project — Release Session Started
1467 " 🔵 TDYolo Project File Structure Confirmed
1468 " 🔵 TouchDesigner MCP Environment Versions Confirmed
1469 8:15a 🔵 MCP Node Inspection and Modification Tools Loaded; Current State Screenshot Reviewed
1470 " 🔵 TDYolo_v2 Label Overlay Node Architecture Mapped
1471 8:16a 🔵 Full TDYolo_v2 Network Map: 85 Nodes Across Detection, Overlay, and Web Server Subsystems
1472 " 🔵 label_updater Script: Full Badge Positioning Logic and Color Palette System
1473 " 🔵 Two Confirmed Bugs Found in Label Overlay: Text Clipping and White-on-White Invisibility
1474 8:17a ⚖️ Fix Plan Written: CHAR_W 5→7 + Black Font for Label Badge Bugs
1475 8:18a ✅ Fix Plan Approved and Execution Mode Entered — Tasks Being Created
1476 8:19a 🔴 label_updater Script Updated: CHAR_W=7 and Black Font Color Applied
1477 " 🔴 Black Font Color Applied to All 12 label_N textTOP Nodes
1478 " 🔴 All Three Fix Tasks Completed — MCP Readback Verified, No Script Errors
S271 TDYolo v2-Fix.15 release day — fix label badge text clipping and white-on-white invisibility bugs (May 15 at 8:20 AM)
1479 9:24a 🔵 Label Position Misalignment with Bounding Box Overlay
1480 9:25a 🔵 Root Cause Found: BBox Rotated 180° But Labels Are Not
1481 9:26a 🔵 Visual Screenshots Examined to Confirm Label-BBox Misalignment

Access 360k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>