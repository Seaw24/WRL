# WRL Gaze Data Collection & Sync Pipeline

This project provides tools for collecting and syncing eye gaze data using Pupil Neon and camera frames. It supports both real-time calibration and headless data collection modes.

## Files Overview

### 1. `collect_gaze_data.py`
**Purpose**: Collects gaze data from Pupil Neon camera with synchronized video recording.

**Features**:
- Two operating modes:
  - **FITTING MODE (default)**: Interactive calibration UI if no calibration file exists
  - **RUN MODE (headless)**: Uses pre-calibrated ROI for immediate data collection
- Records both gaze points and eye video frames
- Fixed ROI cropping for consistent output

**Usage**:
```bash
python collect_gaze_data.py
```

### 2. `sync_offline.py`
**Purpose**: Offline syncing of camera frames with Pupil Neon gaze labels.

**Features**:
- Pairs camera frame timestamps with Neon gaze timestamps
- Supports both original and postprocessed (rotated/flipped) eye videos
- Writes synchronized labels with frame-to-gaze mappings

**Usage**:
```bash
# Default: sync all subjects
python sync_offline.py

# Sync postprocessed videos
python sync_offline.py --postprocessed-root postprocessed_eye_videos

# Sync specific subject
python sync_offline.py --video postprocessed_eye_videos/joey/joey_final_96x96_dataset_rotated_cw45_flip_h.mp4 --source-dir joey
```

### 3. `eye_roi_calibration.json`
**Purpose**: Stores calibrated eye region-of-interest (ROI) settings.

**Format**:
```json
{
  "center_x": <x-coordinate>,
  "center_y": <y-coordinate>,
  "source_size": <crop_box_size>
}
```

**Note**: This file is auto-generated during calibration in `collect_gaze_data.py`. Share this file with team members to use pre-calibrated settings.

### 4. `requirements.txt`
**Purpose**: Python package dependencies.

**Packages**:
- `opencv-python` - Video processing
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `pupil-labs-realtime-api` - Pupil Neon communication

**Installation**:
```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect gaze data (includes calibration if needed)
python collect_gaze_data.py

# 3. Sync collected data offline
python sync_offline.py
```

## Important Notes

### ⚠️ Neon IP Address Configuration
The `collect_gaze_data.py` script connects to the Pupil Neon device via IP. **The Neon IP changes frequently and must be checked before each data collection session.**

**How to find your Neon IP**:
1. Open the Neon Companion App on the phone connected to your Neon glasses
2. On the main screen, tap the button in the **top-right corner** (phone icon)
3. Find the IP address displayed in the device settings
4. Update the IP in `collect_gaze_data.py` before running

### ⚠️ Calibration Required
**Always run calibration before collecting data.** The script will automatically open the calibration UI if no `eye_roi_calibration.json` file exists.

**Calibration Controls**:
- **Left click**: Set eye center
- **W/A/S/D**: Nudge center by 2px
- **+/-**: Grow/shrink crop box
- **ENTER**: Confirm and save
- **ESC**: Cancel

**During calibration**:
1. Roll your eyes through the full gaze range (up, down, left, right, diagonals)
2. Ensure the pupil stays inside the green box at every extreme
3. If pupil exits the box, press `+` to grow the crop box
4. Press ENTER to save calibration

## File Structure

```
.
├── collect_gaze_data.py        # Main data collection script
├── sync_offline.py              # Offline sync tool
├── eye_roi_calibration.json    # Calibration settings (auto-generated)
├── requirements.txt             # Dependencies
└── [subject_folders]/           # Collected data organized by subject
    ├── camera_frames.csv
    ├── neon_gaze_raw.csv
    └── eye_video.mp4
```

## Troubleshooting

- **Cannot connect to Neon**: Verify IP address in Neon Companion App (see Notes above)
- **Calibration shows black screen**: Ensure Neon camera is properly connected
- **Sync failures**: Verify `camera_frames.csv` and `neon_gaze_raw.csv` exist in subject folder
