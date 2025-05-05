# 3D Model Generator

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python voice_to_3d.py
   ```

## Usage
- Speak your prompt when asked.
- The app will generate a 3D model (GIF, PLY, OBJ) from your spoken description.
- Output files are saved in the `generated_models/` directory.

## Dependencies
- torch
- shap_e
- sounddevice
- speechrecognition
- imageio
- numpy

## Notes
- Requires a working microphone.
- For best results, use a machine with a CUDA-enabled GPU.
