# ONNX Detection System

DurskAI — an AI-assisted detection and aiming overlay for Windows.

## Features
- ONNX-based object detection
- Fixed 320×320 capture region centered on screen
- Real mouse control via SendInput, or virtual gamepad via ViGEmBus
- Floating overlay showing what the AI sees
- Self-exclusion so it doesn't lock onto your own character

## Requirements
- Windows 10 or 11 (64-bit)
- Python 3.11 or 3.12
- A `.onnx` model in the `models/` folder

## Setup
1. Double-click `install.bat` to install dependencies
2. Place your `.onnx` model in `models/`
3. Double-click `run.bat` to launch

## Hotkeys
- **F1** — start
- **F2** — stop
