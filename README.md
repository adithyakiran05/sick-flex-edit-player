# Sick Flex Edit Player

A Python-based computer vision project that uses your webcam to detect when you flex your biceps or show your abs, and automatically plays specific hype music (like "Outside" or "Cool for the Summer"). 

Powered by OpenCV, MediaPipe (Tasks API), and Pygame-CE.

## Features
- **Bicep Flex Detection**: Raises the hype by playing your bicep track when you flex.
- **Abs / Bare Torso Detection**: Plays your summer track when you take your shirt off or show your abs.
- **Smart Audio Playback**: Automatically pauses when you stop flexing and resumes exactly where you left off. Restarts the track if it's close to the end.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adithyakiran05/sick-flex-edit-player.git
   cd sick-flex-edit-player
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your music:**
   Place two audio files in the root of the project directory:
   - `cool_for_the_summer.mp3` (Plays when abs are detected)
   - `outside.mp3` (Plays when biceps are flexed)
   *(If you don't add these, the script will automatically generate dummy 'beep' .wav files so it won't crash).*

5. **Download the MediaPipe model:**
   You need the `pose_landmarker_lite.task` file. Download it from Google's official repository and place it in the project root:
   ```bash
   # Using PowerShell
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" -OutFile "pose_landmarker_lite.task"
   ```

## Usage

Run the main script:
```bash
python main.py
```
Stand in front of your webcam, flex those muscles, and enjoy! Press `q` to quit the application gracefully.
