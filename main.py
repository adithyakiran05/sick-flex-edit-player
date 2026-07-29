import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pygame
import os
import math
import time
from scipy.io import wavfile

# --- Audio Initialization ---
pygame.mixer.init()

# Generate dummy wav files if the audio files don't exist
def generate_dummy_audio(filename, frequency, duration=2.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a sine wave
    audio = np.sin(frequency * t * 2 * np.pi)
    # Fade in and out
    fade = min(int(sample_rate * 0.1), len(audio)//2)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    # Convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio)

cool_song_path = 'cool_for_the_summer.mp3'
outside_song_path = 'outside.mp3'

if not os.path.exists(cool_song_path):
    print(f"Generating dummy {cool_song_path}")
    generate_dummy_audio(cool_song_path, 440.0) # A4 note for abs

if not os.path.exists(outside_song_path):
    print(f"Generating dummy {outside_song_path}")
    generate_dummy_audio(outside_song_path, 880.0) # A5 note for biceps

try:
    len_cool = pygame.mixer.Sound(cool_song_path).get_length()
except:
    len_cool = 1000.0 # fallback
try:
    len_outside = pygame.mixer.Sound(outside_song_path).get_length()
except:
    len_outside = 1000.0 # fallback

pos_cool = 0.0
pos_outside = 0.0

# --- MediaPipe Initialization ---
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO)
landmarker = vision.PoseLandmarker.create_from_options(options)

# Helper constants for landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

CONFIDENCE_THRESHOLD = 0.7

def is_confident(landmarks, indices):
    """Checks if all specified landmarks meet the confidence threshold."""
    for idx in indices:
        lm = landmarks[idx]
        # In Tasks API, visibility and presence are floats between 0.0 and 1.0
        vis = lm.visibility if lm.visibility is not None else 1.0
        pres = lm.presence if lm.presence is not None else 1.0
        if vis < CONFIDENCE_THRESHOLD or pres < CONFIDENCE_THRESHOLD:
            return False
    return True

# --- Helper Functions ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_torso_box(landmarks, image_w, image_h):
    """Extracts a bounding box for the torso."""
    # Landmarks for shoulders and hips
    l_shoulder = landmarks[LEFT_SHOULDER]
    r_shoulder = landmarks[RIGHT_SHOULDER]
    l_hip = landmarks[LEFT_HIP]
    r_hip = landmarks[RIGHT_HIP]
    
    x_coords = [l_shoulder.x, r_shoulder.x, l_hip.x, r_hip.x]
    y_coords = [l_shoulder.y, r_shoulder.y, l_hip.y, r_hip.y]
    
    # Check visibility
    if not is_confident(landmarks, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
        return None
        
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add a little padding, but not too much to avoid background
    padding = 0.05
    x_min = max(0, x_min - padding)
    x_max = min(1, x_max + padding)
    y_min = max(0, y_min)
    y_max = min(1, y_max)
    
    return [int(x_min * image_w), int(y_min * image_h), int(x_max * image_w), int(y_max * image_h)]

def check_abs_exposed(image, torso_box):
    """Checks if a significant portion of the torso is skin colored."""
    if torso_box is None:
        return False
        
    x1, y1, x2, y2 = torso_box
    if x2 <= x1 or y2 <= y1:
        return False
        
    torso_img = image[y1:y2, x1:x2]
    if torso_img.size == 0:
        return False
        
    # Convert to HSV for color segmentation
    hsv_torso = cv2.cvtColor(torso_img, cv2.COLOR_BGR2HSV)
    
    # Define a general skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv_torso, lower_skin, upper_skin)
    
    # Calculate percentage of skin pixels
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = (x2 - x1) * (y2 - y1)
    
    if total_pixels == 0:
        return False
        
    skin_percentage = skin_pixels / total_pixels
    
    # Draw rectangle for visualization
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"Skin: {skin_percentage:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Threshold for considering it "abs/bare torso" (40% skin)
    return skin_percentage > 0.4

current_state = None  # Can be 'bicep', 'abs', or None

# --- Main Loop ---
cap = cv2.VideoCapture(0)

print("Starting webcam feed... Press 'q' to quit.")
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image_h, image_w, _ = frame.shape
    
    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
    # Make detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)
    
    try:
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    except Exception as e:
        print(f"Detection error: {e}")
        continue
    
    # Recolor back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    new_state = None
    
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        landmarks = results.pose_landmarks[0]
        
        # 1. Bicep Detection
        # Left arm
        l_shoulder = [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y]
        l_elbow = [landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y]
        l_wrist = [landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y]
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        l_confident = is_confident(landmarks, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST])
        
        # Right arm
        r_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y]
        r_elbow = [landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y]
        r_wrist = [landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y]
        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        r_confident = is_confident(landmarks, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST])
        
        # If elbow is raised (y coord smaller than shoulder) and angle is tight
        # We check if wrist is above elbow to indicate a flex
        l_flexing = l_confident and l_angle < 70 and (landmarks[LEFT_WRIST].y < landmarks[LEFT_ELBOW].y)
        r_flexing = r_confident and r_angle < 70 and (landmarks[RIGHT_WRIST].y < landmarks[RIGHT_ELBOW].y)
        
        is_flexing = l_flexing or r_flexing
        
        # 2. Abs Detection
        torso_box = get_torso_box(landmarks, image_w, image_h)
        is_abs = check_abs_exposed(image, torso_box)
        
        # Determine state priority
        if is_flexing:
            new_state = 'bicep'
            cv2.putText(image, "BICEP FLEX DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_abs:
            new_state = 'abs'
            cv2.putText(image, "ABS/BARE TORSO DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Render detections (Draw small circles on joints if confident)
        for lm_idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP]:
            if is_confident(landmarks, [lm_idx]):
                lm = landmarks[lm_idx]
                cv2.circle(image, (int(lm.x * image_w), int(lm.y * image_h)), 5, (0, 255, 255), -1)
        
    # Audio State Machine
    if new_state != current_state:
        # Debounce/smooth state transitions to avoid flickering
        # We only stop or start if the state actually changed
        if current_state is not None:
            time_played = pygame.mixer.music.get_pos() / 1000.0
            if time_played > 0:
                if current_state == 'bicep':
                    pos_outside += time_played
                elif current_state == 'abs':
                    pos_cool += time_played
                    
        # Stop currently playing
        pygame.mixer.music.stop()
        
        if new_state == 'bicep':
            # Check if close to end (e.g., less than 5 seconds left)
            if len_outside - pos_outside < 5.0:
                pos_outside = 0.0
            pygame.mixer.music.load(outside_song_path)
            try:
                pygame.mixer.music.play(start=pos_outside)
            except pygame.error:
                pygame.mixer.music.play() # fallback if start not supported
            print(f"Playing 'Outside' for Biceps! Resuming from {pos_outside:.1f}s")
        elif new_state == 'abs':
            if len_cool - pos_cool < 5.0:
                pos_cool = 0.0
            pygame.mixer.music.load(cool_song_path)
            try:
                pygame.mixer.music.play(start=pos_cool)
            except pygame.error:
                pygame.mixer.music.play()
            print(f"Playing 'Cool for the Summer' for Abs! Resuming from {pos_cool:.1f}s")
        else: # new_state is None
            print("Stopping music.")
            
        current_state = new_state
        
    cv2.imshow('Gym Flex Tracker', image)
    
    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
landmarker.close()
