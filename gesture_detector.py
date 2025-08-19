#!/usr/bin/python3

import cv2
import mediapipe as mp
import serial
import time
import subprocess
import logging

# Set up logging
logging.basicConfig(filename='/home/ubuntu/Desktop/Project/error_log.txt', 
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def lcd_print(text):
    try:
        # Prepare the command for the subprocess
        command = ['sudo', '-S', 'python3', '/home/ubuntu/Desktop/Project/LCD.py', text]
        
        # Run the subprocess
        subprocess.run(command, input='1234\n', text=True)
    except Exception as e:
        logging.error(f"Error in lcd_print: {e}")

lcd_print("Started .")

def speak_text(text):
    """Speak the given text using espeak."""
    try:
        subprocess.run(["espeak", text])
    except Exception as e:
        logging.error(f"Error in speak_text: {e}")

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

# Gesture Dictionary with Actions
gestures = {
    (1, 1, 1, 1, 1): {"name": "Open Palm", "action": "Open Palm"},
    (0, 0, 0, 0, 0): {"name": "Closed Fist", "action": "Closed Fist"},
    (1, 0, 0, 0, 0): {"name": "Thumbs Up", "action": "Thumbs Up"},
    (0, 1, 0, 0, 0): {"name": "Pointing", "action": "Pointing"},
    (0, 1, 1, 0, 0): {"name": "Peace Sign", "action": "Peace Sign"}
}

# Serial Communication Setup
try:
    arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
except serial.SerialException as e:
    logging.error(f"Could not open serial port: {e}")
    arduino = None

def send_to_arduino(message):
    """Send message to Arduino via Serial."""
    if arduino:
        try:
            arduino.write((message + '\n').encode('utf-8'))
        except Exception as e:
            logging.error(f"Error sending to Arduino: {e}")

# Gesture Detection and Tracking
class GestureTracker:
    def __init__(self):
        self.current_state = None
        self.detection_threshold = 5  # Number of consistent frames to trigger action
        self.frame_counter = 0

    def detect_gesture(self, lmList):
        """Detect hand gesture from landmark list."""
        if len(lmList) != 21:
            # Reset if no proper hand landmarks
            self.current_state = None
            self.frame_counter = 0
            return None

        fingersY = []
        
        # Thumb detection (x-coordinate)
        fingersY.append(1 if lmList[4][1] < lmList[5][1] else 0)
        
        # Other fingers (y-coordinate)
        for tip_id in [8, 12, 16, 20]:
            fingersY.append(1 if lmList[tip_id][2] < lmList[tip_id - 2][2] else 0)
        
        return tuple(fingersY)

    def process_gesture(self, gesture_key):
        """
        Process detected gesture with state machine approach.
        Only trigger action when gesture is consistent for multiple frames.
        """
        # If no change in gesture, increment counter
        if gesture_key == self.current_state:
            self.frame_counter += 1
        else:
            # Reset counter for new gesture
            self.current_state = gesture_key
            self.frame_counter = 1

        # Trigger action only once when threshold is reached
        if (self.frame_counter == self.detection_threshold and 
            gesture_key in gestures):
            
            gesture_info = gestures[gesture_key]
            print(f"Gesture: {gesture_info['name']}")
            print(f"Action: {gesture_info['action']}")
            
            try:
                lcd_print(gesture_info['action'])
                speak_text(gesture_info['action'])
                send_to_arduino(gesture_info['action'])
            except Exception as e:
                logging.error(f"Error in gesture action processing: {e}")
            
            # Prevent repeated triggers
            self.frame_counter = self.detection_threshold + 1
            
            return True
        return False

def main():
    try:
        # Start Webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        gesture_tracker = GestureTracker()

        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame. Exiting...")
                break
            
            # Flip and convert frame
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            result = hands.process(frameRGB)
            
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    
                    # Draw hand landmarks (if needed, but this is optional and can be skipped for the GUI-less version)
                    # mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                    
                    # Detect and process gesture
                    gesture_key = gesture_tracker.detect_gesture(lmList)
                    if gesture_key:
                        gesture_tracker.process_gesture(gesture_key)
            else:
                # Reset tracker if no hand is detected
                gesture_tracker.current_state = None
                gesture_tracker.frame_counter = 0

    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        # Release resources
        if cap.isOpened():
            cap.release()
        if arduino:
            arduino.close()

if __name__ == "__main__":
    main()

