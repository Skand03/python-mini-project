import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from flask import Flask, Response, render_template, jsonify

# Define the paths and options for the gesture recognizer model
model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize a global variable to store the result
latest_result = None

# Callback function to handle results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result  # Update the latest result

# Set up the gesture recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Initialize Flask app
app = Flask(__name__)

# Open the webcam using OpenCV
cap = cv2.VideoCapture(0)

# Create a gesture recognizer instance
recognizer = GestureRecognizer.create_from_options(options)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def generate_frames():
    global latest_result
    last_display_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Convert the frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Get the current time in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # Perform gesture recognition asynchronously
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Get the current time
        current_time = time.time()

        # Check if 1 second has passed since the last display
        if current_time - last_display_time >= 1.0:
            if latest_result is not None:
                # Print the latest results
                gesture_info = []
                for i, gesture in enumerate(latest_result.gestures):
                    handedness = latest_result.handedness[i]
                    hand_info = {'handedness': handedness[0].category_name, 'gestures': []}
                    for g in gesture:
                        if g.category_name == "ILoveYou":
                            g.category_name = "Rock"
                        hand_info['gestures'].append({'gesture': g.category_name, 'score': g.score})
                    gesture_info.append(hand_info)
                latest_result_data = {'gestures': gesture_info}
                print(latest_result_data)
                # Update the last display time
                last_display_time = current_time

        # Convert the frame to RGB for MediaPipe Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hand landmarks
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture_data')
def gesture_data():
    global latest_result
    if latest_result is None:
        return jsonify({'gestures': []})
    
    gesture_info = []
    for i, gesture in enumerate(latest_result.gestures):
        handedness = latest_result.handedness[i]
        hand_info = {'handedness': handedness[0].category_name, 'gestures': []}
        for g in gesture:
            if g.category_name == "ILoveYou":
                g.category_name = "Rock"
            hand_info['gestures'].append({'gesture': g.category_name, 'score': g.score})
        gesture_info.append(hand_info)
    return jsonify({'gestures': gesture_info})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
