import asyncio
import datetime
import os
import cv2
import csv
from deepface import DeepFace
from frame_sdk import Frame
from frame_sdk.display import Alignment
from gtts import gTTS

# Path to pre-trained Haar Cascade for face detection (to detect the person the child is looking at)
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Initialize Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)

# CSV File to track progress
progress_file = "progress_log.csv"

# Create CSV file and write header if it doesn't exist
if not os.path.exists(progress_file):
    with open(progress_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Object", "Emotion", "Focus Time (seconds)", "Reminder"])

# Capture image from the Frame glasses' camera
async def capture_image(frame):
    # Capture image from Frame glasses' camera
    await frame.camera.save_photo("captured_image.jpg")
    print("Image captured.")

# Detect if a face (person) is in the captured image
def detect_face(photo_filename):
    image = cv2.imread(photo_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        return True  # A person (face) is detected
    return False  # No person detected

# Recognize the emotion of the detected person (if a person is detected)
def recognize_emotion(photo_filename):
    result = DeepFace.analyze(photo_filename, actions=['emotion'], enforce_detection=False)
    if len(result) > 0:
        emotion = result[0]['dominant_emotion']  # Get emotion from the first detected face
        return emotion
    return 'neutral'  # Default emotion if none is detected

# Track the child's focus time and log it in a CSV file
def log_focus_time(object_name, emotion, focus_time, reminder):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(progress_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, object_name, emotion, focus_time, reminder])

# Function to convert text to speech using gTTS
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("afplay output.mp3")  # Use afplay on macOS to play the audio

# Main function
async def main():
    async with Frame() as frame:
        print("Connected to Frame.")
        await frame.display.show_text("Starting Joint Attention...", align=Alignment.MIDDLE_CENTER)

        photo_count = 0
        focus_time = {'Toy': 0, 'Person': 0}  # Track focus time for each object
        focus_threshold = 5  # Threshold (in seconds) for prompting a shift in focus (set to 5 seconds for testing)
        last_focused_object = None
        focus_stability_timer = 0  # Timer to track how long the child is focusing on the same object
        focus_time_threshold = 5  # Threshold for reminding to shift focus after 5 seconds
        reminder_displayed = False  # Flag to prevent repeating the reminder message

        while True:
            # Capture the photo
            await capture_image(frame)

            # Detect if the child is looking at a person
            is_looking_at_person = detect_face("captured_image.jpg")
            detected_object = 'Person' if is_looking_at_person else 'Toy'

            # If the child is looking at a person, perform emotion recognition and display the emotion on lenses
            emotion = None
            if is_looking_at_person:
                # Show "Great job looking at the person!" first
                await frame.display.show_text("Great job, you're looking at the person!", align=Alignment.MIDDLE_CENTER)
                speak("Great job, you're looking at the person!")

                # Wait briefly and then show the emotion
                await asyncio.sleep(2)  # Wait for 2 seconds before showing emotion

                emotion = recognize_emotion("captured_image.jpg")
                print(f"Emotion detected: {emotion}")

                # Display the emotion centered on the Frame glasses
                await frame.display.show_text(f"Emotion: {emotion}", align=Alignment.MIDDLE_CENTER)

            # Track last focused object
            if detected_object != last_focused_object:
                last_focused_object = detected_object
                focus_stability_timer = 0  # Reset timer if the object changes
                reminder_displayed = False  # Reset reminder display flag

            # If focus timer exceeds reminder threshold (5 seconds), prompt to look at the last object
            if focus_stability_timer >= focus_time_threshold and not reminder_displayed:
                reminder = f"Look at the {last_focused_object} now!"
                await frame.display.show_text(reminder, align=Alignment.MIDDLE_CENTER)
                speak(reminder)
                log_focus_time(last_focused_object, emotion, focus_time[last_focused_object], reminder)
                reminder_displayed = True  # Mark reminder as displayed
                focus_stability_timer = 0  # Reset the timer after prompting

            # Update focus time based on detected object
            focus_time[detected_object] += 1  # Increment focus time for the detected object

            # Log the focus time and emotion to CSV every 5 seconds
            if focus_time[detected_object] % 5 == 0:
                log_focus_time(detected_object, emotion, focus_time[detected_object], "")

            # Provide feedback based on detected object (toy or person)
            if detected_object == 'Person':
                await frame.display.show_text("Good job looking at the person!", align=Alignment.MIDDLE_CENTER)
                speak("Good job looking at the person!")
            elif detected_object == 'Toy':
                await frame.display.show_text("Great job looking at the toy!", align=Alignment.MIDDLE_CENTER)
                speak("Great job looking at the toy!")

            # If the child focuses on the same object for too long (5 seconds), prompt them to shift focus
            if focus_time[detected_object] >= focus_time_threshold and not reminder_displayed:
                if detected_object == 'Toy':
                    reminder = "Now look at the person!"
                    await frame.display.show_text(reminder, align=Alignment.MIDDLE_CENTER)
                    speak(reminder)
                    log_focus_time(detected_object, emotion, focus_time[detected_object], reminder)
                elif detected_object == 'Person':
                    reminder = "Now look at the toy!"
                    await frame.display.show_text(reminder, align=Alignment.MIDDLE_CENTER)
                    speak(reminder)
                    log_focus_time(detected_object, emotion, focus_time[detected_object], reminder)
                reminder_displayed = True  # Mark reminder as displayed
                focus_time[detected_object] = 0  # Reset the focus time after the reminder

            # Wait for 1 second before capturing the next image
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
