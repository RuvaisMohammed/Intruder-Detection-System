import cv2
import face_recognition
import os
import winsound
import datetime

known_faces = []
known_face_labels = []
# Replace this with your dataset path
dataset_path = 'E:\intruder detection\known_faces'
unauthorized_path = 'E:\intruder detection'

# Create unauthorized folder if it doesn't exist
if not os.path.exists(unauthorized_path):
    os.makedirs(unauthorized_path)

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        face_img = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(face_img)[0]
        known_faces.append(face_encoding)
        known_face_labels.append(label)

video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

# Flags to control beep frequency and count
beep_cooldown = 0
beep_count = 0
MAX_BEEPS = 3

while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    intruder_detected = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        name = "Intruder"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_labels[first_match_index]
        else:
            intruder_detected = True
            # Save intruder image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            intruder_filename = f"intruder_{timestamp}.jpg"
            cv2.imwrite(os.path.join(unauthorized_path, intruder_filename), frame)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    if intruder_detected and beep_cooldown == 0 and beep_count < MAX_BEEPS:
        winsound.Beep(1000, 500)  # Frequency = 1000Hz, Duration = 500ms
        beep_cooldown = 30  # Set cooldown to avoid continuous beeping
        beep_count += 1

    if beep_cooldown > 0:
        beep_cooldown -= 1

    cv2.imshow('Intruder Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
