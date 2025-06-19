import os
import cv2
import face_recognition

# === Step 1: Load Known Faces ===
known_dir = "known_faces"
known_encodings = []
known_names = []

for file in os.listdir(known_dir):
    img_path = os.path.join(known_dir, file)
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(file)[0])  # Use filename (e.g. "john.jpg" → "john")

# === Step 2: Load Test Set ===
test_dir = "test_faces"
total = 0
correct = 0

for file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, file)
    expected_name = os.path.splitext(file)[0].lower()  # filename should match known person

    img = face_recognition.load_image_file(img_path)
    locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, locations)

    if encodings:
        total += 1
        encoding = encodings[0]

        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        if True in matches:
            best_match_index = face_distances.argmin()
            predicted_name = known_names[best_match_index].lower()

            if predicted_name == expected_name:
                correct += 1
                print(f"[✔] {file}: Correctly matched with '{predicted_name}'")
            else:
                print(f"[✘] {file}: Mismatched with '{predicted_name}' (expected '{expected_name}')")
        else:
            print(f"[✘] {file}: No match found")
    else:
        print(f"[⚠] {file}: No face detected")

# === Step 3: Accuracy Summary ===
print("\n--- Evaluation Complete ---")
print(f"Total Test Images: {total}")
print(f"Correct Predictions: {correct}")
print(f"Accuracy: {correct / total * 100:.2f}%" if total > 0 else "No valid test images found.")
