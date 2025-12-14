import os
import cv2
import numpy as np

from utils import get_face_landmarks

# 👉 CHANGE THIS if your images are in test
data_dir = "./data/test"   # use "./data/train" if train images exist

output = []

print("Train/Test dir exists:", os.path.exists(data_dir))
print("Folder content:", os.listdir(data_dir))

for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_path = os.path.join(data_dir, emotion)
    print("\nChecking emotion folder:", emotion_path)

    if not os.path.isdir(emotion_path):
        print("❌ Not a directory, skipping")
        continue

    images = os.listdir(emotion_path)
    print("Images found:", len(images))

    for img_name in images:
        img_path = os.path.join(emotion_path, img_name)
        print("Reading image:", img_path)

        image = cv2.imread(img_path)
        if image is None:
            print("❌ Failed to load image")
            continue

        face_landmarks = get_face_landmarks(image)
        if face_landmarks is None:
            print("❌ No face detected")
            continue

        print("Landmark length:", len(face_landmarks))

        # 🔴 IMPORTANT: refined landmarks = 1434
        if len(face_landmarks) == 1434:
            row = face_landmarks.tolist()
            row.append(emotion_indx)
            output.append(row)
            print("✅ Saved sample")

print("\nTOTAL samples saved:", len(output))

# 🔹 SAVE FILE
if len(output) > 0:
    output = np.array(output)
    save_path = os.path.join(os.getcwd(), "data.txt")
    np.savetxt(save_path, output, fmt="%.6f")
    print("🎉 data.txt saved at:", save_path)
else:
    print("⚠️ No data saved — check folders and images")
