# Import OpenCV module
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to detect face
def detect_face(img):
    # Convert the test image to a gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load OpenCV face detector
    face_cas = cv2.CascadeClassifier('openCV/haarcascade_profileface.xml')
    faces = face_cas.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # If no faces are detected, then return None
    if len(faces) == 0:
        return None, None

    # Extract the face
    x, y, w, h = faces[0]
    # Return only the face part
    return gray[y: y+w, x: x+h], faces[0]

# This function will read all persons' training images, detect face
# from each image and will return two lists of exactly the same size
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels

# Let's first prepare our training data
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# Create LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Train the face recognizer on our training faces
face_recognizer.train(faces, np.array(labels))

# Function to draw rectangle on the image
def draw_rectangle(img, rect):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to draw text on the image
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# This function recognizes the person in the image passed
# and draws a rectangle around the detected face with the name of the subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = f"Person {label} ({confidence:.2f}%)"
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img
# Load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
print("type:")
type(test_img1)
type(test_img2)

# Perform predictions
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)

print("Prediction complete")

# Create a figure with 2 plots (one for each test image)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Display test image1 result
ax1.imshow(cv2.cvtColor(predicted_img1, cv2.COLOR_BGR2RGB))
# Display test image2 result
ax2.imshow(cv2.cvtColor(predicted_img2, cv2.COLOR_BGR2RGB))

# Display both images
#display both images
cv2.imshow("Tom cruise test", predicted_img1)
cv2.imshow("Shahrukh Khan test", predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
