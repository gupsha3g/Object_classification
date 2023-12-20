from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pyttsx3

# Load the model outside the loop
model = load_model("custom_model.h5")

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 200)
cap.set(4, 200)

while True:
    _, frame = cap.read()

    # Resize the frame to match the input size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))

    # Expand the dimensions to create a batch of size 1
    expanded_img_array = np.expand_dims(resized_frame, axis=0)

    # Preprocess the image
    preprocessed_img = expanded_img_array / 255.0

    # prediction using model
    prediction = model.predict(preprocessed_img)
    clas = np.argmax(prediction)

    labels = ['deer', 'human', 'others']
    label = labels[clas]

    print(label)
    engine.say(label)
    engine.runAndWait()

    cv2.putText(frame, label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    print(clas)
    cv2.imshow('output', frame)

    # Add a delay after speech synthesis to allow it to finish
    cv2.waitKey(100)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
