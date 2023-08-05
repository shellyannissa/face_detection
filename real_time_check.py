import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

facetracker = load_model('facetracker.h5')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame  = cap.read()
    frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    y_hat = facetracker.predict(np.expand_dims(resized/255.0, axis = 0))
    sample_coords = y_hat[1][0]

    if y_hat[0] > 0.5:
        cv2.rectangle(frame,
        tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
        tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
        (255, 0 , 0), 2
        )
    cv2.rectangle(frame,
        tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
        [0, -30])),
        tuple(np.add(np.multiply(sample_coords[2:], [450, 450]).astype(int),
        [80, 0])),
        (255, 0 , 0), -1
    )
    cv2.imshow('EyeTrack',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()