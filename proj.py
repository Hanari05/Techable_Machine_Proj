import cv2
import numpy as np
from tensorflow.keras.models import load_model

np.set_printoptions(suppress=True)

# Load model
model = load_model("keras_model.h5", compile=False)

# Load labels
class_names = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()

    if not ret:
        print("Không mở được camera")
        break

    original_image = image.copy()

    # Resize cho model
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize
    image = np.asarray(image, dtype=np.float32)
    image = (image / 127.5) - 1
    image = np.reshape(image, (1, 224, 224, 3))

    # Predict
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = prediction[0][index]

    # Hiển thị
    print(f"Class: {class_name} | Confidence: {confidence*100:.2f}%")

    cv2.imshow("Webcam", original_image)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()