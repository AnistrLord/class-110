import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("keras_model.h5")
video = cv2.VideoCapture(2)

while True:
    check,frame = video.read()
    img = cv2.resize(frame,(224,224))
    test_image = np.array(img,dtype=np.float32)
    test_image = np.expand_dims(test_image,axis=0)
    normalised_image = test_image/255.0
    prediction = model.predict(normalised_image)
    print("With Mask : ",int(prediction[0][0] * 100.0)," %")
    cv2.imshow("Result",frame)
    key = cv2.waitKey(1) 
    if key == 32:
        print("Closing...")
        break

video.release()

        