import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask import Flask, render_template, Response
import tensorflow as tf
import math

app = Flask(__name__)


def generate_frames():

    url = "http://192.168.0.102:8080/video"
    cap = cv2.VideoCapture(0)

    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model_new.h5", "Model/labels_new.txt")

    offset = 20
    imgSize = 300

    folder = "Data/C"
    counter = 0

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
            "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    while True:
        success, img = cap.read()
        frame = cv2.resize(img, (200, 200))
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (255, 0, 255), 4)

            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)

    # cv2.imshow("Image", imgOutput)
    # cv2.waitKey(1)

    # translate()
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        imgOutput = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')


@app.route('/')
def index():
    return render_template('index1.html')


# @app.route('/sign_lang')
# def translate():
#     return render_template('sign_lang.html')

# Stream video

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Driver code
if __name__ == "__main__":
    app.run(debug=True)
