from flask import Flask, render_template, request
from matplotlib.pyplot import text
from sqlalchemy import null
import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np
import os

app = Flask(__name__)

model = models.load_model('model_opt.h5')

CATEGORIES = ["Hitesh", "Labhesh", "Tarun", "Varkha"]


def check_fingerprint(img):
    for file in [file for file in os.listdir("SOCOFing/Altered/Altered-Hard")]:
        fingerprint_database_image = cv2.imread(
            "SOCOFing/Altered/Altered-Hard/" + file)
        #print("SOCOFing/Altered/Altered-Hard/" + file)
        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(
            fingerprint_database_image, None)

        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),
                                        dict()).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []

        for p, q in matches:
            if p.distance < 0.1*q.distance:
                match_points.append(p)

        keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        if (len(match_points) / keypoints) > 0.1:
            print("% match: ", len(match_points) / keypoints * 100)
            a = "matching percentage: " + \
                str(len(match_points) / keypoints * 100)
            print("Figerprint ID: " + str(file))
            result = cv2.drawMatches(img, keypoints_1, fingerprint_database_image,
                                     keypoints_2, match_points, None)
            result = cv2.resize(result, None, fx=4, fy=4)
            cv2.imwrite('static/result_matching.jpg', result)
            return [True, a]
        else:
            print("no match")
            print("score: " + str(len(match_points) / keypoints * 100))
            print("\n")
            return [False]


def check_ear(img):
    img = cv2.resize(img, (60, 100))
    img = np.array(img).reshape(-1, 60, 100, 1)
    output = model.predict([img])
    output = tf.math.argmax(output, axis=1).numpy()
    result = CATEGORIES[output[0]]
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['ear']
        f.save("ear.jpg")
        f = request.files['fingerprint']
        f.save('fingerprint.jpg')

        img = cv2.imread('ear.jpg', cv2.IMREAD_GRAYSCALE)
        result = check_ear(img)

        fing_img = cv2.imread('fingerprint.jpg')
        fing_result = check_fingerprint(fing_img)

        if fing_result[0]:
            return "<h1>"+result+"</h1>"+"<h2>"+fing_result[1]+"</h2>"+" <img src='static/result_matching.jpg' alt='Fingerprint Matching result' />"
        else:
            return "No record found"


if __name__ == '__main__':
    app.run(debug=True, port=9999)
