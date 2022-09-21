import base64
import os

import cv2
import json
import numpy as np
# import tensorflow as tf
from flask import Flask, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from face_reg import encode_face

# Flask Server Backend
app = Flask(__name__)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(5, input_shape=(784,)),
#     tf.keras.layers.Softmax()
# ])
# model.save('/home/phuongpt/dev/faceId/models/my_h5_model.h5')

# Database
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
mongo = PyMongo(app)


# Router
@app.route("/", methods=['GET'])
def hello():
    return "Welcome to Python Flask!"


@app.route('/api/v1/register', methods=['POST'])
def signUp():
    try:
        _json = json.loads(request.data)
        _fullName = _json['fullName']
        _phone = _json['phone']
        _email = _json['email']
        _password = _json['password']

        user = mongo.db.users.find_one({'email': _email})
        if user:
            res = jsonify({'message': 'email in use'})
            res.status_code = 403
            return res

        hash_password = generate_password_hash(_password)

        mongo.db.users.insert_one({
            'fullName': _fullName,
            'phone': _phone,
            'email': _email,
            'password': hash_password
        })

        res = jsonify({'message': 'User added successfully'})
        res.status_code = 201
        return res
    except Exception as error:

        res = jsonify({'message': 'Bad request', 'content': str(error)})
        res.status_code = 400
        return res


@app.route("/api/v1/training", methods=['POST'])
def trainingFace():
    try:
        base64_string_array = json.loads(request.data)['file']
        email = json.loads(request.data)['email']
        print(email)
        img = []
        for i in range(len(base64_string_array)):
            image_string = base64.b64decode(base64_string_array[i])
            # print(image_string)

            jpg_as_np = np.frombuffer(image_string, dtype=np.uint8)
            # print(jpg_as_np)

            img.append(cv2.imdecode(jpg_as_np, flags=0))
            # print(np.array(img[i]))
            path = f'D:/IT/python/practice/final_server_flask/src/datasets/{email}'
            try:
                os.mkdir(path)
            except OSError as error:
                print(error)

            cv2.imwrite(os.path.join(path, f'img{i}.jpg'), img[i])

            # im = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            # # print(im)
            # im = im.reshape(784)
            # # print(im)
            #
            # loaded_model = tf.keras.models.load_model('/home/phuongpt/dev/faceId/models/my_h5_model.h5')
            # predict_x = loaded_model.predict(np.array([im]))
            # print(f'predict = {predict_x}, type: {type(predict_x)}')
            #
            # classes_x = np.argmax(predict_x, axis=1)
        encode_face(email)
        res = jsonify({'message': 'Received'})
        res.status_code = 200
        return res
    except Exception as error:

        res = jsonify({'message': 'Bad request', 'content': str(error)})
        res.status_code = 400
        return res


@app.route('/api/v1/loginByAccount', methods=['POST'])
def signInByAccount():
    try:
        _json = json.loads(request.data)
        email = _json['email']
        password = _json['password']

        if email and password:
            user = mongo.db.users.find_one({'email': email})
            if user is None:
                res = jsonify({'message': 'User not exists'})
                res.status_code = 404
                return res

            isMatch = check_password_hash(user['password'], password)
            if isMatch is False:
                res = jsonify({'message': 'Email or password is not correct'})
                res.status_code = 400
                return res

        res = jsonify({'message': 'Login successfully', 'fullName': user['fullName']})
        res.status_code = 200
        return res
    except Exception as error:

        res = jsonify({'message': 'Bad request', 'content': str(error)})
        res.status_code = 400
        return res


@app.route('/api/v1/loginByFace', methods=['POST'])
def signInByFace():
    pass


# Start Backend
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='6868')
