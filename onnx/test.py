import os

import matplotlib.pyplot as plt
import tensorflow as tf

train, test = tf.keras.datasets.cifar10.load_data()
X_test, y_test = test
X_test = X_test.astype("float32") / 255
print(X_test.shape, y_test.shape)
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

import json
from subprocess import PIPE, Popen, run

import numpy as np

idx = 1
test_example = X_test[idx : idx + 1].tolist()
payload = (
    '{"inputs":[{"name":"input_1:0","datatype":"FP32","shape":[1, 32, 32, 3],"data":'
    + f"{test_example}"
    + "}]}"
)
cmd = f"""curl -d '{payload}' \
   http://35.246.61.0/v2/models/cifar10/infer \
   -H "Content-Type: application/json"
"""
ret = Popen(cmd, shell=True, stdout=PIPE)
raw = ret.stdout.read().decode("utf-8")
res = json.loads(raw)
print(res)
arr = np.array(res["outputs"][0]["data"])
X = X_test[idx].reshape(1, 32, 32, 3)
plt.imshow(X.reshape(32, 32, 3))
plt.axis("off")
plt.show()
print("class:", class_names[y_test[idx][0]])
print("prediction:", class_names[arr.argmax()])
