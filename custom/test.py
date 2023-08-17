import numpy as np
import requests
from mlserver.types import InferenceRequest
from mlserver.codecs import StringCodec

input_data = ["Hello World"]
# inference_url = "http://localhost:8080/v2/models/jokes/infer"
inference_url = "http://localhost:8333/v2/models/joke-model/infer"

# Build the inference request
inference_request = InferenceRequest(
    inputs=[
        StringCodec.encode_input(name="payload", payload=input_data, use_bytes=False)
    ]
)

# Send the inference request and capture response
print("Sending Inference Request...")
res = requests.post(inference_url, json=inference_request.dict())
print("Got Response...")

# Parse the JSON string into a Python dictionary
response_dict = res.json()
print(response_dict["outputs"][0]["data"])
