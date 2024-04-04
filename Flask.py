from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from flask_cors import CORS, cross_origin
import pandas as pd
from pymongo.mongo_client import MongoClient
from deepface import DeepFace

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/')
@cross_origin()
def hello():
    return 'Hi'


@app.route('/process_post', methods=['POST'])
@cross_origin()
def process_post():
    data = request.json
    image = data['base64Data']
    result = verify(img64=image)
    if result != "Unknown":
        uri = "mongodb+srv://dbFaceData:123456Truong@cluster0.5jerjec.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(uri)
        db = client['dbFace']
        collection = db['inforStudents']
        student_info = collection.find_one({"mssv": result}, {"_id": 0})
        return jsonify(student_info)
    else:
        return jsonify({"message": "Unknown"})


def verify(img64):
    # Get image from base64
    image_data = base64.b64decode(img64)
    image_array = np.frombuffer(image_data, np.uint8)
    image_save = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Get Database features
    file_path = r"database\face_embeddings.pkl"
    # Read file .pkl
    data = pd.read_pickle(file_path)
    results = pd.DataFrame(data)
    # Get features image
    embedding_objs = DeepFace.represent(image_save, model_name='ArcFace', detector_backend='skip')
    target_embedding = embedding_objs[0]['embedding']
    distances = []
    target_threshold = 0.6
    for _, instance in results.iterrows():
        source_representation = instance["embedding"]
        # Compute distance
        distance = DeepFace.verification.find_distance(
            alpha_embedding=source_representation, beta_embedding=target_embedding, distance_metric="cosine"
        )
        distances.append(distance)

    # results["threshold"] = target_threshold
    results["distance"] = distances
    results = results.sort_values(by=["distance"], ascending=True)
    if results['distance'].iloc[0] <= target_threshold:
        return results['mssv'].iloc[0]
    else:
        return "Unknown"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)