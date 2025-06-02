from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from heart import cnn_model, xgb_model, gnn_model, extract_features, extract_graph_features

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", file.filename)
        file.save(file_path)

        try:
            img = cv2.imread(file_path)
            img = cv2.resize(img, (128, 128))

            # CNN Prediction
            cnn_input = np.expand_dims(img / 255.0, axis=0)
            cnn_pred = cnn_model.predict(cnn_input)
            cnn_result = "Normal" if np.argmax(cnn_pred) == 0 else "Abnormal"

            # XGBoost Prediction
            xgb_feat = extract_features(np.array([img]))
            xgb_pred = xgb_model.predict(xgb_feat)
            xgb_result = "Normal" if xgb_pred[0] == 0 else "Abnormal"

            # GNN Prediction
            gnn_feat = extract_graph_features(np.array([img]))
            gnn_pred = gnn_model.predict(gnn_feat)
            gnn_result = "Normal" if np.argmax(gnn_pred) == 0 else "Abnormal"

            return jsonify({
                "cnn": cnn_result,
                "xgboost": xgb_result,
                "gnn": gnn_result
            })
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
