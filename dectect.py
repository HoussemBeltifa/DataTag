from ultralytics import YOLO
import os
from flask import request, Response, Flask, jsonify
from waitress import serve
from PIL import Image
import json
from pathlib import Path
import torch
import uuid
import shutil

app = Flask(__name__)



def create_unique_folder():
    unique_id = str(uuid.uuid4())
    folder_path = os.path.join(os.getcwd(), unique_id)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path

model_folder = create_unique_folder()

@app.route('/create_unique_folder', methods=['GET'])
def api_create_unique_folder():
    global model_folder
    model_folder = create_unique_folder()
    return jsonify({"message": "Unique folder created successfully", "model_folder": model_folder})

choice = ""
@app.route('/choose_model', methods=['POST'])
def api_choose_model():
    global model_folder,choice

    choices = ["n", "s", "m", "l", "x"]
    data = request.get_json()
    choice = data.get('choice', '').lower()

    if choice not in choices:
        return jsonify({"error": "Invalid choice. Please provide a valid choice."}), 400

    return jsonify({"message": f"Model choice '{choice}' processed successfully", "model_folder": model_folder})


if choice in ["n", "s", "m", "l", "x"]:
    model = YOLO("yolov8"+choice+".pt")


@app.route('/train_model', methods=['POST'])
def api_train_model():
    global model, model_folder

    dataset = request.files['dataset']
    dataset.save(os.path.join(model_folder, 'data.yaml'))

    model.train(task="detect", data=os.path.join(model_folder, 'data.yaml'), epochs=10)

    source_path = '/content/runs/detect/train3/weights/best.pt'  
    destination_folder = model_folder

    shutil.move(source_path, destination_folder)

    return jsonify({"message": "Model trained and best instance saved successfully"})


@app.route("/detect", methods=["POST"])
def api_detect():
    global model, model_folder

    if model is None:
        return jsonify({"error": "Model not initialized. Please choose a model first."}), 400

    buf = request.files["image_file"]
    image = Image.open(buf.stream)

    annotation_directory = os.path.join(model_folder, "annotation")
    os.makedirs(annotation_directory, exist_ok=True)

    annotation_path = detect_objects_on_image(image, model, annotation_directory)

    return jsonify({"message": "Detection successful", "annotation_path": annotation_path})

def detect_objects_on_image(image, model, annotation_directory):
    with torch.no_grad():
        predictions = model(image, task="detect")

    result = predictions[0].boxes.data.tolist()
    output = []

    for box in result:
        x1, y1, x2, y2 = box[:4]
        class_id = int(box[5])
        confidence = round(box[4], 2)

        output.append({
            "object_type": model.names[class_id],
            "x1": round(x1),
            "y1": round(y1),
            "x2": round(x2),
            "y2": round(y2),
            "probability": confidence
        })

    image_filename = os.path.basename(annotation_directory.rstrip('/'))
    annotation_filename = f'{image_filename}.txt'
    annotation_path = os.path.join(annotation_directory, annotation_filename)

    with open(annotation_path, 'w') as f:
        for i in output:
            f.write(f'{i["object_type"]} : {i["x1"]}, {i["y1"]}, {i["x2"]}, {i["y2"]}, probability = {i["probability"]}')
            f.write('\n')

    return annotation_path




# @app.route("/detect", methods=["POST"])
# def detect():
#     """
#         Handler of /detect POST endpoint
#         Receives uploaded file with a name "image_file", 
#         passes it through YOLOv8 object detection 
#         network and returns an array of bounding boxes.
#         :return: a JSON array of objects bounding 
#         boxes in format 
#         [[x1,y1,x2,y2,object_type,probability],..]
#     """
#     buf = request.files["image_file"]
#     boxes = detect_objects_on_image(Image.open(buf.stream))
#     return Response(
#       json.dumps(boxes),  
#       mimetype='application/json'
#     )


# def detect_objects_on_image(buf):
#     """
#     Function receives an image,
#     passes it through YOLOv8 neural network
#     and returns an array of detected objects
#     and their bounding boxes
#     :param buf: Input image file stream
#     :return: Array of bounding boxes in format 
#     [[x1,y1,x2,y2,object_type,probability],..]
#     """
#     model = YOLO("best.pt")
#     results = model.predict(buf)
#     result = results[0]
#     output = []
#     for box in result.boxes:
#         x1, y1, x2, y2 = [
#           round(x) for x in box.xyxy[0].tolist()
#         ]
#         class_id = box.cls[0].item()
#         prob = round(box.conf[0].item(), 2)
#         output.append([
#           x1, y1, x2, y2, result.names[class_id], prob
#         ])
#     return output

serve(app, host='0.0.0.0', port=8080)




  """
names:
- '0'
- car
- cars
- cups
- human
nc: 5
test: ../test/images
train: DataTag-1/train/images
val: DataTag-1/valid/images
  """