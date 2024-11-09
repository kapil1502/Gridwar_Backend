import cv2
import easyocr
import numpy as np
import google.generativeai as genai
import PIL
import base64
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from helper import sharpen_image, ProductDescription, assess_freshness
from inference_sdk import InferenceHTTPClient
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="oDB7cY30zsGOxsysk7jy"
)

@app.route('/fresh-image-analysis', methods=['POST'])
def detect_fruit_freshness():
    if 'image' not in request.files:
        logging.error("No image provided in the request.")
        return jsonify({"error": "No image provided"}), 400

    file = request.files.get('image')
    image_stream = BytesIO(file.read())
    image = None
    img = None

    file.seek(0)

    # Getting the image reference
    try:
        image = PIL.Image.open(image_stream)
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return jsonify({"error": "Error decoding image"}), 400

    try:
        # Perform inference
        result = CLIENT.infer(image, model_id="smart-refrigerator-quality-and-qunatity-1/1")
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        return jsonify({"error": "Error during model inference"}), 500
    
    print(result)

    data = result
    predictions = data['predictions']

    # Extract bounding box information and annotate image
    detections = []
    try:
        for prediction in predictions:
                x1 = int(prediction['x'])
                y1 = int(prediction['y'])
                x2 = x1 + int(prediction['width'])
                y2 = y1 + int(prediction['height'])
                top_left = (x1-150, y1-150)
                bottom_right = (x2, y2)
                conf = prediction['confidence']
                cls = prediction['class']
                
                names=cls.split("_")
                name = None
                freshness = None
                freshnessIndex = 50
                shelfLife = 50

                
                if 'defective' in names:
                    freshness = 'Not a Fresh Item'
                else:
                    freshness = 'Fresh Item Ready to be consumed'

                if names[0] == 'good' or names[0] == 'defective':
                    name = names[1]
                else:
                    name = names[0]
                
                result = assess_freshness(img)

                freshnessIndex = result['freshness_score']
                shelfLife = f"{freshnessIndex}% of the Remaining Life."

                # Get the label corresponding to the class index
                detections.append({
                    "freshnessIndex": freshnessIndex,
                    "freshness": freshness,
                    "name": name.upper(),
                    "shelfLife": shelfLife,
                    "label": cls,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

                # Draw bounding box on the image
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(img, f'{cls} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the image as base64 string
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": "Error processing image"}), 500

    return jsonify({"resultImage": img_base64,
                    "detectedFruits": detections
                    })


@app.route('/analyze-freshness', methods=['POST'])
def analyzeFreshness():
    if 'image' not in request.files:
        logging.error("No image provided in the request.")
        return jsonify({"error": "No image provided"}), 400
    
   # result = CLIENT.infer(your_image.jpg, model_id="detect_fruit_fresh_or_rotten/1")

    
    img_base64=request.files.get('image')
    return jsonify({
        'resultImage': 'No image recived yet',
        'fruitDetected': 'apple',
        'freshness': 'Not detected yet',
        'freshnessIndex': 'Not detected yet'
    })

@app.route('/image-text-detection', methods=['POST'])
def ocrTextDetection():
    if 'front_image' not in request.files:
        logging.error("No image provided in the request.")
        return jsonify({"error": "No image provided"}), 400

    file = request.files['front_image']

    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return jsonify({"error": "Error decoding image"}), 400
    
    try:
        sharpened_image = sharpen_image(img)
        # Image text detector
        reader = easyocr.Reader(['en'], gpu=False)

        # Detect text on image
        text_ = reader.readtext(sharpened_image)

        # Draw bbox and text
        threshold = 0.25

        for t in text_:
                bbox, text, score = t
                if score > threshold:
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Encode the image as base64 string
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        return jsonify({"error": "Error during model inference"}), 500
    
    return jsonify({
        "image": img_base64
    })

@app.route('/image-text-extraction', methods=['POST'])
@cross_origin(origins=["http://localhost:3000"])
def ocrTextExtraction():
    if 'front_image' not in request.files:
        logging.error("No image provided in the request.")
        return jsonify({"error": "No image provided"}), 400
    

    front_image_file = request.files.get('front_image')
    front_image_stream = BytesIO(front_image_file.read())

    back_image_file = request.files.get('back_image')
    back_image_stream = BytesIO(back_image_file.read()) if back_image_file else None
    back_image = None

    side_image_file = request.files.get('side_image')
    side_image_stream = BytesIO(side_image_file.read()) if side_image_file else None
    side_image = None

    try:
        front_image = PIL.Image.open(front_image_stream)

        if back_image_stream is not None:
          back_image = PIL.Image.open(back_image_stream)

        if side_image_stream is not None:
          side_image = PIL.Image.open(side_image_stream)
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return jsonify({"error": "Error decoding image"}), 400
    
    try:
        # Configuring the Key for gemini
        genai.configure(api_key="AIzaSyBXcegqbDGH8mGKzA-jkyTdXYduSTkHOFo")
        
        # Using LLM for extracting up the product details by Levering the context awareness of it.
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        prompt = "Hello LLM I want you to do an OCR of the images that I had supplied to you by extracting up all the required details of the json format i had delivered to you already. For your Context use by date, consume by date, Best By date is same as expiry date. Also MFD, Manufacture date, created date is same as manufacturing date. price, MRP, selling price, Maximum Retail price is  same as Price of product."

        # Getting a reponse in the JSON format for the details recieved from LLM.
        items = [prompt, front_image]

        # Checking for optional arguement
        if back_image is not None:
            items.append(back_image)

        if side_image is not None:
            items.append(side_image)

        print("Images uploaded Next Porcess Step Starts....")

        response = model.generate_content(items,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=ProductDescription
            ),)
        
        data = json.loads(response.text)
        
        return jsonify(
            {
                "item_information" : response.text,
            }
        )     
    
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        return jsonify({"error": "Error during model inference"}), 500

if __name__ == '__main__':
    app.run(debug=True)