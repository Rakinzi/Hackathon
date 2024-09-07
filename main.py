from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load multiple models
model_names = [
    "dima806/deepfake_vs_real_image_detection",
    "Wvolf/ViT_Deepfake_Detection",
]

processors = [AutoImageProcessor.from_pretrained(name) for name in model_names]
models = [AutoModelForImageClassification.from_pretrained(name) for name in model_names]

CONFIDENCE_THRESHOLD = 70  


def convert_to_jpg(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        jpg_path = f"{os.path.splitext(image_path)[0]}.jpg"
        image.save(jpg_path, 'JPEG')
        return jpg_path
    except Exception as e:
        print(f"Error converting image: {e}")
        return image_path


def adjust_saturation(image_path, output_path, factor=1.5):
    try:
        image = Image.open(image_path).convert("RGB")
        enhancer = ImageEnhance.Color(image)
        saturated_image = enhancer.enhance(factor)
        saturated_image.save(output_path)
    except Exception as e:
        print(f"Error adjusting saturation: {e}")
        return None
    return output_path


def process_model(model, processor, image, original_image_path):
    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        confidences = F.softmax(logits, dim=-1).squeeze().tolist()
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        face_with_mask = "Fake" if predicted_class_idx == 1 else "Real"
        prediction_confidence = confidences[predicted_class_idx] * 100

        saturated_image_path = f"{os.path.splitext(original_image_path)[0]}_saturated.jpg"
        adjust_saturation(original_image_path, saturated_image_path)

        return {
            "model": model.config._name_or_path,
            "confidences": list(enumerate(confidences)),
            "face_with_mask": face_with_mask,
            "confidence_percentage": prediction_confidence,
            'prediction': predicted_class_idx,
            'saturated_image': saturated_image_path
        }
    except Exception as e:
        print(f"Error in process_model: {e}")
        return {"error": str(e)}


def check_deepfake(image_path):
    try:
        image_path = convert_to_jpg(image_path)
        image = Image.open(image_path).convert("RGB")

        results = []
        for model, processor in zip(models, processors):
            result = process_model(model, processor, image, image_path)
            if 'confidences' in result:
                result['confidences'] = [(idx, float(conf)) for idx, conf in result['confidences']]
            results.append(result)

        return results
    except Exception as e:
        print(f"Error in check_deepfake: {e}")
        return {"error": str(e)}

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename.endswith(('png', 'jpg', 'jpeg', 'webp')):
            print(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            results = check_deepfake(file_path)
            return jsonify(results)
        else:
            return jsonify({"error": "Invalid file format. Please upload an image in PNG, JPG, JPEG, or WebP format."}), 400


if __name__ == '__main__':
    app.run(debug=True)
