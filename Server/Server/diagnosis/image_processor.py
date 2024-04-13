import cv2
import numpy as np
import os
import tensorflow as tf
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models/model_image')

if not os.path.exists(model_path):
    error_msg = f"No file or directory found at {model_path}"
    raise IOError(error_msg)


def pre_processing_image(filepath):
    image = cv2.imread(filepath)

    if image is None:
        raise ValueError("Error reading the image")

    img_resized = cv2.resize(image, (250, 250))

    b, g, r = cv2.split(img_resized)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)

    normalized = cv2.merge((b_clahe, g_clahe, r_clahe))

    processed_image_path = 'processed_' + filepath
    cv2.imwrite(processed_image_path, normalized)

    processed_image = normalized[np.newaxis, ...]
    print("Image processed")

    return processed_image


def format_probabilities(probabilities):
    diseases = {
        0: "Urticaria",
        1: "Psoriasis",
        2: "Lupus",
        3: "Dermatitis",
        4: "Melanoma"
    }

    formatted_probabilities = {"diagnosis": {}}
    for i, prob in enumerate(probabilities[0]):
        disease_name = diseases[i]
        formatted_probabilities["diagnosis"][disease_name] = f"{prob * 100:.2f}"

    return formatted_probabilities


def verify_output_type(output):
    if isinstance(output, np.ndarray):
        output = output.tolist()
        print("Prediction in numpy treated")
    return output


def runImageModel(filepath):
    try:
        processed_image = pre_processing_image(filepath)
        if not os.path.exists(model_path):
            error_msg = f"No file or directory found at {model_path}"
            raise IOError(error_msg)

        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(processed_image)
        prediction = verify_output_type(prediction)
        prediction_format = format_probabilities(prediction)
        return prediction_format
    except IOError as e:
        print("Error loading model:", e)
        return {"error": str(e)}, 500
    except Exception as e:
        print("Unexpected error:", e)
        return {"error": "Internal Server Error"}, 500


model = tf.keras.models.load_model(model_path)
