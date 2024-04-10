import cv2
import numpy as np
import keras
import os
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up
model_path = os.path.abspath(
    os.path.join(current_dir, '..', '..', 'ClassificationOfSkinDiseases/Models_to_Pred/CNN_model'))
# model_path = '../ClassificationOfSkinDiseases/Models_to_Pred/CNN_model'
# model_path = '../ClassificationOfSkinDiseases/Models_to_Pred/CNN_randomsearch.h5'
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
    formatted_probabilities = []
    for prob in probabilities:
        for p in prob:
            percentage = p * 100
            print(f"Original: {p}, Novo: {percentage:.2f}%")
            formatted_probabilities.append(percentage)

    return formatted_probabilities


def verify_output_type(output):
    if isinstance(output, np.ndarray):
        output = output.tolist()
        print("Prediction in numpy treated")
    return output


def map_disease(number):
    disease_mapping = {
        0: "URTICARIA",
        1: "PSORIASIS",
        2: "LUPUS",
        3: "DERMATITIS",
        4: "MELANOMA"
    }
    return disease_mapping.get(number, "Invalid number")  # Returns "Invalid number" if the number is not in the mapping

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
        # Extract the class labels
        class_labels = np.argmax(prediction, axis=1)
        label = map_disease(class_labels[0])
        # Print or use the class labels as needed
        print("Predicted class labels:", class_labels)
        return ({'diagnosis': class_labels})
    except IOError as e:
        print("Error loading model:", e)
        return {"error": str(e)}, 500
    except Exception as e:
        print("Unexpected error:", e)
        return {"error": "Internal Server Error"}, 500



if __name__ == "__main__":
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate two levels up
    filepath = os.path.abspath(
        os.path.join(current_dir, '..', '..', 'diagnosis/Uploads/08lichenPlanusTongue1122052.jpg'))
    runImageModel(filepath)