
import cv2
#from keras.models import load_model
import numpy as np

def pre_processing_image( filepath):

    image = cv2.imread(filepath)

    if image is None:
        raise ValueError("Erro ao ler a imagem")

    # Redimensiona a imagem
    img_redimensionada = cv2.resize(image, (250, 250))

    # Split the BGR image into separate channels
    b, g, r = cv2.split(img_redimensionada)

    # Apply CLAHE to each channel separately
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b_clahe = clahe.apply(b)
    g_clahe = clahe.apply(g)
    r_clahe = clahe.apply(r)

    # Merge the CLAHE-enhanced channels back into a BGR image
    normalized = cv2.merge((b_clahe, g_clahe, r_clahe))

    # Salva a imagem pré-processada
    processed_image_path = 'processed_' + filepath
    cv2.imwrite(processed_image_path, normalized)

    return image


#model = load_model('../ClassificationOfSkinDiseases/CNN_randomsearch/CNN_randomsearch.h5')


def runModel(filepath):
    processed_image = pre_processing_image(filepath)
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(processed_image)
    print(prediction)
    disease = prediction
    return disease

