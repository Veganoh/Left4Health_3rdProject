import nltk
from django.http import JsonResponse, HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt
from .models.intent_classification.svc.chatloader import generate_intent_svc, generate_response
from .models.conversation.models.lstm.trainbot import train
from .models.intent_classification.svc.trainbot_intent_svc import train_model_intent
from .models.intent_classification.lstm.trainbot_intent_lstm import train_model_intent_lstm
from .models.intent_classification.lstm.chatloader_intent_lstm import predict_intent_lstm
from .models.intent_classification.bilstm.chatbot_intent_bilstm_pos import train_intent_bilstm_pos
from .models.intent_classification.bilstm.chatbot_intent_bilstm_pos import predict_intent_bilstm_pos
from .models.intent_classification.bert.chatbot_intent_bert import predict_intent_bert
from .models.intent_classification.bert.chatbot_intent_bert_intents import predict_intent_bert_intents
from .models.conversation.models.llm.chatgpt import generate_answer_with_intent
from .models.conversation.models.llm.chatgpt import generate_answer_without_intent
from .models.conversation.models.roberta.HaystackQuestionAnserting import generate_response_haystack
from .models.conversation.models.roberta.HaystackQuestionAnserting import generate_response_haystack_llm
from .models.conversation.models.roberta.Tester import get_analytics_queries
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from diagnosis.pre_processing import runModel
from diagnosis.image_processor import runImageModel
from werkzeug.utils import secure_filename

import os

from ..diagnosis.image_processor import runImageModel

nltk.download('words')

valid_diseases = ['melanoma', 'dermatitis', 'psoriasis', 'urticaria', 'lupus']

current_dir = os.path.dirname(os.path.abspath(__file__))
two_levels_up = os.path.abspath(os.path.join(current_dir, '..'))
image_file_path = os.path.join(two_levels_up, 'diagnosis')

@csrf_exempt
def chatbot_message_intent(request, model_type):
    # this rest api service is called from chatbot
    # depending on parameter one of the models will be used
    # after many tests at the moment llm is the most prolific followd by haystack
    # haystack hybrid retrieval colab can be found
    # at https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/33_Hybrid_Retrieval.ipynb

    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print(data)
        query = data['messages'][0]['text']
        disease_intent = ''
        process_disease = False
        if 'disease_intent' in data:
            # Convert disease_intent to lowercase for case-insensitive comparison
            disease_intent = data['disease_intent'].lower()

            # Check if disease_intent matches any valid diseases
            if disease_intent in valid_diseases:
                process_disease = True

        print(data)
        answer = 'NA'
        match model_type:
            case 'bilstm_pos':
                # is_mostly_english does quick scan
                if not is_mostly_english(query):
                    return JsonResponse({"role": "ai", "text": "I couldn't understand what you said, can you please rephrase?"})
                answer = predict_intent_bilstm_pos(query)
            case 'svc':
                if not is_mostly_english(query):
                    return JsonResponse({"role": "ai", "text": "I couldn't understand what you said, can you please rephrase?"})
                answer = generate_intent_svc(query)
            case 'lstm':
                if not is_mostly_english(query):
                    return JsonResponse({"role": "ai", "text": "I couldn't understand what you said, can you please rephrase?"})
                answer = predict_intent_lstm(query)
            case 'bert':
                if not is_mostly_english(query):
                    return JsonResponse({"role": "ai", "text": "I couldn't understand what you said, can you please rephrase?"})
                answer = predict_intent_bert(query)
            case 'multitaskbert':
                if not is_mostly_english(query):
                    return JsonResponse({"role": "ai", "text": "I couldn't understand what you said, can you please rephrase?"})
                answer = predict_intent_bert_intents(query)
            case 'haystack':
                if not is_mostly_english(query):
                    return JsonResponse({"role": "ai", "text": "I couldn't understand what you said, can you please rephrase?"})
                answer = generate_response_haystack(query, disease_intent)
                print(answer)

                # in case score is bad we think it is not correct, or we dont know, we ask GPT
                if answer.score < 0.1:
                    # Return "Disease not detected" as the answer
                    #answer = generate_response_haystack_llm(query, disease_intent)
                    return JsonResponse({"role": "ai", "text": "I am sorry but I did not get a good enough response for your query, can you please reformulate"})
                if 'OOS' in answer.meta['abstract']:
                    return JsonResponse({"role": "ai", "text": "I am sorry but I am trained to answer about skin diseases only"})

                # handle response for pretty print
                resp = answer.content.split('[SEP]')[1]
                resp = resp.replace('"', '')

                return JsonResponse({"role": "ai", "text": resp})
            case 'gpt':
                if process_disease:
                    formatted_answer = generate_answer_with_intent(data['messages'], disease_intent)
                else:
                    formatted_answer = generate_answer_without_intent(data['messages'])
                return JsonResponse(formatted_answer, safe=False)
        print(answer)
        disease_name = answer[0][0]
        prediction_value = answer[0][1]
        # Check if the prediction value is less than 0.5
        if prediction_value < 0.5:
            # Return "Disease not detected" as the answer
            return JsonResponse({"role": "ai", "text": "Disease not detected"})
        if disease_name == 'OOS':
            # Return "Disease not detected" as the answer
            return JsonResponse({"role": "ai", "text": "I am sorry but I am trained to answer about skin diseases only"})

        # Create a formatted string with the disease name and prediction value
        formatted_text = f"{disease_name} ({prediction_value:.2f})"
        # Return the JSON response
        return JsonResponse({"text": formatted_text})
    if request.method == 'OPTIONS':
        response = HttpResponse()
        response['allow'] = 'post'
        return response

@csrf_exempt
def text_diagnosis(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        user_input = data['User_input']
        disease = runModel(user_input)
        return JsonResponse({'diagnosis': disease})
    return JsonResponse({'error': 'Invalid request method'})


@csrf_exempt
def image_diagnosis(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:

            return JsonResponse({'error': 'No file part'}, status=400)

        file = request.FILES['image']

        if file.name == '':

            return JsonResponse({'error': 'No selected file'}, status=400)

        if file:
            filename = file.name
            if not os.path.exists(image_file_path):
                os.makedirs(image_file_path)
            filepath = os.path.join(image_file_path, filename)
            with open(filepath, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            label = runImageModel(filepath)
            return JsonResponse({'diagnosis': label})


        return JsonResponse({'error': 'Invalid file format'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def get_response_intent(request, model_type):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        query = data['messages'][0]['text']
        print(data)
        answer = 'NA'
        match model_type:
            case 'bilstm_pos':
                answer = predict_intent_bilstm_pos(query)
            case 'svc':
                answer = generate_intent_svc(query)
            case 'lstm':
                answer = predict_intent_lstm(query)
            case 'bert':
                answer = predict_intent_bert(query)
            case 'multitaskbert':
                answer = predict_intent_bert(query)

        # Return the JSON response
        return JsonResponse({"status": "success",
                             "data": answer})
    if request.method == 'OPTIONS':
        response = HttpResponse()
        response['allow'] = 'post'
        return response

@csrf_exempt
def get_response_intent_lstm(request) :
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print(data)
        answer = predict_intent_lstm(data['query'])
        # Return the JSON response
        return JsonResponse({"status": "success",
                             "data": answer})

    return JsonResponse({"status": "error"})


# Set of English words
english_vocab = set(w.lower() for w in words.words())


def is_mostly_english(sentence, threshold=0.6):
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)
    # Count the number of English words
    english_words_count = sum(1 for word in tokens if word.lower() in english_vocab)

    # Calculate the percentage of English words
    english_percentage = english_words_count / len(tokens)

    # Check if the percentage of English words is above the threshold
    return english_percentage >= threshold


def evaluate_haystack_results(request):
    analytics = get_analytics_queries()
    intent = None
    valid_arr = []
    llm_arr = []
    invalid_arr=[]

    # valid queries
    for query in analytics['list_of_queries_valid']:
        answer = generate_response_haystack(query['query'], intent)
        ans = answer.content.split('[SEP]')[1]
        result = f'[ {query}, Answer : {answer}, Ranking: {answer.score}'
        valid_arr.append(result)
        print(result)
        # in case score is bad we think it is not correct, or we dont know, we ask GPT
        if answer.score < 0.6:
            # Return "Disease not detected" as the answer
            llm_answer = generate_response_haystack_llm(query['query'], intent)
            result = f'[ {query}, Answer : {answer}, Ranking: {answer.score}, LLM_Answer: {llm_answer}'
            print(result)
            llm_arr.append(result)

    # invalid queries
    for query in analytics['list_of_queries_invalid']:
        print(query)
        print(query['query'])
        answer = generate_response_haystack(query['query'], intent)
        ans = answer.content.split('[SEP]')[1]
        result = f'[ {query}, Answer : {answer}, Ranking: {answer.score}'
        print(result)
        invalid_arr.append(result)


    with open('chatbot_report_valid.txt', "w") as file:
        for string in valid_arr:
            file.write(string + "\n")
    with open('chatbot_report_valid_augmented.txt', "w") as file:
        for string in llm_arr:
            file.write(string + "\n")
    with open('chatbot_report_invalid.txt', "w") as file:
        for string in invalid_arr:
            file.write(string + "\n")

    return JsonResponse({"status": "success"})