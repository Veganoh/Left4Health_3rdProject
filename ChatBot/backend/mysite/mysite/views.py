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
from nltk.corpus import words
from nltk.tokenize import word_tokenize

nltk.download('words')

valid_diseases = ['melanoma', 'dermatitis', 'psoriasis', 'urticaria', 'lupus']

@csrf_exempt
def train_model(request):
    train()
    return JsonResponse({"status": "error"})


@csrf_exempt
def train_intent(request, model_type):
    match model_type:
        case 'bilstm_pos': train_intent_bilstm_pos()
        case 'svc': train_model_intent()
        case 'lstm': train_model_intent_lstm(request)

    return JsonResponse({"status": "success"})


@csrf_exempt
def get_response(request) :
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print(data)
        answer = generate_response(data['query'])
        # Return the JSON response
        return JsonResponse({"status": "success",  "data": answer})

    return JsonResponse({"status": "error"})


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
                if answer.score < 0.6:
                    # Return "Disease not detected" as the answer
                    answer = generate_response_haystack_llm(query, disease_intent)
                    return JsonResponse({"role": "ai", "text": answer})
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