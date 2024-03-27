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
from .models.intent_classification.bert.chatbot_intent_bart import predict_intent_bert

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

        disease_name = answer[0][0]
        prediction_value = answer[0][1]
        # Check if the prediction value is less than 0.5
        if prediction_value < 0.41:
            # Return "Disease not detected" as the answer
            return JsonResponse({"text": "Disease not detected"})

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