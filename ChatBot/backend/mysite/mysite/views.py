from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from .models.chatloader import generate_intent_svc, generate_response
from .models.trainbot import train
from .models.trainbot_intent_svc import train_model_intent
from .models.trainbot_intent_lstm import train_model_intent_lstm
from .models.chatloader_intent_lstm import predict_intent_lstm
from .models.intent_classification.bilstm.chatbot_intent_bilstm_pos import train_intent_bilstm_pos
from .models.intent_classification.bilstm.chatbot_intent_bilstm_pos import predict_intent_bilstm_pos


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
def get_response_intent(request, model_type):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print(data)
        answer = 'NA'
        match model_type:
            case 'bilstm_pos':
                answer = predict_intent_bilstm_pos(data['query'])
            case 'svc':
                answer = generate_intent_svc(data['query'])
            case 'lstm':
                answer = predict_intent_lstm(data['query'])

        # Return the JSON response
        return JsonResponse({"status": "success",
                             "data": answer})

    return JsonResponse({"status": "error"})

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