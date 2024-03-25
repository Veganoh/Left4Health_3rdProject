from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from .models.chatloader import generate_intent_svc, generate_response
from .models.trainbot import train
from .models.trainbot_intent_svc import train_model_intent
from .models.trainbot_intent_lstm import train_model_intent_lstm
from .models.chatloader_intent_lstm import predict_intent_lstm


@csrf_exempt
def train_model(request):
    train()
    return JsonResponse({"status": "error"})


@csrf_exempt
def train_intent(request):
    train_model_intent()
    return JsonResponse({"status": "success"})


def train_intent_lstm(request):
    train_model_intent_lstm()
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
def get_response_intent(request) :
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print(data)
        answer = generate_intent_svc(data['query'])
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