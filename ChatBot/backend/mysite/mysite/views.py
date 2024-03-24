from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from .models.chatloader import generate_intent_svc
from .models.trainbot import train
from .models.trainbot_intent_svc import train_model_intent


@csrf_exempt
def train_model(request):
    train()
    return JsonResponse({"status": "error"})


@csrf_exempt
def train_intent(request):
    train_model_intent()
    return JsonResponse({"status": "success"})


@csrf_exempt
def get_response(request) :
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print(data)
        # answer = generate_response(data.query)
        # Return the JSON response
      #  return JsonResponse({"status": "success",  "data": answer})

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