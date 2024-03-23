from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from .models.chatloader import generate_response
from .models.trainbot import preprocess_text


@csrf_exempt
def train_model(request):
    return JsonResponse({"status": "error"})


@csrf_exempt
def get_response(request) :
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print(data)
        answer = generate_response(data.query)
        # Return the JSON response
        return JsonResponse({"status": "success",
                             "data": answer})

    return JsonResponse({"status": "error"})