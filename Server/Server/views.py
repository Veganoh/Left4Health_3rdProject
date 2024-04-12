from django.http import JsonResponse, HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib.sessions.backends.db import SessionStore
from .chatbot.dialogmanager import DialogueManager
from .chatbot.models.intent_classification.svc.chatloader import generate_intent_svc
from .chatbot.models.intent_classification.lstm.chatloader_intent_lstm import predict_intent_lstm
from .chatbot.models.intent_classification.bilstm.chatbot_intent_bilstm_pos import predict_intent_bilstm_pos
from .chatbot.models.intent_classification.bert.chatbot_intent_bert import predict_intent_bert
from .chatbot.models.conversation.models.llm.chatgpt import generate_answer_with_intent
from .chatbot.models.conversation.models.llm.chatgpt import generate_answer_without_intent
from .chatbot.models.conversation.models.transformers.question_answering import generate_response_haystack
from .chatbot.models.conversation.models.transformers.question_answering import generate_response_haystack_llm
from .chatbot.models.conversation.models.transformers.Tester import get_analytics_queries
from .diagnosis.image_processor import runImageModel
from .diagnosis.pre_processing import runModel
import os


valid_diseases = ['melanoma', 'dermatitis', 'psoriasis', 'urticaria', 'lupus']

current_dir = os.path.dirname(os.path.abspath(__file__))
two_levels_up = os.path.abspath(os.path.join(current_dir, '..'))
image_file_path = os.path.join(two_levels_up, 'diagnosis')



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
def text_diagnosis(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        user_input = data['User_input']
        disease = runModel(user_input)
        return JsonResponse({'diagnosis': disease})
    return JsonResponse({'error': 'Invalid request method'})


@csrf_exempt
def chatbot_message_intent(request, model_type):

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

            case 'haystack':
                # Initializing the dialogue manager
                dialogue_manager = DialogueManager()
                dialogue_manager.set_session(request)

                if not dialogue_manager.is_mostly_english(query):
                    return dialogue_manager.not_understood()

                if dialogue_manager.is_greeting(query):
                    return dialogue_manager.greet(query)

                results = dialogue_manager.process_user_query(query,disease_intent)

                # Generate bot response
                bot_response = dialogue_manager.generate_bot_response(results)

                # Update session with conversation turn
                dialogue_manager.update_session(query, bot_response, disease_intent)

                return bot_response
            case 'gpt':
                if process_disease:
                    formatted_answer = generate_answer_with_intent(data['messages'], disease_intent)
                else:
                    formatted_answer = generate_answer_without_intent(data['messages'])
                return JsonResponse(formatted_answer, safe=False)

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