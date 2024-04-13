import string

import nltk
from django.contrib.sessions.backends.db import SessionStore
from django.http import JsonResponse
from nltk import word_tokenize
from .models.intent_classification.bert.chatbot_intent_bert_predict import predict_intent_bert
from .models.conversation.models.transformers.question_answering import generate_response_haystack
from nltk.corpus import words
from django.db.utils import OperationalError

nltk.download('words')


class DialogueManager:
    def __init__(self ):
        self.session = None
         # Reference to question answering script

    def set_session(self, request):
        self.session = SessionStore(session_key=request.session.session_key)


    def process_user_query(self, user_query, intent):
        # Tries to predict intent using BERT
        predictionquery = f'{intent} {user_query}'
        question_intent = predict_intent_bert(predictionquery)
        # intent must be respected, since we obtain it via diagnostic, in case we cant detect one intent we use same
        if question_intent is None:
            question_intent = ''
        if intent is not None:
            question_intent = f'{question_intent} {intent}'

        # is follow up
        if self.is_follow_up_query(user_query):
            try:
                last_conversation = self.get_last_non_follow_up()
                results = generate_response_haystack(f'Elaborate in detail about  {last_conversation["user_query"]}',
                                                     last_conversation["disease_intent"], self.get_conversation_history_document_ids())
                return results
            except IndexError:
                return JsonResponse(
                    {"role": "ai", "text": "I am sorry but I am missing context to answer correctly to you"})

        results = generate_response_haystack(user_query, question_intent, self.get_conversation_history_document_ids())

        return results


    def generate_bot_response(self, results):
        print(results)
        # in case score is bad we think it is not correct, or we dont know, we ask GPT
        if 'FOLLOW_UP' in results.meta['abstract']:
            try:

                last_conversation = self.get_last_non_follow_up()
                results = generate_response_haystack(f'Elaborate in detail about {last_conversation["user_query"]}',
                                                     last_conversation["disease_intent"],
                                                     self.get_conversation_history_document_ids())
            except IndexError:
                return JsonResponse(
                    {"role": "ai", "text": "I am sorry but I am missing context to answer correctly to you"})

        if results.score < 0.4:
            # Return "Disease not detected" as the answer
            # answer = generate_response_haystack_llm(query, disease_intent)
            return JsonResponse({"role": "ai",
                                 "text": "I am sorry but I did not get a good enough response for your query, can you please reformulate"})
        if 'OOS' in results.meta['abstract']:
            return JsonResponse(
                {"role": "ai", "text": "I am sorry but I am trained to answer about skin diseases only"})

        # handle response for pretty print
        resp = results.content.split('[SEP]')[1]
        resp = resp.replace('"', '')
        print(results)
        return JsonResponse({"role": "ai", "text": resp})

    def update_session(self, user_query, bot_response, disease_intent, document_id):
        # Retrieve existing conversation history from session
        conversation_history = self.session.get('conversation_history', [])

        # Append new turn to conversation history
        conversation_history.append({
            'user_query': user_query,
            'bot_response': bot_response.content.decode(),
            'disease_intent': disease_intent,
            'document_id': document_id
        })

        try:
            # Update session data with updated conversation history
            self.session['conversation_history'] = conversation_history
            self.session.save()
        except OperationalError:
            print('Not possible to save the session, please perform django migration for multi turn')


    def get_last_non_follow_up(self):
        # Iterate backward through the conversation history
        for conversation_turn in reversed(self.get_conversation_history()):
            # Normalize the user_query by converting it to lowercase
            user_query = conversation_turn['user_query'].lower()
            # Check if the user_query is not a follow-up term
            if user_query not in follow_up_terms:
                return conversation_turn
        # If all user queries are follow-up terms, return None or the first conversation turn
        return None if self.get_conversation_history() else self.get_conversation_history()[0]


    def get_conversation_history(self):
        # Retrieve conversation history from session
        return self.session.get('conversation_history', [])


    def get_conversation_history_document_ids(self):
        try:
            conversation_history = self.session.get('conversation_history', [])
            return [turn['document_id'] for turn in conversation_history]
        except KeyError:
            return None

    def is_mostly_english(self, sentence, threshold=0.6):
        # Tokenize the sentence into words
        tokens = word_tokenize(sentence)
        # Count the number of English words
        english_words_count = sum(1 for word in tokens if word.lower() in english_vocab)

        # Calculate the percentage of English words
        english_percentage = english_words_count / len(tokens)

        # Check if the percentage of English words is above the threshold
        return english_percentage >= threshold


    def not_understood(self):
        return JsonResponse({"role": "ai", "text": "I couldn't understand what you said, can you please rephrase?"})


    def is_greeting(self, user_query):
        # Remove punctuation
        sentence_no_punctuation = user_query.lower().translate(str.maketrans('', '', string.punctuation))

        return sentence_no_punctuation in greetings_map


    def greet(self, user_query):
        # Remove punctuation
        sentence_no_punctuation = user_query.lower().translate(str.maketrans('', '', string.punctuation))
        return JsonResponse({"role": "ai", "text": greetings_map[sentence_no_punctuation]})

    def is_follow_up_query(self, query):
        # Normalize the query by converting it to lowercase and removing punctuation
        normalized_query = query.lower().translate(str.maketrans('', '', string.punctuation))

        # Check if the normalized query matches any of the follow-up terms
        return normalized_query in follow_up_terms


# Set of English words
english_vocab = set(w.lower() for w in words.words())


greetings_map = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I do for you?",
    "hey": "Hey! How can I help you?",
    "good morning": "Good morning! How may I assist you?",
    "good afternoon": "Good afternoon! What can I do for you?",
    "good evening": "Good evening! How can I assist you?",
    "howdy": "Howdy! What can I do for you?",
    "greetings": "Greetings! How may I assist you?",
    "yo": "Yo! What can I help you with?",
    "what's up": "Hey! What's up? How can I assist you?",
    "thank you": "Goodbye, stay safe",
    "how are you": "I am great, how can I help?",
    "all right": "Piece",
    "ok": "Anything else?",
    "will i die":"I am trained to answer about medical diseases only",
    "should i see a doctor":"There is no substitute for medical advice, when in doubt always consult with your doctor",
    "should i see a specialist":"There is no substitute for medical advice, when in doubt always consult with your doctor"
}



follow_up_terms = [
    "how so",
    "can you please elaborate",
    "tell me more",
    "could you clarify that",
    "i'd like to hear more",
    "go on",
    "what do you mean",
    "can you explain further",
    "please explain that",
    "continue"
]


