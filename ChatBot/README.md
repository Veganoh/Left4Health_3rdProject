To start the chatbot:

 Start the django service, go to ChatBot - backend - mysite
python3 manage.py runserver

views.py handle all the http requests and inside models/conversation/models/ you can find the algorithms used
HaystackQuestionAnserting.py is the main script

Please also do not forget to read the jupyter notebook dermachat_assistant_bot.ipynb

After it starts go to http://localhost:8000/ to see the backend APIs published
You must make a POST using postman to http://localhost:8000/chatbot/intents/haystack or http://localhost:8000/chatbot/intents/gpt
You can also use our SPA to make invocations