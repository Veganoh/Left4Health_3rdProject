"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include, path
from .views import get_response
from .views import train_model_intent_lstm
from .views import train_model
from .views import train_intent
from .views import get_response_intent
from .views import chatbot_message_intent
from .views import get_response_intent_lstm

urlpatterns = [
    path("admin/", admin.site.urls),
    path('chitchat/',get_response, name='generate_response'),
    path('train/', train_model, name='train_model'),
    path('train_intent/<str:model_type>', train_intent, name='train_intent'),
    path('intents/<str:model_type>', get_response_intent, name='get_response_intent'),
    path('chatbot/intents/<str:model_type>', chatbot_message_intent, name='chatbot_intents')

]
