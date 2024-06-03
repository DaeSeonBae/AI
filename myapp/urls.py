from django.urls import path
from . import views

urlpatterns = [
    path('receive-query/', views.receive_query),
]
