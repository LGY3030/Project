from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='main_home'),
    path('price/', views.price, name='main_price'),
    path('volume/', views.volume, name='main_volume'),
]
