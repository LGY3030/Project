from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='main_home'),
    path('price/', views.price, name='main_price'),
    path('volume/', views.volume, name='main_volume'),
    path('price_trend/', views.price_trend, name='main_price_trend'),
    path('price_trend_compare/', views.price_trend_compare, name='main_price_trend_compare'),
]

