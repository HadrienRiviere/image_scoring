from django.urls import path
from . import views

app_name = 'image_score_app'

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    path('display/<int:pk>/', views.display_image, name='display_image'),
]