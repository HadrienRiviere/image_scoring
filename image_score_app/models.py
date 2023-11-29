from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    # score = models.FloatField(null=True, blank=True)