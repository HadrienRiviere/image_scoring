from dataclasses import dataclass
import logging

import base64
import cv2
from django.db.utils import OperationalError
from django.contrib import messages
import numpy as np

from .forms import ImageUploadForm


@dataclass
class DataClassifier:
    """
    Scorer for images: if above 30%, we shoudl consider this a success.
    Very basic approach of the topic:
    - Analysis of luminosity, blurness
    - Face, smile and eyes detection
    """
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def has_face(self, img):
      faces = self.face_detector.detectMultiScale(img)
      return faces

    def is_smiling(self, img, faces):
        smile_cnt = 0
        for face in faces:
            tlx, tly, w, h = face[0], face[1], face[2], face[3]
            cropped_face = img[tly:tly+h, tlx:tlx+w]

            smile = self.smile_detector.detectMultiScale(cropped_face)
            if len(smile) > 0:
                smile_cnt += 1
        return 1 if len(faces) == smile_cnt and len(faces) != 0 else 0

    def has_eyes(self, img, faces):
        eyes_cnt = 0
        if len(faces) ==0:
            return 0.0

        for face in faces:
            tlx, tly, w, h = face[0], face[1], face[2], face[3]
            cropped_face = img[tly:tly+h, tlx:tlx+w]

            eyes = self.smile_detector.detectMultiScale(cropped_face)
            if len(eyes):
                eyes_cnt += len(eyes)
        return 1 if eyes_cnt / len(faces) >= 2 else 0

    def nice_lighting(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1].mean()
        distance = np.abs(128-saturation)
        return 1 - distance/128

    def bokeh_effect(self, img, faces):
        if len(faces) == 0:
            return 0
        face_blur = []
        for face in faces:
            tlx, tly, w, h = face[0], face[1], face[2], face[3]
            cropped_face = img[tly:tly+h, tlx:tlx+w]

            face_blur.append(self.is_sharp(cropped_face))
        return np.mean(face_blur)

    def is_sharp(self, img):
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        return np.min([300, blur])/300

    def infer(self, input_file):
        """
        Retrieve the picture data, load it and infer the score from all the features we want to detect
        :param input_file: input path from form upload
        :return: score and if the quality is sufficient
        """
        input_file = "uploads/" + input_file
        try:
            image = cv2.imread(input_file, 0)
            image_colored = cv2.imread(input_file)
            score = self.infer_scores(image, image_colored)
        except cv2.error:
            return None
        if score > 30:
            return (score, "Success")
        return (score, "Fail")

    def infer_scores(self, img, colored):
        """
        Retrieve the faces detected, the bokeh factor (sharpness of the face(s) detected
        :param img: image
        :param colored: colored image
        :return: score
        """
        faces = self.has_face(img)
        bokeh = self.bokeh_effect(img, faces)
        bokeh_bonus = np.sqrt((2*bokeh-0.5*self.is_sharp(img))) if bokeh != 0 else 0.5
        full_score = ((self.nice_lighting(colored) + bokeh)/2)*(self.is_smiling(img, faces)*30 + self.has_opened_eyes(img, faces)*30 + 40)*bokeh_bonus
        return full_score

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            error_message = ""
            image = form.cleaned_data['image']
            image_content = base64.b64encode(image.file.read())
            classifier = DataClassifier()
            upload_image = UploadedImage(image=image)
            try:
                upload_image.save()
            except OperationalError:
                logging.debug("Couldn't push to DB")
            score = classifier.infer(input_file=image.name)
            if score == None:
                messages.error(request, f"Error during preprocessing: Upload a new resource")
                return render(request, 'image_score_app/upload_image.html', {'form': form, "error": error_message})
            score = f"{str(round(score[0], 1))}% Likely to be printable photo: {score[1]}"
            return render(request, 'image_score_app/display_image.html', {'image': image_content, 'score': score})
    else:
        form = ImageUploadForm()
    return render(request, 'image_score_app/upload_image.html', {'form': form})


from django.shortcuts import render, get_object_or_404
from .models import UploadedImage

def display_image(request, pk):
    uploaded_image = get_object_or_404(UploadedImage, pk=pk)
    return render(request, 'image_score_app/display_image.html', {'uploaded_image': uploaded_image})