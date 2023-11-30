## Interview about CV project`

How can we evaluate a picture on its ability to be printable based on a few simple criteria. 
- Face present
- Smiling people
- Eyes opened
- Sharp image 
- If blurry, bokeh effect must be visible
- Luminous and crisp colors

## Application setup

If installed on local, please run 
````bash
pip install -r requirements.txt
````

With Django installed, run: 
````bash
python manage.py runserver
````

Open your browser to: 
````
http://127.0.0.1:8000/image_score/upload/
````

## Cloud access 
 This application is running during the process on:
 ````
 https://image-scoring-build-d26039211250.herokuapp.com/image_score/upload/
 ````

Upload a picture and let's take it from there! 


## Improvements

A lot of improvements as this project stands more as a proof of concept, to use a few feature are indicative on how printable a picture is. 
- Improve the face/smile/eye detection with Data Science projects 
- Improve the use of classic processing of picture to detect sharp images, colored images...
- The scoring method to mesh all those metrics all together
