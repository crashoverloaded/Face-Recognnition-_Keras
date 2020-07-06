from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from mtcnn.mtcnn import MTCNN
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
from keras.preprocessing import image
model = load_model('facefeatures_new_model.h5')


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
	try:
		detector = MTCNN()
		a = detector.detect_faces(img)[0]
		box = a.get('box')
		cv2.rectangle(img , (box[0],box[1]) , (box[0]+box[2] , box[1]+box[3]) , (0,255,0) , 2)
		cropped_face = img[box[1]:box[1]+box[3] , box[0]:box[0]+box[2]]

		return cropped_face
	except:
		pass    

video_capture = cv2.VideoCapture(0)
while True:
	_, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
	face=face_extractor(frame)

	if type(face) is np.ndarray:
		face = cv2.resize(face, (224, 224))
		im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
		img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
		img_array = np.expand_dims(img_array, axis=0)
		pred = model.predict(img_array)
		print(pred)     
		name="None matching"

		if(pred[0][0]>0.5):
			name='Priyank'
		cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
	else:
		cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
