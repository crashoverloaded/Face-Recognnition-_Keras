import numpy as np 
import cv2


# Haar Cascade Face Classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Face Extractor Function 

def face_extractor(img):
	# Detects the Face and Returns the Cropped Image
	
	faces = face_classifier.detectMultiScale(img , 1.3 , 5)
	
	if faces is ():
		return None
	
	# Cropping Faces
	
	for (x , y, w, h) in faces:	
		x-=10
		y-=10
	#	cv2.rectangle(img , (x,y) , (x+w , y+h), (0,255,0) , 3)
		cropped_face = img[y:y+h+50 , x:x+w+50]		
		
	return cropped_face

	
cap = cv2.VideoCapture(0)
count = 0

# Collecting 100 Samples of my Face 
	
while True:
	_ , frame = cap.read()
	if face_extractor(frame) is not None:
		count+=1
		face = cv2.resize(face_extractor(frame) , (400,400),cv2.INTER_AREA)

		# Saving the Faces
		file_path = './Images/'+str(count)+'.jpg'
		cv2.imwrite(file_path , face)
		cv2.putText(face , str(count) , (50,50) , cv2.FONT_HERSHEY_SIMPLEX , 1 ,(0,255,0) , 2)
		cv2.imshow('Face Cropper' , face) 
		#cv2.imshow('Full' , frame)
	
	else:
		print("Face Not Found")
		pass
	if cv2.waitKey(1) == ord('q') or count == 100:
		break

cap.release()
cv2.destroyAllWindows()

		
