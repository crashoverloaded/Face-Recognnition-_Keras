import numpy as np 
import cv2
from mtcnn.mtcnn import MTCNN


def face_extractor(img):
	# Detects the Face and Returns the Cropped Image
	
	try:	
		detector = MTCNN()
		a = detector.detect_faces(img)[0]
		box = a.get('box')
		cv2.rectangle(img , (box[0],box[1]) , (box[0]+box[2] , box[1]+box[3]) , (0,255,0) , 2)
		cropped_face = img[box[1]:box[1]+box[3] , box[0]:box[0]+box[2]]		
		
		return cropped_face
	except:	
		pass
	

	
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

		
