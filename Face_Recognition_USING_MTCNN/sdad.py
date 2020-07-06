from mtcnn.mtcnn import MTCNN

import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
	_ , frame = cap.read()
	
	try:
		detector = MTCNN()
		a = detector.detect_faces(frame)[0]
		box = a.get('box')
		cv2.rectangle(frame , (box[0],box[1]) , (box[0]+box[2] , box[1]+box[3]) , (0,255,0) , 2)
		l_eye = a.get('keypoints').get('left_eye')
		r_eye = a.get('keypoints').get('right_eye')
		nose = a.get('keypoints').get('nose')
		l_mouth = a.get('keypoints').get('mouth_left')
		r_mouth = a.get('keypoints').get('mouth_right')

		cv2.circle(frame,l_eye,2,(0,255,0),-1)
		cv2.circle(frame,r_eye,2,(0,255,0),-1)
		cv2.circle(frame,nose,2,(0,255,0),-1)
		cv2.circle(frame,l_mouth,2,(0,255,0),-1)
		cv2.circle(frame,r_mouth,2,(0,255,0),-1)

		cv2.imshow('sa' , frame)
		if cv2.waitKey(1) == ord('q'):
			break
	except:
		pass

cap.release()
cv2.destroyAllWindows()

