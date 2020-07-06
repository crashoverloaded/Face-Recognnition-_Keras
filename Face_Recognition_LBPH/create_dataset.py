import cv2
import sys

count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
	_ , frame = cap.read()
		
	cv2.imshow("Test_Frame" , frame) 
	
	file_path = './Images/'+str(count)+'.jpg'
	
	cv2.imwrite(file_path , frame)
	
	count+=1
	
	if cv2.waitKey(1) == ord('q') or count == 1000:
		break
cap.release()
cv2.destroyAllWindows() 
	 
