import cv2
import os


def datasetCreator(Id):
	cam = cv2.VideoCapture(0)
	detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	sampleNum=0
	dl_path = "knn_examples/train/"+Id;
	if not os.path.exists(dl_path):
		os.makedirs(dl_path)
	while(True):
		ret, img = cam.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			#incrementing sample number
			sampleNum=sampleNum+1
			#saving the captured face in the dataset folder
			cv2.imwrite("knn_examples/train/"+Id+"/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
			cv2.imshow('frame',img)
		
		#wait for 100 miliseconds 
		if cv2.waitKey(200) & 0xFF == ord('q'):
			break
		# break if the sample number is morethan 20
		elif sampleNum>40:
			break
	cam.release()
	cv2.destroyAllWindows()
	return "done"
