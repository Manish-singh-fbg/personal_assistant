from tkinter import *
from tkinter import ttk 
from tkinter import messagebox
from tkinter import filedialog
import dataset_maker as dm
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image,ImageTk, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import way2sms
import sys

#varib 
phn='9040802126'
#import traindataset as td
sys.setrecursionlimit(10000)
print(sys.getrecursionlimit())
window = Tk()
window.title("Welcome to Face Recognition app")
#lbl = Label(window, text="Welcome to APP" , font=("Arial Bold", 50)) 
#lbl.grid(column=0, row=0) 
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image , 2)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(face_locations,face_encodings, knn_clf=None, model_path=None, distance_threshold=0.5):
    
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
   
    # If no faces are found in the image, return an empty result.
    if len(face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    #faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches)]


def clicked():
	res = txt.get()
	if res.strip():
		print('done')
		confirm = messagebox.askyesno('Alert',message='you want to create user with "'+ res +'" name')
		if confirm:
			result = dm.datasetCreator(res)
			if result == 'done':
				messagebox.showinfo('Thanks','Dataset Created')
			else:
				messagebox.showerror('Error','Try Again')	
		else:
			messagebox.showerror('Error','Check again')
	else:
		print('no')
		messagebox.showerror('Error','Enter User Name')
    #res = txt.get()
	#if (res !=null):
	#	print('test')
	#else:
	#	messagebox.showinfo('Message title', 'Please enter user id')

#def chooseTrainData():
#	global folder_path
#	filename = filedialog.askdirectory()
#	folder_path.set(filename)

def trainData():
	print("Training KNN classifier...")
	#path = lbt21.cget("textvariable")
	print('start')
	window.config(cursor="wait")
	window.update()
	classifire = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
	window.config(cursor="")

def saveNumber():
	mbl = phone.get()
	if len(mbl) == 10 and mbl.isdigit():
		phone1.set(mbl)	
		phn = str(phone1)
		print(type(phn))
	else:
		messagebox.showerror("Phone Number","Phone number must be of 10 digit")


def sendMessage():
					q=way2sms.Sms('9504751691','G7462H')
					q.send(phn,'unauthorized access found')
					n=q.msgSentToday()
					q.logout()

def StartApp():
	video_capture = cv2.VideoCapture(0)
	process_this_frame = True
	face_locations = []
	face_encodings = []
	face_names = []
	distance_threshold=0.5
	with open('trained_knn_model.clf', 'rb') as f:
		knn_clf = pickle.load(f)
	
	while True:
		# Grab a single frame of video
		ret, frame = video_capture.read()
		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]

		# Only process every other frame of video to save time
		if process_this_frame:
			# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame,3)
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
			predictions = predict(face_locations,face_encodings, model_path="trained_knn_model.clf")
			# Print results on the console
			face_names = []
			for name, (top, right, bottom, left) in predictions:
				print("- Found {} at ({}, {})".format(name, left, top))
				face_names.append(name)
				if "unknown" in face_names:
					try:
						sendMessage()
					except:
						print("Error in sending message")
			
			# If a match was found in known_face_encodings, just use the first one.
		process_this_frame = not process_this_frame

		for (top, right, bottom, left), name in zip(face_locations, face_names):
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video_capture.release()
	cv2.destroyAllWindows()

def StopApp():
	print('stop')
	cv2.destroyAllWindows()

window.geometry("400x300")
tab_control = ttk.Notebook(window)
 
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control) 
tab_control.add(tab1, text='Home') 
tab_control.add(tab2, text='Dataset')
tab_control.add(tab3, text='Setting') 
lbl1 = Label(tab1, text="Welcome to Face Recognition" , font=("Arial Bold", 20),bg="cyan",fg="white")  
lbl1.grid(sticky=E+W) 
img = ImageTk.PhotoImage(Image.open('a.jpg'))
panel = Label(tab1, image = img)
panel.grid(row=1)
btn1 = Button(tab1, text="Start Program", command=StartApp , height="2" , width="5",bg="yellow green",fg="black",font=("Arial Bold", 10)) 
btn1.grid(row=2,sticky=E+W)
#btn2 = Button(tab1, text="Stop Program", command=StopApp) 
#btn2.grid(column=0, row=1)


#dataset creation tab
lbl21 = Label(tab2, text="Welcome to Face Recognition" , font=("Arial Bold", 20),bg="cyan",fg="white")  
lbl21.grid(sticky=E+W, columnspan=2) 
lbl22 = Label(tab2, text="Create And Train Data" , font=("Arial Bold", 20),bg="turquoise1",fg="black")  
lbl22.grid(sticky=E+W,columnspan=2,row=1, pady=2)
lbl23 = Label(tab2, text="Enter User Name" , font=("Arial Bold", 10),bg="yellow green",fg="black")  
lbl23.grid(sticky=E+W,column=0, row=2, pady=2)
txt = Entry(tab2,width=30)
txt.grid(column=1, row=2,pady=2)
btn = Button(tab2, text="Create Dataset", command=clicked ,bg="yellow green",fg="black",font=("Arial Bold", 10)) 
btn.grid(row=3,sticky=E+W,columnspan=2)
#folder_path = StringVar()
#lbt21 = Label(tab2,textvariable=folder_path)
#lbt21.grid(column=0,row=1 )
#btn = Button(tab2, text="Browse Dataset", command=chooseTrainData)
#btn.grid(column=1, row=1)

btn = Button(tab2, text="Train Dataset", command=trainData ,bg="yellow green",fg="black",font=("Arial Bold", 10))
btn.grid(sticky=E+W,columnspan=2, row=4,pady=2)


#tab3
lbl31 = Label(tab3, text="Welcome to Face Recognition" , font=("Arial Bold", 20),bg="cyan",fg="white")  
lbl31.grid(sticky=E+W, columnspan=2) 
lbl32 = Label(tab3, text="ALert System Configuration" , font=("Arial Bold", 20),bg="cyan",fg="black")  
lbl32.grid(sticky=E+W,columnspan=2,row=1, pady=4)
lbl33 = Label(tab3, text="Enter Phone Number" , font=("Arial Bold", 10),bg="yellow green",fg="black")  
lbl33.grid(sticky=E+W,column=0, row=2, pady=4)
phone = StringVar()
phone1 = StringVar()
txt31 = Entry(tab3,width=30, textvariable=phone)
phone1.set("9040802126")
txt31.grid(column=1, row=2 ,pady =4)
lbt31 = Label(tab3,textvariable=phone1 , font=("Arial Bold", 10),bg="yellow green",fg="black")
lbt31.grid(row=3,columnspan=2,pady=4 )
btn = Button(tab3, text="Update Number", command=saveNumber , font=("Arial Bold", 10),bg="green",fg="White")
btn.grid(row=4,columnspan=2,pady=4)
tab_control.pack(expand=1, fill='both')
window.mainloop()