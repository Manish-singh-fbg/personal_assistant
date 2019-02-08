import speech_recognition
from time import ctime
import time
import os
import sys
import pyttsx3
#from gtts import gTTS
 
speech_engine = pyttsx3.init('sapi5') # see http://pyttsx.readthedocs.org/en/latest/engine.html#pyttsx.init
speech_engine.setProperty('rate', 150)
def speak(text):
	speech_engine.say(text)
	speech_engine.runAndWait()
 
 
recognizer = speech_recognition.Recognizer()
def recordAudio():
# Record Audio
	with speech_recognition.Microphone() as source:
		recognizer.adjust_for_ambient_noise(source)
		print("Say something!")
		audio = recognizer.listen(source)
		print(audio)
	data = ""
	try:
		data =  recognizer.recognize_google(audio) #recognizer.recognize_sphinx(audio)
		print(data)
		# or: return recognizer.recognize_google(audio)
	except speech_recognition.UnknownValueError:
		print("Could not understand audio")
	except speech_recognition.RequestError as e:
		print("Recog Error; {0}".format(e))

	return data
def jarvis(data):
	if "how are you" in data:
		speak("I am fine")
	
	if "what time is it" in data:
	    speak(ctime())
 
	if "where is" in data:
		data = data.split(" ")
		location = data[2]
		speak("Hold on Frank, I will show you where " + location + " is.")
		os.system("start chrome https://www.google.nl/maps/place/" + location + "/&amp;")
		#os.system("chromium-browser https://www.google.nl/maps/place/" + location + "/&amp;")
 
# initialization
time.sleep(2)
speak("Hi Frank, what can I do for you?")
while 1:
	data = recordAudio()
	jarvis(data)