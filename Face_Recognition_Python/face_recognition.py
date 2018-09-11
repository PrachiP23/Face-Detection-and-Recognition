
# coding: utf-8

# In[1]:

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:

face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')


# In[3]:

def detect_face(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = (face_cascade.detectMultiScale(gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE))
    print("Found {0} faces!".format(len(faces)))

#    for (x, y, w, h) in faces:
 #       cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
  #  return gray[y:y+w, x:x+h],faces[0]
    return faces


# In[4]:

import urllib.request
import numpy as np
import cv2
import urllib.parse
import os

def url_to_image1(url):
    # url = 'http://movies.dosthana.com/sites/default/files/image-gallery/Aaron-Eckhart-Image.jpg'
    user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64'
    #'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11','Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Referer': 'https://cssspritegenerator.com','Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3','Accept-Encoding': 'none','Accept-Language': 'en-US,en;q=0.8','Connection': 'keep-alive' 
    headers = {'User-Agent': user_agent}
    req = urllib.request.Request(url,None, headers)
    with urllib.request.urlopen(req) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image



# In[5]:

def prepare_training_data(data_folder_path):
    faces = []
    labels = []

    url_path_train_data = os.path.join("faceScrub", "train_data.txt")
    train_data = open(url_path_train_data, "r")
    train_data.readline()

    for line in train_data:
        values = line.split("\t")
        try:
            print(values[3])
            image = url_to_image1(values[3])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Image", image)
            cv2.waitKey(100)
            faceDetected = detect_face(image)
            if faceDetected is not None and len(faceDetected) ==1:
                for (x, y, w, h) in faceDetected:
                    faces.append(gray[y:y+w, x:x+h])
                    labels.append(int(values[6].replace("\n", "")))
        except:
            pass
            

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


# In[6]:

print("Preparing data...")
faces, labels = prepare_training_data("faceScrub")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


# In[7]:

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
print(type(face_recognizer))
subjects = ["", "Brad Pitt", "Christian Bale", "Courtney Cox", "David Schwimmer", "Anne Hathway","6", "Dwayne Johnson", "Jennifer Aniston", "Mathew Perry"]

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


# In[8]:

def predict(test_img, image_name):
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    faceDetected = detect_face(img)
    #predict the image using our face recognizer 
    i=0
    #draw a rectangle around face detected
    for (x, y, w, h) in faceDetected:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = 0
        label= face_recognizer.predict(gray[y:y+w, x:x+h])
        #get name of respective label returned by face recognizer
        label_text = subjects[int(label[0])]
         #draw name of predicted person
        draw_text(img, label_text, faceDetected[i][0], faceDetected[i][1]-5)
        
        #save image to respective folder
        classified_path = os.path.join("classified_images",label_text) 
        os.makedirs(classified_path, exist_ok = True)
        classified_path =os.path.join(classified_path , image_name)
        cv2.imwrite(classified_path,test_img)
        i=i+1
            
    
    return img


# In[9]:

def process_test_data(path, image_name):
    test_img = cv2.imread(os.path.join(path, image_name))
    predicted_image = predict(test_img, image_name)
    cv2.imshow(image_name, predicted_image)
    return predicted_image


# In[11]:

path = os.path.join("faceScrub","test_data")
test_images = os.listdir(path)
for image_name in test_images:
    print(image_name)
    process_test_data(path, image_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:



