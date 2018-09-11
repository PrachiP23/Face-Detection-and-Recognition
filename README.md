# Face Detection and Recognition
1.	Read the images from URL mentioned in the data file. Histogram equalization was applied before any processing to improve detection accuracy for images. 
2.	Face detection was implemented using two methods:
    1)	Skin segmentation and morphological processing (MATLAB)
    2) Viola-Jones Detection (OpenCV) - better performance
3.	LBP face recognizer was trained using the detected faces and their corresponding labels.
4.	For a new image, predicted the faces detected using the trained recognizer and displayed a rectangle around it with the predicted label.
