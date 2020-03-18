# FaceID
This is an implementation of face detector and recognizer which can identify the face of the person showing on a webcam.


Image preprocessing: face detection, alignment and resizing.
Built a database containing embeddings of identity images
Computed euclidean distance of embeddings between tested and identity faces to perform identity recognition 
![image](https://github.com/MengSunS/FaceID/raw/master/pictures/model.png)

For Face Detection (in real time or an image): we show how to implement face detection using OpenCV or Multi-task CNN; 

For Face Recognition (in real time or an image): we use a deep neural network, the model we use is based on [FaceNet](https://arxiv.org/pdf/1503.03832.pdf), which was published by Google in 2015 and achieved 99.57% accuracy on a popular face recognition dataset named “Labeled Faces in thae Wild(LFW)". You can find its open-source Keras version [here](https://github.com/iwantooxxoox/Keras-OpenFace) and Tensorflow version [here](https://github.com/davidsandberg/facenet), and play around to build your own models.

Click here https://github.com/MengSunS/FaceID/blob/master/DeepLearning-RealTimeFaceRecognition.ipynb to see the implementation. 









