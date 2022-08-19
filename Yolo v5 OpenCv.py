
import torch
import numpy as np
import cv2
import time
class test:

    #class is created
    def __init__(self,model_name):

        self.model=self.imodel(model_name)
        
        self.classes=self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
    def icamera(self):

        #opeinig camera

        return cv2.VideoCapture(0)

    def imodel(self,model_name):
        #Giving modelname and custom model  
        if model_name:

           model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5', pretrained=True)   
        return model
    def score_frame(self, img):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        img = [img]
        results = self.model(img)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]
        
    def plot_boxes(self ,results, img):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = img.shape[1], img.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(img, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return img


    def __call__(self):


        cap=self.icamera()  
        assert cap.isOpened()

        while cap.isOpened():
           ## global frame
            while True:
             ret, img=cap.read()
             #img = cv2.resize(img, (416,416))
             start = time.time()
             
             results = self.score_frame(img)
             img = self.plot_boxes(results, img)
             
             if not ret:
              print("Ignore") 
              continue
             


    
    
             k=cv2.waitKey(10)
             end = time.time()
             fps = 1 / (end - start)
             print("FPS: ",fps)
             
             cv2.putText(img,f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            # cv2.putText(img,f'Working: ',(200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (10,255,10), 2)
             cv2.imshow('sijan cam',img)
             k=cv2.waitKey(5)
             if k==5:
                break;
            cap.release()   
             
s = test(model_name='yolov5s')

s()




