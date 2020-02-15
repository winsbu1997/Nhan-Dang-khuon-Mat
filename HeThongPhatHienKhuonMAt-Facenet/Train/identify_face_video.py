import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import Train.facenet as facenet
import Train.detect_face as detect_face
import os
import time
import pickle
from PyQt5.QtGui import QImage, QPixmap

img_crop = 0
#url = 'http://192.168.8.102:8080/shot.jpg'
#input_video="akshay_mov.mp4"
def CameraNet(self,pnet,rnet,onet,minsize,threshold,factor,margin,frame_interval,batch_size,image_size,input_image_size,HumanNames,images_placeholder,embeddings,phase_train_placeholder,embedding_size,model,sess):
    # url = 'http://192.168.8.102:8080/shot.jpg'
    # imgResp = requests.get(url)
    # img_arr = np.array(bytearray(imgResp.content), dtype = np.uint8)
    # frame = cv2.imdecode(img_arr, -1)
    global img_crop
    faceClose = 0
    ret, frame = self.cap.read()

        #convert imgae in BGR format              

    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

    curTime = time.time()+1    # calc fps
    timeF = frame_interval
    c = 0
    if (c % timeF == 0):
        find_results = []

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Detect: %d' % nrof_faces)

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(frame.shape)[0:2]

            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces,4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    print('Face is very close!')
                    faceClose = faceClose + 1
                    continue
                try:
                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                        interpolation=cv2.INTER_CUBIC)
                    
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}

                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    print(predictions)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    # print("predictions")
                    print(best_class_indices,' with accuracy ',best_class_probabilities)

                    self.Predict.setPlainText(str(best_class_probabilities))    
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)               
                    # print(best_class_probabilities)
                    if best_class_probabilities > 0.7:
                        img_crop = frame[bb[i][1] : bb[i][3] , bb[i][0] : bb[i][2],:]
                                             
                        #plot result idx under box
                        # text_x = bb[i][0]
                        # text_y = bb[i][3] + 20
                        print('Kết quả tốt nhất: ', best_class_indices[0]+ 1)
                        print(HumanNames)
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:                                       
                                self.Name.setPlainText(str(HumanNames[best_class_indices[0]]))
                    
                      
                except IndexError :
                    print("Oops! IndexError : list index out of range for multi-faces")
        else:
            print('Không có khuôn mặt')
        if(nrof_faces > int(self.lbCoMat.text())):
            k = nrof_faces - int(self.lbCoMat.text()) - faceClose
        else:
            k = 0
        self.lbUnknow.setText(str(k)) 
        self.lbFaceClose.setText(str(faceClose))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # get image infos
    height, width, channel = frame.shape
    step = channel * width
    # create QImage from image
    qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
    # show image in img_label
    self.lbAnh.setPixmap(QPixmap.fromImage(qImg))

