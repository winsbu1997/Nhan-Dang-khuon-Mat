from Train.preprocess import preprocesses
from Train.classifier import training
import sys
# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow
from PyQt5.QtGui import QImage, QPixmap 
from PyQt5.QtCore import QTimer, QThread
from PyQt5 import uic
import cv2
import sqlite3
import os
from TabWidget2 import *

form_class  = uic.loadUiType("main.ui")[0]
count = 0

def Insert(TEN, KHOAHOC, self):                                            
    connect = sqlite3.connect("database.db")                                  
    cmd = "SELECT * FROM SINHVIEN WHERE TEN = '" + TEN + "'"                    
    cursor = connect.execute(cmd)
    isRecordExist = 0
    cmd1 = "select count(*) from SINHVIEN"
    STT = connect.execute(cmd1).fetchone()[0]
    for row in cursor:                                                          
        isRecordExist = 1
    if isRecordExist == 1:                                                    
        QMessageBox.about(self, "Error","Trùng tên mời nhập lại")
        return 0
    else:
    	params = (TEN, KHOAHOC, STT)                                              
    	connect.execute("INSERT INTO SINHVIEN(TEN, KHOAHOC, STT) VALUES(?, ?, ?)", params)
    connect.commit()                                                           
    connect.close()  
    return 1

class MyWindowClass(QMainWindow, form_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
            
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.ViewCam)

        # set btnControl callback clicked  function
        self.btnControl.clicked.connect(self.ControlTimer)
        self.btnTrain.clicked.connect(self.Train)

        self.tabWidget.addTab(Tab2(),"Phát hiện và nhận dạng")

    # view camera
    def ViewCam(self):
        global count
        count = count + 1
        folderName = str(self.txtName.toPlainText())
        folderPath = "./Train/train_img/"+ folderName + "/"
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #print(img2)
        if(count <= 100):
            cv2.imwrite(folderPath + str(count) + ".png", img2)
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.camera.setPixmap(QPixmap.fromImage(qImg))
        else:
            self.camera.setText("Hoàn thành!")
            count = 0
            self.timer.stop()
            self.cap.release()
            self.btnControl.setText("Bắt đầu lấy thông tin")
    # start/stop timer
    def ControlTimer(self):
        # if timer is stopped
        global count
        TEN = str(self.txtName.toPlainText())
        KHOAHOC = str(self.txtKhoaHoc.toPlainText())
        if not self.timer.isActive():
            # create video capture
            if(TEN != "" and KHOAHOC != ""):
                kt = Insert(TEN, KHOAHOC, self)
                if(kt == 1):
                    self.cap = cv2.VideoCapture(1)
                    # start timer
                    self.timer.start(50)
                    # update btnControl text
                    self.btnControl.setText("Dừng")
                    self.lbTrain.setText("Loading... ")
            else:   
                QMessageBox.about(self, "Error","Bạn chưa nhập tên và khóa học")
                self.btnControl.setText("Bắt đầu lấy thông tin")
        # if timer is started
        else:
            count = 0
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update btnControl text
            self.btnControl.setText("Bắt đầu lấy thông tin")
        
    def Train(self):
        input_datadir = './Train/train_img'
        output_datadir = './Train/pre_img'

        obj=preprocesses(input_datadir,output_datadir)
        nrof_images_total,nrof_successfully_aligned=obj.collect_data()

        self.lbTrain.setText('Total number of images: %d' % nrof_images_total)
        self.lbTrain.setText('Number of successfully aligned images: %d' % nrof_successfully_aligned)

        datadir = './Train/pre_img'
        modeldir = './Train/model/20170511-185253.pb'
        classifier_filename = './Train/class/classifier.pkl'
        self.lbTrain.setText("Train Start")
        obj=training(datadir,modeldir,classifier_filename)
        get_file=obj.main_train()
        self.lbTrain.setText('Saved classifier model to file "%s"' % get_file)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # create and show mainWindow
    mainWindow = MyWindowClass(None)
    mainWindow.show()

    app.exec_()