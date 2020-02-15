from Detection.tiny_face_eval import TinyFace
import Train.identifyImage as identifyImage
import Train.identify_face_video as identify_face_video

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QTableWidget, QTableWidgetItem, QFileDialog, QTextEdit, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5 import uic
import cv2
import sqlite3
import os
import sys
from Detection.kaka import test
import tensorflow as tf
import Train.facenet as facenet
import Train.detect_face as detect_face
import pickle
import numpy as np

tab2 = uic.loadUiType("Tab2.ui")[0]
pathImg = ""
Img = []
modeldir = './Train/model/20170511-185253.pb'
classifier_filename = './Train/class/classifier.pkl'
npy= "./Train/npy"
train_img= "./Train/train_img/"
danhdau = []
Trace = []
comat = 0
tong = 0
check = 0
for i in range(100):
    danhdau.append(0)
    Trace.append(0)

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        HumanNames = os.listdir(train_img)
        #HumanNames.sort()

        print('Loading Modal')
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

def ResetTable(self):
    global comat
    global tong
    x = "0"
    comat = 0 
    self.lbCoMat.setText(x)
    self.lbVangMat.setText(x)
    self.lbUnknow.setText(x)
    self.lbFaceClose.setText(x)
    connect = sqlite3.connect("database.db")
    cmd = "select TEN,KHOAHOC from SINHVIEN"
    cmd1 = "select count(*) from SINHVIEN"
    tong = connect.execute(cmd1).fetchone()[0]
    cusor = connect.execute(cmd)
    self.tableWidget.setRowCount(0)
    for row_number, row_data in enumerate(cusor):
        self.tableWidget.insertRow(row_number)
        #khoi tao ion delete
        number = QTableWidgetItem(QIcon("iconDelete.png"),"")
        self.tableWidget.setCurrentCell(row_number,2)
        self.tableWidget.setItem(row_number, 2, number)
        for column_number, column_data in enumerate(row_data):
            self.tableWidget.setItem(row_number, column_number, QTableWidgetItem(str(column_data)))           
    connect.close()

def CheckTable(self, name, predict):
    global comat
    global tong
    connect = sqlite3.connect("database.db")
    cmd = "select STT from SINHVIEN where TEN = '" + name + "'" 
    cursor = connect.execute(cmd)
    isRecordExist = 0
    for row in cursor:                                                          
        isRecordExist = 1
    if isRecordExist == 1: 
        stt = connect.execute(cmd).fetchone()[0]
    
        number = QTableWidgetItem(QIcon("icon.png"),"")
        self.tableWidget.setCurrentCell(stt,2)
        self.tableWidget.setItem(stt, 2, number)

        number = QTableWidgetItem(str(predict))
        self.tableWidget.setCurrentCell(stt,3)
        self.tableWidget.setItem(stt, 3, number)
        if(check == 3):
            Trace[stt] = identify_face_video.img_crop      
        if(check == 2):
            Trace[stt] = identifyImage.img_crop 
        if(danhdau[stt] == 0):
            danhdau[stt] = 1
            comat = comat + 1
            self.lbCoMat.setText(str(comat))
            self.lbVangMat.setText(str(tong - comat))

    connect.close()

def LoadLbImg(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channel = image.shape
    step = channel * width
    # create QImage from image
    qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qImg)
    pixmap_image = QPixmap(pixmap)
    return pixmap_image

class Tab2(QWidget, tab2):
    def __init__(self, parent = None):
        super(Tab2, self).__init__(parent)
        self.setupUi(self)
        ResetTable(self)

        self.lbAnh.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbAnh.setScaledContents(True)
        self.lbAnhPhatHien.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbAnhPhatHien.setScaledContents(True)
        self.lbAnhGoc.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbAnhGoc.setScaledContents(True)

        self.btnDialog.clicked.connect(self.openDialog)
        self.btnTiny.clicked.connect(self.TinyImage)
        
        self.btnFace.clicked.connect(self.FaceImg)
        self.tableWidget.clicked.connect(self.DisplayImg)
        self.btnChup.clicked.connect(self.TakeImg)
        self.btnReset.clicked.connect(self.Reset)

        currentTime = QDateTime.currentDateTime()
        self.dtNow.setDateTime(currentTime)

        self.Name = QTextEdit()
        self.Name.setPlainText("")
        self.Predict = QTextEdit()
        self.Predict.setPlainText("")
        self.Name.textChanged.connect(self.ChangeTable)

        self.timer = QTimer()
        self.timer.timeout.connect(self.ViewCam)
        self.btnFaceCamera.clicked.connect(self.ControlCamera)
    
    def Reset(self):
        global pathImg
        pathImg = ""
        self.lbAnh.setText("Loading ... ")
        self.lbAnhGoc.setText("...")
        self.lbAnhPhatHien.setText("...")
        ResetTable(self)
        self.timer.stop()
        #self.cap.release()
        self.btnFaceCamera.setText("Nhận dạng qua Camera")
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()

    def openDialog(self):
        global pathImg
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files ('*.png', '*.gif', '*.jpg', '*.jpeg')", options=options)
        self.lbAnh.setText("")
        if(fileName):
            image = cv2.imread(fileName)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            pixmap_image = QPixmap(pixmap)
            self.lbAnh.setPixmap(pixmap_image)
            pathImg = fileName

    def TinyImage(self):
        global check
        check = 1
        global pathImg
        if(pathImg != ""):
            tiny = TinyFace(pathImg)
        else:
            QMessageBox.about(self, "Error","Chưa có ảnh đầu vào")
            self.btnDialog.click()

    def FaceImg(self):
        global check
        check = 2
        global pathImg
        if(pathImg != ""):
            facenet.load_model(modeldir)
            identifyImage.identify(self, pathImg,pnet,rnet,onet,minsize,threshold,factor,margin,frame_interval,batch_size,image_size,input_image_size,HumanNames,images_placeholder,embeddings,phase_train_placeholder,embedding_size,model,sess)
        else:
            QMessageBox.about(self, "Error","Chưa có ảnh đầu vào")
            self.btnDialog.click()

    def ViewCam(self):
        global check
        check = 3
        self.lbAnh.setText("Loading ... ")
        identify_face_video.CameraNet(self,pnet,rnet,onet,minsize,threshold,factor,margin,frame_interval,batch_size,image_size,input_image_size,HumanNames,images_placeholder,embeddings,phase_train_placeholder, embedding_size, model, sess)

    def ChangeTable(self):
        CheckTable(self, self.Name.toPlainText(), self.Predict.toPlainText())

    def ControlCamera(self):
        if not self.timer.isActive(): 
            facenet.load_model(modeldir)           
            self.cap = cv2.VideoCapture(0)
            self.timer.start(100)
            self.btnFaceCamera.setText("Dừng nhận dạng")
        else:
            self.timer.stop()
            self.cap.release()
            self.btnFaceCamera.setText("Nhận dạng qua Camera")

    def DisplayImg(self):
        row = self.tableWidget.currentRow()
        col = 0

        name = self.tableWidget.item(row, col).text()
        check = ""
        try:
            check = self.tableWidget.item(row, 3).text()
        except:
            print("error")
        pathImg = "./Train/train_img/" + str(name) + "/1.png"

        image = cv2.imread(pathImg)
        pixmap_image = LoadLbImg(image)
        self.lbAnhGoc.setPixmap(pixmap_image)

        # Hien thi anh phat hien dk 
        if(check != ""):
            image = Trace[row]
            pixmap_image = LoadLbImg(image)
            self.lbAnhPhatHien.setPixmap(pixmap_image)
        else:
            self.lbAnhPhatHien.setText("...")
    
    def TakeImg(self):
        if not self.timer.isActive(): 
            facenet.load_model(modeldir)           
            self.cap = cv2.VideoCapture(1)
            self.timer.start(100)
            self.btnChup.setText("Dừng nhận dạng")
        else:
            self.timer.stop()
            self.cap.release()
            self.btnChup.setText("Camera qua dien thoai")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = Tab2(None)
    main.show()
    
    app.exec_()