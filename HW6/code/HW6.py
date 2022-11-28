import sys
import cv2
import math
import matplotlib as plt
import numpy as np
import copy
import pywt
from PyQt5 import QtGui, QtWidgets,QtCore
from HW6layout import Ui_Form
from PyQt5.QtCore import Qt
class HW6(QtWidgets.QDialog):
  grayTable = cv2.imread('grayTable.png', 1)
  def __init__(self):
    super().__init__()
    self.ui = Ui_Form()
    self.ui.setupUi(self)
    self.ui.load.clicked.connect(self.loadImage)
    self.ui.p2Load.clicked.connect(self.loadManyImage)
    self.ui.p1Confirm.clicked.connect(self.problem1Confirm)
    self.ui.p2Confirm.clicked.connect(self.problem2Confirm)
    self.ui.p3Confirm.clicked.connect(self.problem3Confirm)
    self.show()
  def loadImage(self):
    # self.refreshImage()
    filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open Image', 'Image', '*.png *.jpg *.bmp')
    if filename == '':
        return
    # Load color image
    self.image = cv2.imread(filename, 1)
    self.image = self.image[:,:,::-1]
    self.showImage(self.image,self.ui.originalImage)
  def loadManyImage(self):
    filename, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Open Image', 'Image', '*.png *.jpg *.bmp')
    if filename == '':
        return

    self.img = []

    # Load color image
    for i in range(len(filename)):
        self.img.append(cv2.cvtColor(cv2.imread(filename[i], 1), cv2.COLOR_BGR2RGB))
    self.refreshImage(self.ui.modifiedImage1)
    self.refreshImage(self.ui.modifiedImage1_3)
    self.refreshImage(self.ui.modifiedImage2)
    self.showImage(self.img[0],self.ui.originalImage)
    if len(self.img) >= 2:
      self.showImage(self.img[1],self.ui.modifiedImage1)
    if len(self.img) >= 3:
      self.showImage(self.img[2],self.ui.modifiedImage1_3)
  def showImage(self,imageToShow,block):
    if len(imageToShow.shape) == 3:
      height,width ,color = imageToShow.shape
      outputImage = QtGui.QImage(imageToShow.astype(
          np.uint8), width, height, 3 * width, QtGui.QImage.Format_RGB888)
    else:
      height,width = imageToShow.shape
      outputImage = QtGui.QImage(imageToShow.astype(
          np.uint8), width, height,width, QtGui.QImage.Format_Grayscale8)
    pixmap = QtGui.QPixmap(outputImage)
    # width = 380 , height= 380
    pixmap = pixmap.scaled(380, 380, aspectRatioMode=Qt.KeepAspectRatio)
    block.setPixmap(pixmap)
  def problem1Confirm(self):
    self.trapezoidalTransform()
    self.wavyTransform()
    self.circularTransform()
    
  def trapezoidalTransform(self):
    x_offset = int(self.ui.p1_x.text())
    y_offset = int(self.ui.p1_y.text())
    img_gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
    rows,cols = img_gray.shape
    input_pts = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
    output_pts = np.float32([[0,0],[cols,0],[x_offset,rows-y_offset],[cols-x_offset,rows-y_offset]])
    transform_matrix = cv2.getPerspectiveTransform(input_pts,output_pts)
    img_gray = cv2.warpPerspective(img_gray,transform_matrix,(cols,rows))
    self.showImage(img_gray,self.ui.modifiedImage1)
  def wavyTransform(self):
    # https://subscription.packtpub.com/book/application-development/9781788396905/1/ch01lvl1sec19/image-warping
    img_gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
    rows,cols = img_gray.shape
    img_show = np.zeros(img_gray.shape, dtype=img_gray.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(25.0 * math.sin(2 * 3.14 * i / 150))
            offset_y = int(25.0 * math.cos(2 * 3.14 * j / 150))
            if i+offset_y < rows and j+offset_x < cols:
                img_show[i,j] = img_gray[(i+offset_y)%rows,(j+offset_x)%cols]
            else:
                img_show[i,j] = 0
    self.showImage(img_show,self.ui.modifiedImage2)
  def circularTransform(self):
    img_gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
    rows,cols = img_gray.shape
    img_show = np.zeros(img_gray.shape, dtype=img_gray.dtype)
    for i in range(rows):
      for j in range(cols):
        norm_x = (i-rows/2)/(rows/2)
        norm_y = (j-cols/2)/(cols/2)
        offset_x = int(norm_x*math.sqrt(1-(norm_y**2)/2)*(rows/2)+(rows/2))
        offset_y = int(norm_y*math.sqrt(1-(norm_x**2)/2)*(cols/2)+(cols/2))
        img_show[offset_x,offset_y] = img_gray[i,j]
    self.showImage(img_show,self.ui.modifiedImage3)
  def problem2Confirm(self):
    img_gray = copy.deepcopy(self.img)
    for img in img_gray:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for i in range(len(img_gray)-1):
      if i == 0:
        img_fused = self.fusion(img_gray[0], img_gray[1])
      else:
        img_fused = self.fusion(img_fused, img_gray[i+1])
    img_fused = cv2.cvtColor(img_fused, cv2.COLOR_RGB2GRAY)

    self.showImage(img_fused, self.ui.modifiedImage2)
  def fusion(self,img1, img2):
    ## Separate channels
    iR1 = img1.copy()
    iR1[:,:,1] = iR1[:,:,2] = 0
    iR2 = img2.copy()
    iR2[:,:,1] = iR2[:,:,2] = 0

    iG1 = img1.copy()
    iG1[:,:,0] = iG1[:,:,2] = 0
    iG2 = img2.copy()
    iG2[:,:,0] = iG2[:,:,2] = 0

    iB1 = img1.copy()
    iB1[:,:,0] = iB1[:,:,1] = 0
    iB2 = img2.copy()
    iB2[:,:,0] = iB2[:,:,1] = 0

    shape = (img1.shape[1], img1.shape[0])
    # Wavelet transformation on red channel
    outImageR = self.channelTransform(iR1, iR2, shape)
    outImageG = self.channelTransform(iG1, iG2, shape)
    outImageB = self.channelTransform(iB1, iB2, shape)

    outImage = img1.copy()
    outImage[:,:,0] = outImage[:,:,1] = outImage[:,:,2] = 0
    outImage[:,:,0] = outImageR[:,:,0]
    outImage[:,:,1] = outImageG[:,:,1]
    outImage[:,:,2] = outImageB[:,:,2]

    outImage = np.multiply(np.divide(outImage - np.min(outImage),(np.max(outImage) - np.min(outImage))),255)
    outImage = outImage.astype(np.uint8)

    return outImage
  def fuse_method(self,n1, n2, method):
    if (method == 'mean'):
        n = (n1 + n2) / 2
    elif (method == 'min'):
        n = np.minimum(n1,n2)
    elif (method == 'max'):
        n = np.maximum(n1,n2)
    return n

  def channelTransform(self,ch1, ch2, shape):
    cooef1 = pywt.wavedec2(ch1, 'db5', level=1, mode = 'periodization')
    cooef2 = pywt.wavedec2(ch2, 'db5', level=1, mode = 'periodization')
    cA1, (cH1, cV1, cD1) = cooef1
    cA2, (cH2, cV2, cD2) = cooef2

    cA = self.fuse_method(cA1, cA2, 'mean')
    cH = self.fuse_method(cH1, cH2, 'max')
    cV = self.fuse_method(cV1, cV2, 'max')
    cD = self.fuse_method(cD1, cD2, 'max')
    fincoC = cA, (cH,cV,cD)
    outImageC = pywt.waverec2(fincoC, 'db5', mode = 'periodization')
    outImageC = cv2.resize(outImageC,(shape[0],shape[1]))
    return outImageC
  def problem3Confirm(self):
    self.refreshImage(self.ui.modifiedImage1)
    self.refreshImage(self.ui.modifiedImage1_3)
    self.refreshImage(self.ui.modifiedImage2)
    img_copy = self.image.copy()
    rows, cols, colors=img_copy.shape
    img_canny = cv2.Canny(img_copy,100,200)
    lines = cv2.HoughLinesP(img_canny, 1, np.pi/180, 30, minLineLength=60, maxLineGap=20)
    lines1 = lines[:, 0, :]

    img_hough = np.zeros((rows,cols), np.uint8)
    for x1, y1, x2, y2 in lines1[:]:
      cv2.line(img_hough, (x1,y1), (x2,y2), 255, 5)

    contours, hierarchy = cv2.findContours(img_hough.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros((rows,cols), np.uint8)
    cv2.drawContours(img_contours, [contours[0]], -1, 255, -1)

    area = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], closed=True)
    self.showImage(img_hough,self.ui.modifiedImage2)
    self.showImage(img_contours,self.ui.modifiedImage3)
    self.ui.p3Areas.setText(QtCore.QCoreApplication.translate("Form",str(0.5*0.5*area)))
    self.ui.p3Perimeters.setText(QtCore.QCoreApplication.translate("Form",str(0.5*perimeter)))
  
  def refreshImage(self,block):
      imageCopy = self.grayTable.copy().astype(np.float64)
      self.showImage(imageCopy-imageCopy+236, block)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = HW6()
    ui.show()
    sys.exit(app.exec_())