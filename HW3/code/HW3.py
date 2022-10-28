import sys
import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets
from HW3layout import Ui_Form
from PyQt5.QtCore import Qt


class HW3(QtWidgets.QDialog):
  def __init__(self):
    super().__init__()
    self.ui = Ui_Form()
    self.ui.setupUi(self)
    self.ui.load.clicked.connect(self.loadImage)
    self.ui.p2Confirm.clicked.connect(self.maskOperation)
    self.ui.p3Confirm.clicked.connect(self.LoGOperation)
    self.ui.p4Confirm.clicked.connect(self.histogramEqualization)
    self.show()

  def loadImage(self):
    self.filename = QtWidgets.QFileDialog.getOpenFileName(
        filter="Image (*.*)")[0]
    # self.image = imutils.resize(self.image, width=500,height = 400)
    self.image = cv2.imread(self.filename, -1)
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    # print(self.image)
    self.showImage(self.image, self.ui.originalImage)

  def showImage(self, imageToShow, block):
    height, width, color = imageToShow.shape
    # width = 400
    # height= 500
    outputImage = QtGui.QImage(imageToShow.astype(
        np.uint8), width, height, 3 * width, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap(outputImage)
    pixmap = pixmap.scaled(300, 300, aspectRatioMode=Qt.KeepAspectRatio)
    block.setPixmap(pixmap)

  def showGrayImage(self, imageToShow, block):
    height, width = imageToShow.shape
    outputImage = QtGui.QImage(imageToShow.astype(
        np.uint8), width, height, width, QtGui.QImage.Format_Grayscale8)
    pixmap = QtGui.QPixmap(outputImage)
    pixmap = pixmap.scaled(300, 300, aspectRatioMode=Qt.KeepAspectRatio)
    block.setPixmap(pixmap)

  def maskOperation(self):  # problem2
    filterRow = int(self.ui.p2Row.text())
    filterColumn = int(self.ui.p2Column.text())
    coefficient = float(self.ui.p2Coefficients.text())
    filter = np.full(shape=(filterRow, filterColumn),
                     fill_value=coefficient, dtype=np.float)
    self.showImage(self.filteringImage(
        self.image, filter), self.ui.modifiedImageA)

  def LoGOperation(self):  # problem3
    threshold = int(self.ui.p3Threshold.text())
    if self.ui.p3LoG.isChecked():
      gaussianFilter = np.array(
          [[0.3679, 0.6065, 0.3679], [0.6065, 1.0, 0.6065], [0.3679, 0.6065, 0.3679]])
      gaussianFilter = gaussianFilter / 4.8976
      tempImage = self.filteringImage(
          self.image.astype(np.int16), gaussianFilter)
      laplacianMask = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
      tempImage = self.filteringImage(tempImage, laplacianMask)
      self.showImage(self.zeroCrossing(
          tempImage, threshold), self.ui.modifiedImageA)
      sovbelOperator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
      self.showImage(self.filteringImage(self.image.astype(
          np.int16), sovbelOperator), self.ui.modifiedImageB)

  # return a new image with filtering operation
  def filteringImage(self, imageToConvert, filter):
    rowPadding = int(filter.shape[0]/2)
    columnPadding = int(filter.shape[1]/2)
    paddingImage = np.pad(imageToConvert, ((rowPadding, rowPadding),
                          (columnPadding, columnPadding), (0, 0)), 'constant')  # zero padding
    # print(paddingImage.shape)
    outputImage = imageToConvert.copy()
    for y in range(outputImage.shape[0]):
      for x in range(outputImage.shape[1]):
        for z in range(outputImage.shape[2]):
          temp = 0
          for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
              # paddingX = x + rowPadding
              # paddingY = y + columnPadding
              temp += filter[i][j]*paddingImage[y+i][x+j][z]
          outputImage[y][x][z] = temp
    return outputImage

  def localEnhancement(self, imageToConvert, region):
    constC = float(self.ui.constantC.text())
    constK0 = float(self.ui.constantK1.text())
    constK1 = float(self.ui.constantK2_2.text())
    constK2 = float(self.ui.constantK2.text())
    constK3 = float(self.ui.constantK3.text())
    globalMean = np.mean(imageToConvert.ravel())
    globalStd = np.std(imageToConvert.ravel(), ddof=1)
    outputImage = imageToConvert.copy()
    paddingSize = int(region/2)
    paddingImage = np.pad(imageToConvert, ((
        paddingSize, paddingSize), (paddingSize, paddingSize)), 'constant')
    meanAndStd = np.zeros(
        shape=(imageToConvert.shape[0], imageToConvert.shape[1], 2))
    for y in range(meanAndStd.shape[0]):
      for x in range(meanAndStd.shape[1]):
          tempArray = np.zeros((region*region))
          n = 0
          for i in range(region):
            for j in range(region):
              # print(n)
              tempArray[n] = paddingImage[y+i][x+j]
              n += 1
          localMean = np.mean(tempArray)
          localStd = np.std(tempArray, ddof=1)
          if constK0*globalMean <= localMean and localMean <= constK1*globalMean:
            if constK2*globalStd <= localStd and localStd <= constK3 * globalStd:
              outputImage[y][x] *= constC
          # print(meanAndStd[y][x][0], meanAndStd[y][x][0])
    return outputImage

  def zeroCrossing(self, imageToConvert, threshold):
    outputImage = np.zeros(shape=imageToConvert.shape)
    for y in range(1, imageToConvert.shape[1]):
      if y > imageToConvert.shape[1] - 2:
          break
      for x in range(1, imageToConvert.shape[0]):
          if x > imageToConvert.shape[0] - 2:
              break
          for k in range(imageToConvert.shape[2]):
              for j in range(y-1, y+2):
                  for i in range(x-1, x+2):
                      if (((imageToConvert[x][y][k] >= threshold) & (imageToConvert[i][j][k] < threshold)) |
                              ((imageToConvert[x][y][k] < threshold) & (imageToConvert[i][j][k] >= threshold))):
                          outputImage[x][y][k] = imageToConvert[x][y][k]
    return outputImage

  def globalHistogramEq(self, inputArray):
    hist, bins = np.histogram(inputArray.ravel(), 256)
    pdf = hist/inputArray.size  # hist = 出現次數。出現次數/總像素點 = 概率 (pdf)
    cdf = pdf.cumsum()  # 將每一個灰度級的概率利用cumsum()累加，變成書中寫的「累積概率」(cdf)。
    # 將cdf的結果，乘以255 (255 = 灰度範圍的最大值) ，再四捨五入，得出「均衡化值(新的灰度級)」。
    equ_value = np.around(cdf * 255).astype('uint8')
    result = equ_value[inputArray]
    return result

  def localHistogramEq(self, inputArray, region):
    paddingRange = int(region/2)
    # print(paddingRange)
    paddingImage = np.pad(inputArray, ((
        paddingRange, paddingRange), (paddingRange, paddingRange)), 'constant')
    # print(paddingImage)
    outputImage = inputArray.copy()
    for y in range(outputImage.shape[0]):
      for x in range(outputImage.shape[1]):
          tempArray = np.zeros((region*region))
          n = 0
          for i in range(region):
            for j in range(region):
              tempArray[n] = paddingImage[y+i][x+j]
              n += 1
          hist, bins = np.histogram(tempArray, 256)
          pdf = hist/len(tempArray)
          cdf = pdf.cumsum()
          T = tempArray[int(region*region/2)]
          # print(T)
          # print(outputImage[y][x])
          outputImage[y][x] = np.around(cdf[int(T)]*255)
          # print(outputImage[y][x])
    return outputImage

  def histogramEqualization(self):
    region = int(self.ui.regionSize.text())
    output = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    outputLocalEnhancement = self.localEnhancement(output, region)
    outputGlobalHistogramEq = self.globalHistogramEq(output)
    outputLocalHistogramEq = self.localHistogramEq(output, region)
    self.showGrayImage(outputLocalEnhancement, self.ui.modifiedImageA)
    self.showGrayImage(outputGlobalHistogramEq, self.ui.modifiedImageC)
    self.showGrayImage(outputLocalHistogramEq, self.ui.modifiedImageB)

  def calculateList(self, imageToCovert):
    listA = [0] * 256
    tempList = imageToCovert.ravel()
    # print(tempList)
    for i in tempList:
        listA[i] += 1
    return listA


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = HW3()
    ui.show()
    sys.exit(app.exec_())
