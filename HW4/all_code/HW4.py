import sys
import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets
from HW4layout import Ui_Form
from PyQt5.QtCore import Qt


class HW4(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.load.clicked.connect(self.loadImage)
        self.ui.p1Confirm.clicked.connect(self.FFT)
        self.ui.p2Confirm.clicked.connect(self.filtering)
        self.ui.p3Confirm.clicked.connect(self.homomorphic)
        self.ui.p4Confirm.clicked.connect(self.motionBlurredImage)
        self.show()

    def loadImage(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if filename == '':
            return
        self.image = cv2.imread(filename, 0)
        self.showGrayImage(self.image, self.ui.originalImage)
        self.refreshImage()

    def convertTo0_255(self, inputImage):
        returnImage = inputImage.copy()
        returnImage = np.round(
            (returnImage - returnImage.min()) / (returnImage.max() - returnImage.min()) * 255)
        return returnImage.astype(np.uint8)

    def showGrayImage(self, inputImage, block):
        imageToShow = self.convertTo0_255(inputImage.copy())
        height, width = imageToShow.shape
        outputImage = QtGui.QImage(
            imageToShow, width, height, width, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(outputImage)
        pixmap = pixmap.scaled(250, 250, aspectRatioMode=Qt.KeepAspectRatio)
        block.setPixmap(pixmap)

    # Refresh showing image
    def refreshImage(self):
        img_empty = np.array([[1]])
        # self.showGrayImage(img_empty, self.ui.originalImage)
        self.showGrayImage(img_empty, self.ui.modifiedImage1)
        self.showGrayImage(img_empty, self.ui.modifiedImage2)
        self.showGrayImage(img_empty, self.ui.modifiedImage3)
        self.showGrayImage(img_empty, self.ui.modifiedImage4)
        self.showGrayImage(img_empty, self.ui.modifiedImage5)
        self.showGrayImage(img_empty, self.ui.modifiedImage6)
        self.showGrayImage(img_empty, self.ui.modifiedImage7)
        self.showGrayImage(img_empty, self.ui.modifiedImage8)
    #part1
    def FFT(self):
        self.refreshImage()
        imageCopy = self.image.copy().astype(np.float64)
        # Fourier transform
        imageShift = np.fft.fft2(imageCopy)
        imageShift = np.fft.fftshift(imageShift)
        # convert to image spectrum
        spectrum = np.log(1+np.abs(imageShift))
        Fmin = np.log(1+np.abs(spectrum.min()))
        Fmax = np.log(1+np.abs(spectrum.max()))
        # count spectrum and angle
        spectrum = 255*(spectrum-Fmin)/(Fmax-Fmin)
        angle = np.arctan(np.imag(imageShift)/np.real(imageShift))
        self.showGrayImage(spectrum, self.ui.modifiedImage1)
        self.showGrayImage(angle, self.ui.modifiedImage2)
        # inverse Fourier transform
        imageInverseShift = np.fft.ifftshift(imageShift)
        imageInverseBack = np.fft.ifft2(imageInverseShift)
        imageInverseBack = np.abs(imageInverseBack)
        imageDifferent = imageCopy - imageInverseBack
        self.showGrayImage(imageInverseBack, self.ui.modifiedImage3)
        self.showGrayImage(imageDifferent, self.ui.modifiedImage6)  
    #part2
    def filtering(self):
        self.refreshImage()
        cutoff = float(self.ui.p2Cutoff.text())
        n = float(self.ui.p2nValue.text())
        imageCopy = self.image.copy().astype(np.float64)
        self.showGrayImage(self.idealLowPass(imageCopy, cutoff), self.ui.modifiedImage3)
        self.showGrayImage(self.idealHighPass(imageCopy, cutoff), self.ui.modifiedImage6)
        self.showGrayImage(self.ButterWorthLowPass(imageCopy, cutoff,n), self.ui.modifiedImage4)
        self.showGrayImage(self.ButterWorthHighPass(imageCopy, cutoff,n), self.ui.modifiedImage7)
        self.showGrayImage(self.gaussianHighPass(imageCopy, cutoff), self.ui.modifiedImage8)
        self.showGrayImage(self.gaussianLowPass(imageCopy, cutoff), self.ui.modifiedImage5)

    def gaussianLowPass(self, inputImage, cutoff):
        def filter(dist):
            return np.exp(-dist / (2*(cutoff**2)))
        return self.distFourierFilterOperations(inputImage, filter)
    def gaussianHighPass(self,inputImage,cutoff):
        def filter(dist):
            return 1 - np.exp(-(dist / (2*(cutoff**2))))
        return self.distFourierFilterOperations(inputImage, filter)
    def ButterWorthHighPass(self,inputImage,cutoff,n):
        def filter(dist):
            return 1 / (1 + ((cutoff**(2*n))/(dist**n)))
        return self.distFourierFilterOperations(inputImage, filter)
    def ButterWorthLowPass(self,inputImage,cutoff,n):
        def filter(dist):
            return 1/(1+((dist**n)/(cutoff**(2*n))))
        return self.distFourierFilterOperations(inputImage, filter)
    def idealHighPass(self,inputImage,cutoff):
        def filter(dist):
            return dist > cutoff ** 2
        return self.distFourierFilterOperations(inputImage, filter)
    def idealLowPass(self, inputImage, cutoff):
        def filter(dist):
            return dist <= cutoff ** 2
        return self.distFourierFilterOperations(inputImage, filter)

    def distFourierFilterOperations(self, inputImage, func):
        def filter(i, j):
            # distance to the power of two
            return func(i**2 + j**2)
        return self.fourierFilterOperations(inputImage, filter)

    def fourierFilterOperations(self, inputImage, func):
        # Create filter
        s = inputImage.shape
        j, i = np.meshgrid(np.arange(s[1]) - s[1] // 2,
                           np.arange(s[0]) - s[0] // 2)
        j = j.astype(np.float64)
        i = i.astype(np.float64)
        j[j == 0] = 1
        i[i == 0] = 1
        filter = func(i, j)

        # Fourier transformation
        f = np.fft.fft2(inputImage)
        imageShift = np.fft.fftshift(f)

        # Convolution
        filterImage = filter * imageShift

        # Inverse fourier transformation
        shifted = np.fft.ifftshift(filterImage)
        returnImage = np.fft.ifft2(shifted)
        returnImage = np.abs(returnImage)
        return returnImage
    #part3
    def homomorphic(self):
        self.refreshImage()
        cutoff = float(self.ui.p3Cutoff.text())
        gammaH = float(self.ui.p3GammaH.text())
        gammaL = float(self.ui.p3GammaL.text())
        c = float(self.ui.p3cValue.text())
        imageCopy = self.image.copy().astype(np.float64)
        imageCopy = np.log(imageCopy+1)
        def filter(dist):
            return (gammaH-gammaL) * (1 - np.exp(-c*dist/(cutoff)**2)) + gammaL
        returnImage = np.exp(self.distFourierFilterOperations(imageCopy, filter))
        self.showGrayImage(returnImage,self.ui.modifiedImage1)
    
    #part4
    def motionBlurredImage(self):
        self.refreshImage()
        constA = float(self.ui.p4constantA.text()) # 0.1
        constB = float(self.ui.p4constantB.text()) # 0.2
        constT = float(self.ui.p4constantT.text())  # 1
        constK = float(self.ui.p4constantK.text())
        imageCopy = self.image.copy().astype(np.float64)
        #add noise
        noise = np.random.normal(0,20,imageCopy.shape)
        noisedImage = noise + imageCopy
        blurredImage = self.motionDegradation(noisedImage, constA, constB, constK)
        inverse2DImage = self.inverse2D(blurredImage,constA,constB,constT)
        subtractedImage = imageCopy-self.convertTo0_255(inverse2DImage)
        inverseWienerImage = self.wienerInverse(blurredImage, constA, constB, constT,constK)
        subtractedWienerImage = imageCopy - self.convertTo0_255(inverseWienerImage)
        self.showGrayImage(blurredImage, self.ui.modifiedImage1)
        self.showGrayImage(inverse2DImage, self.ui.modifiedImage3)
        self.showGrayImage(inverseWienerImage, self.ui.modifiedImage6)
        self.showGrayImage(subtractedImage, self.ui.modifiedImage4)
        self.showGrayImage(subtractedWienerImage, self.ui.modifiedImage7)
    
    def motionEffect(self,a, b, T):

        def filter(u, v):
            uv = (u*a + v*b) * np.pi
            H = T / uv * np.sin(uv) * np.exp(-1j * uv)
            H[np.abs(uv) < 1e-6] = 0
            return H
        return filter
    def wienerInverse(self, inputImage, a, b, T, K):
        def wienerFilter(K, formula):
            def filter(i, j):
                forward_filter = formula(i, j)
                return forward_filter.conj() / (np.abs(forward_filter) ** 2 + K)
            return filter
        return self.fourierFilterOperations(inputImage, wienerFilter(K, self.motionEffect(a, b, T)))
    def inverse2D(self,inputImage, a, b, T):
        def invFilter(formula):
            def filter(i, j):
                forward_filter = formula(i, j)
                forward_filter[np.abs(forward_filter) < 1e-6] = np.inf
                return forward_filter ** -1
            return filter

        return self.fourierFilterOperations(inputImage, invFilter(self.motionEffect(a, b, T)))

    def motionDegradation(self,inputImage, a, b, T):
        def motionEffect(a, b, T):
            def filter(u, v):
                uv = (u*a + v*b) * np.pi
                H = T / uv * np.sin(uv) * np.exp(-1j * uv)
                H[np.abs(uv) < 1e-6] = 0
                return H
            return filter
        return self.fourierFilterOperations(inputImage, motionEffect(a, b, T))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = HW4()
    ui.show()
    sys.exit(app.exec_())
