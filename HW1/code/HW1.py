import numpy as np
from matplotlib import pyplot as plt


def grayTransfer(imageName, imageList):
  with open(imageName, "r") as file:
    for line in file:
      if line != '\x1a':
        imageList.extend(line.rstrip("\n\x1a"))
  for i in range(len(imageList)):
    if imageList[i].isalpha():
      imageList[i] = str(ord(imageList[i].lower()) - 87)


def addConst(imageList, const):
  newImage = []
  for i in range(len(imageList)):
    if imageList[i]+const <= 31:
      if imageList[i] + const >= 0:
        newImage.append(imageList[i] + const)
      else:
        newImage.append(0)
    else:
      newImage.append(31)
  return newImage


def multiplyConst(imageList, const):
  newImage = []
  for i in range(len(imageList)):
    if imageList[i]*const <= 31:
      if imageList[i] * const >= 0:
        newImage.append(imageList[i] * const)
      else:
        newImage.append(0)
    else:
      newImage.append(31)
  return newImage


def avgTwo(imageList1, imageList2):
  return (imageList1+imageList2)/2


def modifyPixel(imageList):
  newImage = []
  for i in range(len(imageList)):
    if imageList[i] - imageList[i-1] >= 0:
      newImage.append(imageList[i] - imageList[i-1])
    else:
      newImage.append(0)
  return newImage

imagesName = ['JET.64', 'LIBERTY.64', 'LINCOLN.64', 'LISA.64']
num = 1
for image in imagesName:
  imageList = []
  addTen = []
  minusEight = []
  multiplyFive = []
  multiplyMinusTen = []
  multiplyMinusOneFifth = []
  modifyImage = []
  grayTransfer(image, imageList)
  imageList = np.array([int(x) for x in imageList])
  # origin version
  plt.figure(1, figsize=(12, 6))
  plt.suptitle("origin version")
  plt.subplot(2, 4, num).imshow(imageList.reshape(64, 64), cmap='gray')
  plt.xlabel(image)
  plt.subplot(2, 4, num+1).hist(imageList, bins=32, color="black")
  # add value = 10 to each pixel
  addTen = np.array([int(x) for x in addConst(imageList, 10)])
  plt.figure(2, figsize=(12, 6))
  plt.suptitle("add ten to each pixel version")
  plt.subplot(2, 4, num).imshow(addTen.reshape(64, 64), cmap='gray')
  plt.xlabel(image)
  plt.subplot(2, 4, num+1).hist(addTen, bins=32, color="black")
  # add value = -8 to each pixel
  minusEight = np.array([int(x) for x in addConst(imageList, -8)])
  plt.figure(3, figsize=(12, 6))
  plt.suptitle("minus eight to each pixel version")
  plt.subplot(2, 4, num).imshow(minusEight.reshape(64, 64), cmap='gray')
  plt.xlabel(image)
  plt.subplot(2, 4, num+1).hist(minusEight, bins=32, color="black")
  # multiply value = 5 to each pixel
  multiplyFive = np.array([int(x) for x in multiplyConst(imageList, 5)])
  plt.figure(4, figsize=(12, 6))
  plt.suptitle("multiply five to each pixel version")
  plt.subplot(2, 4, num).imshow(multiplyFive.reshape(64, 64), cmap='gray')
  plt.xlabel(image)
  plt.subplot(2, 4, num+1).hist(multiplyFive, bins=32, color="black")
  # multiply value = -10 to each pixel
  multiplyMinusTen = np.array([int(x) for x in multiplyConst(imageList, -10)])
  plt.figure(5, figsize=(12, 6))
  plt.suptitle("multiply -10 to each pixel version")
  plt.subplot(2, 4, num).imshow(multiplyMinusTen.reshape(64, 64), cmap='gray')
  plt.xlabel(image)
  plt.subplot(2, 4, num+1).hist(multiplyMinusTen, bins=32, color="black")
  # multiply value = 1/5 to each pixel
  multiplyMinusOneFifth = np.array([int(x) for x in multiplyConst(imageList, 1/5)])
  plt.figure(6, figsize=(12, 6))
  plt.suptitle("multiply one fifth to each pixel version")
  plt.subplot(2, 4, num).imshow(multiplyMinusOneFifth.reshape(64, 64), cmap='gray')
  plt.xlabel(image)
  plt.subplot(2, 4, num+1).hist(multiplyMinusOneFifth, bins=32, color="black")
  #change pixel by equation
  modifyImage = np.array([int(x) for x in modifyPixel(imageList)])
  plt.figure(8, figsize=(12, 6))
  plt.suptitle("change each pixel by a equation version")
  plt.subplot(2, 4, num).imshow(modifyImage.reshape(64, 64), cmap='gray')
  plt.xlabel(image)
  plt.subplot(2, 4, num+1).hist(modifyImage, bins=32, color="black")
  num += 2

#Create a new image which is the average image of two input images
imageList1 = []
imageList2 = []
grayTransfer('JET.64', imageList1)
grayTransfer('LISA.64', imageList2)
imageList1 = np.array([int(x) for x in imageList1])
imageList2 = np.array([int(x) for x in imageList2])
newImage = np.array([int(x) for x in avgTwo(imageList1, imageList2)])
plt.figure(7)
plt.suptitle("average image of two input images version")
plt.subplot(1, 2, 1).imshow(newImage.reshape(64, 64), cmap='gray')
plt.xlabel('JET.64 + LISA.64')
plt.subplot(1, 2, 2).hist(newImage, bins=32, color="black")
plt.show()