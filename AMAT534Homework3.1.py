# --------------------------------------------------------------------------- #
# Name: Kevin Johansmeyer                                                     #
# Course: AMAT-534: Data-Driven Modeling and Computation                      #
# Professor: Dr. Eric Forgoston                                               #
# Assignment: Homework #3 (PARTS 1-4)                                         #
# --------------------------------------------------------------------------- #

# Note: The code may take a few minutes to run. Many calculations (including
# SVD) are performed, and many plots are generated.

#%% PART 1:
# -- Create a figure showing a single image for each of the 36 individuals -- #
# Citation: https://stackoverflow.com/questions/35723865/read-a-pgm-file-in-python
# Citation: https://www.geeksforgeeks.org/matplotlib-pyplot-imread-in-python/
# Citation: https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly

import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Citation: https://datatofish.com/measure-time-to-run-python-script/
startTime = time.time()

firstPoseData = []

fig = plt.figure(figsize=(10, 10))
for i in range(1, 38): # 6*6 + 2 (an extra 1 since 14 is missing)
    if i < 10: # 'yaleB0{}' becomes 'yaleB01' when i = 1
        with open('C:/Users/kevin/Documents/Python/AMAT-534Homework3/CroppedYale/yaleB0{}/yaleB0{}_P00A+000E+00.pgm'.format(i,i), 'rb') as pgmf:
            img = plt.imread(pgmf)
            firstPoseData.append(img) # stores each array in a list
            fig.add_subplot(6,6,i)
            plt.title('{}'.format(i))
            plt.imshow(img,cmap='gray')
            plt.axis('off')
    elif 10 <= i <= 13: # 'yaleB{}' becomes 'yaleB10' when i = 10
        with open('C:/Users/kevin/Documents/Python/AMAT-534Homework3/CroppedYale/yaleB{}/yaleB{}_P00A+000E+00.pgm'.format(i,i), 'rb') as pgmf:
            img = plt.imread(pgmf)
            firstPoseData.append(img) # stores each array in a list
            fig.add_subplot(6,6,i)
            plt.title('{}'.format(i))
            plt.imshow(img,cmap='gray')
            plt.axis('off')
    elif i == 14:
        continue
    elif i > 14: # There's no 14 in the database for some reason
        with open('C:/Users/kevin/Documents/Python/AMAT-534Homework3/CroppedYale/yaleB{}/yaleB{}_P00A+000E+00.pgm'.format(i,i), 'rb') as pgmf:
            img = plt.imread(pgmf)
            firstPoseData.append(img) # stores each array in a list
            fig.add_subplot(6,6,i-1)
            plt.title('{}'.format(i-1))
            plt.imshow(img,cmap='gray')
            plt.axis('off')
            
plt.suptitle('First Facial Image for First 36 People In Dataset',fontsize=16)
plt.figure(1)

print('Number of images to construct X Matrix:',len(firstPoseData))

# ----- Create a figure showing all 64 images for a single individuals ------ #
# Citation: https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/

firstFaceData = []

# Change directory string 'yaleB01' to see images of different person (ex: 'yaleB02')
directory = 'C:/Users/kevin/Documents/Python/AMAT-534Homework3/CroppedYale/yaleB01/'
fig = plt.figure(figsize=(10, 10))
j = 1
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        img = plt.imread(f)
        firstFaceData.append(img) # stores each array in a list
        fig.add_subplot(8, 8, j)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
        j = j + 1

plt.suptitle('All 64 Facial Images for First Person In Dataset',fontsize=16)
plt.figure(2)

#%% PART 2:
# ---- Reshape Images, Compute and Subtract Average, Construct X Matrix ----- #

# Reshape each single pose matrix into a column vector:
imgHeight = firstPoseData[0].shape[0]
imgWidth = firstPoseData[0].shape[1]
colLength = imgWidth * imgHeight

firstPoseDataCols = []

sumFirstPoseDataCols = 0
for i in range(0,36):
     colVector = firstPoseData[i].reshape((colLength,))
     firstPoseDataCols.append(colVector)
     # Find sum of face columns in order to de-mean the data:
     sumFirstPoseDataCols = firstPoseDataCols[i] + sumFirstPoseDataCols

# Find average of face columns in order to de-mean the data: 
avgFirstPoseDataCols = sumFirstPoseDataCols/len(sumFirstPoseDataCols)

# De-mean each column:
demeanedCols = []
for i in range(0,36):
    singleDemeanedCol = firstPoseDataCols[i] - avgFirstPoseDataCols
    demeanedCols.append(singleDemeanedCol) # stores each array in a list

# Horizontally stack de-meaned colummns:
xMatrix = demeanedCols[0]
for i in range(1,36):
    xMatrix = np.column_stack((xMatrix,demeanedCols[i]))
    
#%% ---------------------- Finding PCA/SVD of Matrix X ---------------------- #

U, sigmaVals, vAdjoint = np.linalg.svd(xMatrix)


#%% ----------------- Plotting Average Face and EigenFaces ------------------ #

# Eigenfaces:
eigenFaces = []

fig = plt.figure(figsize=(10, 10))
for i in range(0,U[0].shape[0]):
    singleEigenFace = U[:,i].reshape((imgHeight,imgWidth)) # reshape column to original shape
    eigenFaces.append(singleEigenFace) # stores each array in a list
    if i < 36:
        fig.add_subplot(6, 6, i+1)
        plt.title('{}'.format(i+1))
        plt.imshow(singleEigenFace,cmap='gray')
        plt.axis('off')

plt.suptitle('First Thirty-six Eigenfaces',fontsize=16)
plt.figure(3)

fig = plt.figure(figsize=(10, 10))
singleEigenFace = U[:,0].reshape((imgHeight,imgWidth)) # reshape column to original shape
plt.imshow(255*singleEigenFace/np.amax(singleEigenFace),cmap='gray')
plt.axis('off')

plt.suptitle('Average of Thirty-six Faces',fontsize=16)
plt.figure(4)

#%% PART 3:
# --------------------- Testing on Remaining Two Images --------------------- #

# Importing the Images:
firstPoseDataTwo = []
j = 1

fig = plt.figure(figsize=(10, 10))
for i in range(38, 40):
    with open('C:/Users/kevin/Documents/Python/AMAT-534Homework3/CroppedYale/yaleB{}/yaleB{}_P00A+000E+00.pgm'.format(i,i), 'rb') as pgmf:
        img = plt.imread(pgmf)
        firstPoseDataTwo.append(img) # stores each array in a list
        fig.add_subplot(1,2,j)
        plt.title('{}'.format(i-1))
        plt.imshow(img,cmap='gray')
        plt.axis('off')
        j = j + 1

# plt.suptitle('Two Withheld Faces from Matrix X',fontsize=16)
plt.suptitle('Original Face and Projection of Face',fontsize=16)
plt.figure(5)

# # --------------------------- Projection of Face 37 ------------------------- #
# # Citation: https://github.com/kjohansmeyer/SVDImageCompression/blob/main/BWCompression.py

modes = [25,50,100,200,400,800,1600]
# modes = [1,2,3,5,10,15,25]

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(2, 4, 1)
plt.imshow(firstPoseDataTwo[0],cmap='gray')
plt.title('original')
plt.axis('off')
for i in range(0,len(modes)):
    truncU = U[:,:modes[i]]
    # x^~test = U^~ U^~* xtest
    colVectorTest0 = firstPoseDataTwo[0].reshape((colLength,))
    xTest0Col = truncU @ np.transpose(truncU) @ colVectorTest0
    xTest0Reshape = xTest0Col.reshape((imgHeight,imgWidth))
    fig.add_subplot(2, 4, i+2)
    plt.imshow(xTest0Reshape, cmap='gray')
    plt.axis('off')
    plt.title('r = {}'.format(modes[i]))
    

plt.figure(6)
plt.suptitle('Original Face and Facial Projections',fontsize=16)

# --------------------------- Projection of Face 38 ------------------------- #

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(2, 4, 1)
plt.imshow(firstPoseDataTwo[1],cmap='gray')
plt.title('original')
plt.axis('off')
for i in range(0,len(modes)):
    truncU = U[:,:modes[i]]
    # x^~test = U^~ U^~* xtest
    colVectorTest1 = firstPoseDataTwo[1].reshape((colLength,))
    xTest1Col = truncU @ np.transpose(truncU) @ colVectorTest1
    xTest1Reshape = xTest1Col.reshape((imgHeight,imgWidth))
    fig.add_subplot(2, 4, i+2)
    plt.imshow(xTest1Reshape, cmap='gray')
    plt.axis('off')
    plt.title('r = {}'.format(modes[i]))
    
plt.figure(7)
plt.suptitle('Original Face and Facial Projections',fontsize=16)

# #%% PART 4:
# # -------------------- Testing on Animal/Object Images ---------------------- #
## This second is commented out for a fair runtime comparison. Uncomment to produce figures.

# # Favorite animal:
# # Citation: https://github.com/kjohansmeyer/SVDImageCompression/blob/main/BWCompression.py
# favoriteAnimal = plt.imread('C:/Users/kevin/Documents/Python/AMAT-534Homework3/favoriteAnimal.png')

# M = img.shape[0] # number of pixels (height)
# N = img.shape[1] # number of pixels (width)

# animalBW = np.zeros((M,N))
        
# for m in range (0,M):
#     for n in range (0,N):
#         # RBG weights from NTSC: 0.2989 ∙ Red + 0.5870 ∙ Green + 0.1140 ∙ Blue
#         animalBW[m][n] = (0.2989)*favoriteAnimal[m][n][0] + (0.5870)*favoriteAnimal[m][n][1] + (0.1140)*favoriteAnimal[m][n][2]


# fig = plt.figure(figsize=(10, 10))
# fig.add_subplot(2, 4, 1)
# plt.imshow(animalBW,cmap='gray')
# plt.title('original')
# plt.axis('off')
# for i in range(0,len(modes)):
#     truncU = U[:,:modes[i]]
#     # x^~test = U^~ U^~* xtest
#     colVectorTest2 = animalBW.reshape((colLength,))
#     xTest2Col = truncU @ np.transpose(truncU) @ colVectorTest2
#     xTest2Reshape = xTest2Col.reshape((imgHeight,imgWidth))
#     fig.add_subplot(2, 4, i+2)
#     plt.imshow(xTest2Reshape, cmap='gray')
#     plt.axis('off')
#     plt.title('r = {}'.format(modes[i]))
    
# plt.suptitle('Facial Projection on Image of Dog',fontsize=16)
# plt.figure(8)


# # Favorite object:
# # Citation: https://github.com/kjohansmeyer/SVDImageCompression/blob/main/BWCompression.py
# favoriteObject = plt.imread('C:/Users/kevin/Documents/Python/AMAT-534Homework3/favoriteObject.png')

# M = img.shape[0] # number of pixels (height)
# N = img.shape[1] # number of pixels (width)

# objectBW = np.zeros((M,N))
        
# for m in range (0,M):
#     for n in range (0,N):
#         # RBG weights from NTSC: 0.2989 ∙ Red + 0.5870 ∙ Green + 0.1140 ∙ Blue
#         objectBW[m][n] = (0.2989)*favoriteObject[m][n][0] + (0.5870)*favoriteObject[m][n][1] + (0.1140)*favoriteObject[m][n][2]


# fig = plt.figure(figsize=(10, 10))
# fig.add_subplot(2, 4, 1)
# plt.imshow(objectBW,cmap='gray')
# plt.title('original')
# plt.axis('off')
# for i in range(0,len(modes)):
#     truncU = U[:,:modes[i]]
#     # x^~test = U^~ U^~* xtest
#     colVectorTest3 = objectBW.reshape((colLength,))
#     xTest3Col = truncU @ np.transpose(truncU) @ colVectorTest3
#     xTest3Reshape = xTest3Col.reshape((imgHeight,imgWidth))
#     fig.add_subplot(2, 4, i+2)
#     plt.imshow(xTest3Reshape, cmap='gray')
#     plt.axis('off')
#     plt.title('r = {}'.format(modes[i]))

# plt.suptitle('Facial Projection on Image of Roller Coaster',fontsize=16)
# plt.figure(9)

# ----------------------------- Print Runtime ------------------------------- #
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
