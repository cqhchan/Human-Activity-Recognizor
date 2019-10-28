import numpy as np
import sys
import re
import os
from random import sample


import ctypes
print(ctypes.sizeof(ctypes.c_voidp))
inputFilePath = sys.argv[1]
outputFilePath = sys.argv[2]

print(inputFilePath)
processedDataFile = os.path.join(inputFilePath,"ProcessedWindowedData.npy")
processedLabelFile = os.path.join(inputFilePath,"WindowedLabels.npy")

processedWindowDataSet = np.load(processedDataFile,allow_pickle=True)
processedLabel = np.load(processedLabelFile)

print(processedWindowDataSet.shape)
print(processedLabel.shape)

testCount = int(processedLabel.shape[0]*1)

mask = np.zeros((processedLabel.shape[0]),dtype=bool)

print(mask.shape)

np.random.seed(0)
testIndex = sample(range(processedLabel.shape[0]), testCount)

mask[testIndex] = 1


testDataSet = processedWindowDataSet[mask][:]
testLabels = processedLabel[mask]


print(testDataSet.shape)
print(testLabels.shape)

trainDataSet = processedWindowDataSet[~mask][:]
trainLabels = processedLabel[~mask]

print(trainDataSet.shape)
print(trainLabels.shape)


#np.save(os.path.join(outputFilePath, "train", "train_data" ), trainDataSet,allow_pickle=True)
#np.save(os.path.join(outputFilePath, "train", "train_label" ), trainLabels,allow_pickle=True)

np.save(os.path.join(outputFilePath, "test", "test_data" ), testDataSet,allow_pickle=True)
np.save(os.path.join(outputFilePath, "test", "test_label" ), testLabels,allow_pickle=True)
