from preprocess import *
import numpy as np
import sys
import re
import os
from scipy import fft



path = sys.argv[1]
rawFile = os.path.join(path,"RawWindowedData.npy")
rawWindowDataSet = np.load(rawFile)
# np.set_printoptions(threshold=sys.maxsize)
processedDataSet = []
print(rawWindowDataSet.shape)

for windowIndex in range(rawWindowDataSet.shape[0]):
    rawWindowData = rawWindowDataSet[windowIndex]

    # Get the basic data from each sensor
    processedData = []

    for sensorsIndex in range(rawWindowData.shape[1]):


        sensorData = rawWindowData[:][sensorsIndex]
        processedData.append(mean(sensorData))

        processedData.append(std(sensorData))
        processedData.append(mad(sensorData)[0])
        processedData.append(max(sensorData))
        processedData.append(min(sensorData))
        processedData.append(energy(sensorData))
        #processedData.append(areasimps(sensorData))
        #processedData.append(areatrapz(sensorData))



        processedData.append(iqr(sensorData))
        #processedData.append(entropy(sensorData))

        sigma , rho = arCoeff(sensorData)

        # if windowIndex == 917:
        #     print(sensorData)
        #     print(sigma)

        processedData.append(sigma[0])
        processedData.append(sigma[1])
        processedData.append(sigma[2])
        processedData.append(sigma[3])
        processedData.append(rho)


        processedData.append(correlation(sensorData))
        fSensorData = fft(sensorData)

        processedData.append(maxInds(fSensorData))

        meanf = meanFreq(fSensorData)

        processedData.append(meanf.real+ meanf.imag)

        skewnessf = skewness(fSensorData)

        processedData.append(skewnessf.real+ skewnessf.imag)

        kurtosisf = kurtosis(fSensorData)
        processedData.append(kurtosisf.real+ kurtosisf.imag)

    #processedData.append(totalabs(rawWindowData[:][0],rawWindowData[:][1],rawWindowData[:][2]))
    #processedData.append(totalabs(rawWindowData[:][6],rawWindowData[:][7],rawWindowData[:][8]))
    #processedData.append(totalabs(rawWindowData[:][12],rawWindowData[:][13],rawWindowData[:][14]))

    processedData.append(sma(rawWindowData[:][0],rawWindowData[:][1],rawWindowData[:][2]))
    processedData.append(sma(rawWindowData[:][3],rawWindowData[:][4],rawWindowData[:][5]))
    processedData.append(sma(rawWindowData[:][6],rawWindowData[:][7],rawWindowData[:][8]))
    processedData.append(sma(rawWindowData[:][9],rawWindowData[:][10],rawWindowData[:][11]))
    processedData.append(sma(rawWindowData[:][12],rawWindowData[:][13],rawWindowData[:][14]))
    processedData.append(sma(rawWindowData[:][15],rawWindowData[:][16],rawWindowData[:][17]))

    processedData = np.asarray(processedData)

    processedData[np.isnan(processedData)] = 0

    processedDataSet.append(np.asarray(processedData))


    if np.isnan(processedData).any() or float('Inf') in processedData or -float('Inf') in \
            processedData:
        print(windowIndex)


processedDataSet = np.asarray(processedDataSet)

processedData = processedDataSet.astype(float)
np.save(os.path.join(path, "ProcessedWindowedData"), processedDataSet)
test = np.load(os.path.join(path, "ProcessedWindowedData.npy"))

