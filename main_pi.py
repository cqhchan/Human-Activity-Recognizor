from preprocess import *
import threading
import time
from scipy import fft
import pickle
import os
import re

dataList = []
windowSize = 40
overlapPercentage = 0.3

class PiToArduinoComs(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global dataList
        rawdata = []
        path = 'Dataset1/Raw/Extra'
        for file in os.listdir(path):
            if file.endswith(".txt"):

                print(os.path.join(path, file))
                filename = os.path.join(path, file)
                f = open(filename, "r")

                for dataString in f:
                    #  str = "0.41,-0.04,0.18,-22.39,-17.73,-13.47|0.08,0.11,1.00,13.48,19.89,28.49|0.08,-0.11,
                    # -0.17,-86.15,28.58,69.42"
                    values = [float(x.strip()) for x in re.split('\||,', dataString)]
                    rawdata.append(values)
        while len(rawdata) > 0:
            time.sleep(0.05)

            # Run your data collection from arduino here
            data = rawdata[0].copy()
            del rawdata[0]

            dataList.append(data)
            while len(dataList)>windowSize:
                #check the systex for me might be wrong
                del dataList[0]



class MLServer(threading.Thread):
    #List of Predictions
    predictedList = []
    #Number of consequitive predictions
    pNumber = 3

    #the Model
    model = 0
    def __init__(self):
        global model
        # Load the Model
        filename = "main_model.sav"
        model = pickle.load(open(filename, 'rb'))
        threading.Thread.__init__(self)


    def run(self):
        while (True):

            prediction = self.predict()

            if prediction == -1:
                print("Invalid Prediction")
                time.sleep(0.5)

            else:
                print("Prediction " + str(prediction))
                time.sleep(5)


    #Process the RawData
    def processdata(self):
        global dataList
        #Copy the Raw Data
        rawWindowData = np.asarray(dataList.copy())
        #Delete 1 - Overlap of data
        dataList = dataList[int(windowSize * (1-overlapPercentage)):][:]

        processedData = []
        for sensorsIndex in range(rawWindowData.shape[1]):

            sensorData = rawWindowData[:][sensorsIndex]
            processedData.append(mean(sensorData))

            processedData.append(std(sensorData))
            processedData.append(mad(sensorData)[0])
            processedData.append(max(sensorData))
            processedData.append(min(sensorData))
            processedData.append(energy(sensorData))
            processedData.append(iqr(sensorData))

            sigma, rho = arCoeff(sensorData)

            processedData.append(sigma[0])
            processedData.append(sigma[1])
            processedData.append(sigma[2])
            processedData.append(sigma[3])
            processedData.append(rho)

            processedData.append(correlation(sensorData))
            fSensorData = fft(sensorData)

            processedData.append(maxInds(fSensorData))

            meanf = meanFreq(fSensorData)

            processedData.append(meanf.real + meanf.imag)

            skewnessf = skewness(fSensorData)

            processedData.append(skewnessf.real + skewnessf.imag)

            kurtosisf = kurtosis(fSensorData)
            
            processedData.append(kurtosisf.real + kurtosisf.imag)

        processedData.append(sma(rawWindowData[:][0], rawWindowData[:][1], rawWindowData[:][2]))
        processedData.append(sma(rawWindowData[:][3], rawWindowData[:][4], rawWindowData[:][5]))
        processedData.append(sma(rawWindowData[:][6], rawWindowData[:][7], rawWindowData[:][8]))
        processedData.append(sma(rawWindowData[:][9], rawWindowData[:][10], rawWindowData[:][11]))
        processedData.append(sma(rawWindowData[:][12], rawWindowData[:][13], rawWindowData[:][14]))
        processedData.append(sma(rawWindowData[:][15], rawWindowData[:][16], rawWindowData[:][17]))

        processedData = np.asarray(processedData)
        processedData[np.isnan(processedData)] = 0

        return processedData

    def predict(self):
        global model
        global dataList

        ## Check if number of data same as window size
        if (len(dataList)) == windowSize:

            print("Processing")
            processedData = self.processdata()

            test_prediction = model.predict([processedData])
            predictedValue = test_prediction[0]

            self.predictedList.append(predictedValue)

            ## if len of predicted List is more than number delete
            while len(self.predictedList) > self.pNumber:
                print(self.predictedList)
                del self.predictedList[0]
                print(self.predictedList)

            #check if len of predicted list of correct
            if len(self.predictedList) == self.pNumber:
                #check if all values in predicted list are the same
                if self.predictedList.count(self.predictedList[0]) == len(self.predictedList):
                    #delete the predicted list
                    valueToReturn = self.predictedList[0].copy()
                    del self.predictedList[:]
                    return valueToReturn
        return -1


print("testing started")
thread1 = PiToArduinoComs()
thread2 = MLServer()
thread1.start()
thread2.start()


while(True):

    time.sleep(1)


