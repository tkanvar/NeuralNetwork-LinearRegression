import numpy as np
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys, os
from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import math
import random
import time

############################# INITIALIZATIONS

def prepareLayersOfNN(noOfInputs, hiddenLayers, noOfOutputs):
	layers = []
	layers.append(noOfInputs)
	for i in range(len(hiddenLayers)):
		layers.append(hiddenLayers[i])
	layers.append(noOfOutputs)

	return layers

def initializeWB(layers):
	noOfLayers = len(layers)

	# w, b
	w = []
	b = []

	w.append(np.zeros((1,1)))		# 1st layer w doesnt exist
	b.append(np.zeros((1,1)))		# 1st layer b doesnt exist

	for i in range(1, noOfLayers):			# Input layer doesnt have w or b
		layerW = np.zeros((layers[i-1], layers[i]))

		noOfB = layers[i]
		layerB = np.zeros((noOfB,1))

		w.append(layerW)
		b.append(layerB)

	for i in range(1, len(w)):
		w[i] = np.random.uniform(size=(layers[i-1], layers[i]))

	for i in range(1, len(b)):
		b[i] = np.random.uniform(size=(layers[i], 1))

	return w, b

def initializeAZ(layers):
	noOfLayers = len(layers)

	# a, z
	a = []
	z = []

	z.append(np.zeros((1,1)))		# 1st layer z doesnt exist

	for i in range(noOfLayers):
		layerA = np.zeros((layers[i],1))
		a.append(layerA)

	for i in range(1, noOfLayers):
		layerZ = np.zeros((layers[i],1))
		z.append(layerZ)

	return a, z

def initializeDelta(layers):
	noOfLayers = len(layers)
	delta = []

	delta.append(np.zeros((1,1)))	# 1st layer delta doesnt exist

	for i in range(1, noOfLayers):
		layerDelta = np.zeros((layers[i],1))
		delta.append(layerDelta)

	return delta

def readData(trainfilename, testfilename):
	traindata = pd.read_table(trainfilename, delimiter=',', header=None)
	testdata = pd.read_table(testfilename, delimiter=',', header=None)

	trainX = traindata.iloc[:,:-1]
	testX = testdata.iloc[:,:-1]
	trainY = traindata.iloc[:,-1]
	testY = testdata.iloc[:,-1]

	return trainX, trainY, testX, testY

def getBatch(min_batch_cntr, batchSize, trainX, trainY):
	startInd = min_batch_cntr * batchSize
	endInd = startInd + batchSize

	trainX1 = trainX[startInd:endInd, :]
	trainY1 = trainY[startInd:endInd, :]

	return trainX1, trainY1

def initializeBatchWB(layers):
	noOfLayers = len(layers)

	batchW = []
	batchB = []

	batchW.append(np.zeros((1,1)))		# 1st layer w doesnt exist
	batchB.append(np.zeros((1,1)))		# 1st layer b doesnt exist

	for i in range(1, noOfLayers):			# Output layer doesnt have w or b
		layerW = np.zeros((layers[i-1], layers[i]))
		layerB = np.zeros((layers[i],1))

		batchW.append(layerW)
		batchB.append(layerB)

	return batchW, batchB

def initializeInputLayer(a, trainRow):
	a[0] = trainRow
	return a

############################# LOGIC

def sigmoid(z, useSigmoid, isLastLayer):
	a = 0.0
	if useSigmoid == True or isLastLayer == True:
		e = np.exp(-z)
		a = (1.0 / (1.0 + e))
	else:
		zerovec = np.zeros((z.shape[0], z.shape[1]))
		a = np.maximum(zerovec, z)
	return a

def forwardPropagate(w, b, z, a, layers, useSigmoid):
	for l in range(1, len(layers)):
		z[l] = (w[l].T @ a[l - 1]) + b[l]

		isLastLayer = False
		if l == (len(layers) - 1): 
			isLastLayer = True

		if isLastLayer:
			a[l] = z[l]
		else:
			a[l] = sigmoid(z[l], useSigmoid, isLastLayer)

	return a, z

def derivativeSigmoid(a, useSigmoid, isLastLayer):
	r = 0.0
	if useSigmoid == True or isLastLayer == True:
		r = a * (1 - a)
	else:
		r = np.zeros((a.shape[0], a.shape[1]))
		np.place(r, a > 0, 1)
		np.place(r, a <= 0, 0)

	return r

def calculateOutputLayerDelta(delta, trainY, a, layers, useSigmoid):
	L = len(layers) - 1
	delta[L] = (a[L] - trainY)
	return delta

def backwardPropagate(w, b, a, delta, layers, useSigmoid):
	for l in range((len(layers) - 2), 0, -1):						# Going from second last layer to 2nd layer. Excluding 1st layer. For loop runs till l > 0
		delta[l] = (w[l+1] @ delta[l+1]) * derivativeSigmoid(a[l], useSigmoid, False)

	return delta

def updateDelWB(batchDelW, batchDelB, delta, a, layers):
	for l in range(1, len(layers)):
		batchDelW[l] = a[l - 1] @ delta[l].T
		batchDelB[l] = delta[l]

	return batchDelW, batchDelB

def updateC(trainY, a, layers):
	L = len(layers) - 1

	aL = a[L]

	batchC = np.sum(np.power((trainY - aL), 2)) / 2.0
	return batchC

def updateWB(w, batchDelW, b, batchDelB, learning_rate, layers, batchSize):
	for l in range(1, len(layers)):
		w[l] = w[l] - learning_rate * batchDelW[l]
		b[l] = b[l] - learning_rate * (np.sum(batchDelB[l], axis=1)).reshape((-1, 1))

	return w, b

def readAndOneHotEncodingData(trainfilename, testfilename):
	trainX, trainY, testX, testY = readData(trainfilename, testfilename)
	trainX = trainX.values
	trainY = trainY.values.reshape((-1, 1))
	testX = testX.values
	testY = testY.values.reshape((-1, 1))

	print ("Done Encoding")

	return trainX, trainY, testX, testY

def trainNN(trainX, trainY, maxepocs, batchSize, noOfInputs, hiddenLayers, noOfOutputs, threshold, learning_rate, tolerance, useSigmoid):
	# Variables
	layers = prepareLayersOfNN(noOfInputs, hiddenLayers, noOfOutputs)
	w, b = initializeWB(layers)
	a, z = initializeAZ(layers)
	delta = initializeDelta(layers)

	# SGD
	isFirstLoop = True
	withinThresholdCounter = 0
	reachedOptimum = False
	pvsBatchC = 0.0
	curBatchC = 0.0
	pvsEpochC = 0.0
	curEpochC = 0.0
	isEpochLossFail = 0
	epoc = 0
	printStr = ""

	batchDelW, batchDelB = initializeBatchWB(layers)

	for epoc in range(maxepocs):
		if epoc % 100 == 0:
			printStr = "epoc = " + str(epoc) + "; "
		withinThresholdCounter = 0

		for min_batch_cntr in range(int(len(trainX) / batchSize)):
			trainXBatch, trainYBatch = getBatch(min_batch_cntr, batchSize, trainX, trainY)

			a = initializeInputLayer(a, trainXBatch.T)
			a, z = forwardPropagate(w, b, z, a, layers, useSigmoid)
			delta = calculateOutputLayerDelta(delta, trainYBatch.T, a, layers, useSigmoid)
			delta = backwardPropagate(w, b, a, delta, layers, useSigmoid)

			batchDelW, batchDelB = updateDelWB(batchDelW, batchDelB, delta, a, layers)
			curBatchC = updateC(trainYBatch.T, a, layers)
			w, b = updateWB(w, batchDelW, b, batchDelB, learning_rate, layers, batchSize)

			if isFirstLoop == False and abs(pvsBatchC - curBatchC) <= threshold:
				withinThresholdCounter += 1
				if (withinThresholdCounter >= (int(len(trainX) / (batchSize * 2)))):
					reachedOptimum = True
					break
			else: isFirstLoop = False

			if epoc % 100 == 0 and min_batch_cntr == 0:
				printStr += "DiffError = " + str(abs(pvsBatchC - curBatchC))
			pvsBatchC = curBatchC

		curEpochC = curBatchC
		if epoc % 100 == 0:
			print (printStr, "; Diff Epoch Error = ", abs(pvsEpochC - curEpochC))
		if tolerance != -1 and abs(pvsEpochC - curEpochC) <= tolerance:
			#print ("Diff Epoch Error = ", abs(pvsEpochC - curEpochC))

			isEpochLossFail += 1
			if isEpochLossFail == 2:
				#print ("Epoch fail to converge. Reduce tolerance from ", learning_rate, " to ", (learning_rate/5.0))
				learning_rate = learning_rate / 5.0
				isEpochLossFail = 0;
		else:
			isEpochLossFail = 0
		pvsEpochC = curEpochC

		if reachedOptimum == True:
			break

	print ("# of epoc = ", epoc)
	print ("Error = ", curBatchC)

	return w, b, layers

def testNN(x, y, w, b, layers, printStr, useSigmoid):
	print ("Testing ",)
	
	a, z = initializeAZ(layers)
	actualY = []
	predY = []

	correct = 0
	total = 0
	error = 0
	for row in range(x.shape[0]):
		XRow = x[row]
		XRow = XRow.reshape((-1, 1))
		YRow = y[row]
		YRow = YRow.reshape((-1, 1))

		a = initializeInputLayer(a, XRow)
		a, z = forwardPropagate(w, b, z, a, layers, useSigmoid)

		L = len(layers) - 1
		aL = a[L]

		print ("aL = ", aL)

		error += abs(YRow - aL) / abs(YRow)

		actualY.append(YRow)
		predY.append(aL)

	print (printStr, " ",)
	acc = (100 - (error * 100 / x.shape[0]))
	print ("Accuracy = ", acc)

	return actualY, predY, acc

trainfilename = "C:\\tanuk\\WorkArea\\IITD\\MS\\NN\\nn-LnrRegrsn\\train.txt"
testfilename = "C:\\tanuk\\WorkArea\\IITD\\MS\\NN\\nn-LnrRegrsn\\test.txt"

maxepocs = 15000
batchSize = 20
noOfInputs = 1
hiddenLayers = [1]
noOfOutputs = 1
threshold = 1e-13
learning_rate = 0.001
trainX, trainY, validX, validY = readAndOneHotEncodingData(trainfilename, testfilename)

w, b, layers = trainNN(trainX, trainY, maxepocs, batchSize, noOfInputs, hiddenLayers, noOfOutputs, threshold, learning_rate, -1, True)

actualTestY, predTestY, acc = testNN(trainX, trainY, w, b, layers, "Training Examples", True)
actualTestY, predTestY, acc = testNN(validX, validY, w, b, layers, "Testing Examples", True)
