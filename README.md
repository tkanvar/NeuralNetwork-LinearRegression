# Neural Network - Linear Regression

Linear Regression is done using Neural Networks. The folder contains sample training and testing data with file names as train.txt and test.txt respectively. The input feature is 1 and the output is a real number. A linear line is fit on the given data set

Change the path of the training file, in nn.py, to the location where train.txt is downloaded to (you can find train.txt in this folder)
Chnage the path of the testing file, in nn.py, to the location where test.txt is downloaded to (you can find test.txt in this folder)

Run the following command
  python nn.py
  
This will print the 
1) epochs to train the neural network
2) accuracy of the training set
3) accuracy of testing set
on the command window.

The code has following parameters in nn.py that can be varied 
1) maxepochs - maximum number of epochs till which to train the NN
2) batchSize - batch size
3) noOfInputs - Number of features of the input
4) hiddenLayers - Array of hidden layers. Each element in the array is the number of nodes in that layer
5) noOfOutputs - Number of outputs expected. Since we are doing linear regression, it will always be 1. It will change in case the output is discrete.
6) threshold - terminating condition, i.e. when experimental output and expected output fall in the threshold range for an epoch, then it has converged, i.e. terminate
7) learning_rate - how fast converge the weights of the NN
8) trainfilename - name and path of the training file
9) testfilename - name and path of the testing file
