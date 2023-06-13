# Created by: Simon Lönnqvist & Oscar Eriksson

import tensorflow as tf
from tensorflow.keras import models, layers
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import sys
import os
import re
import numpy as np

# Static lists for which optimizer will be used for which respective network
# The available optimizers are: ['adadelta', 'adagrad', 'adam', 'sgd', 'adamax', 'ftrl', 'nadam', 'rmsprop']
selectedOptimizersA = ['ftrl']
selectedOptimizersB = ['ftrl']
selectedOptimizersC = ['adagrad']

# The loss functions that are used in the networks, with the variants logits true and false
inputLossFunctionTrue = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
inputLossFunctionFalse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# The following three functions are the methods used to invoke the actual neural networks but allowing us
# to record plots, confusion matrices and training data
def alphaNet(trainingData, validationData, selectedOptimizers, inputLossFunction, numberOfEpochs, testData):
    print("Starting ANet")
    count = 0
    loggerPath = pathlib.Path(os.path.join(sys.argv[2]))
    writeFile = open(os.path.join(loggerPath, "resultsANet"), "w")
    for currentOptimizer in selectedOptimizers:
        plt.clf()
        hist, testData, testLoss = cnnNetworkAlpha(trainingData, validationData, currentOptimizer, inputLossFunction,
                                         numberOfEpochs, testData, loggerPath)
        count += 1
        plt.plot(hist.history['accuracy'], 'red', linewidth=3.0)
        plt.plot(hist.history['val_accuracy'], 'green', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Accuracy Curves [{opti:}]'.format(opti=currentOptimizer), fontsize=12)
        print("Saving plot to: [{currentDest:}]".format(currentDest=sys.argv[2]))
        dataPath = pathlib.Path(os.path.join(sys.argv[2], 'plotsANET'))
        dataPath.mkdir(parents=True, exist_ok=True)
        writeFile.write("Results for optimizer [{opti:}] are: TestAcc:{testValue:}, TestLoss: {testLoss:}, Accuracy:{Acc:}/AccLoss:{accLoss:}, ValAcc:{valAcc:}/ValLoss{valLoss:}".format(opti=currentOptimizer,                                                                                        testValue=testData, testLoss=testLoss, Acc=hist.history['accuracy'][-1], accLoss=hist.history['loss'][-1], valAcc=hist.history['val_accuracy'][-1], valLoss=hist.history['val_loss'][-1]))
        plt.savefig(os.path.join(dataPath, 'plotANET_{num:}.png'.format(num=count)))
    writeFile.close()


def betaNet(trainingData, validationData, selectedOptimizers, inputLossFunction, numberOfEpochs, testData):
    print("Starting BNet")
    count = 0
    loggerPath = pathlib.Path(os.path.join(sys.argv[2]))
    writeFile = open(os.path.join(loggerPath, "resultsBNet"), "w")
    for currentOptimizer in selectedOptimizers:
        plt.clf()
        hist, testData, testLoss = cnnNetworkBeta(trainingData, validationData, currentOptimizer, inputLossFunction,
                                        numberOfEpochs, testData, loggerPath)
        count += 1
        plt.plot(hist.history['accuracy'], 'red', linewidth=3.0)
        plt.plot(hist.history['val_accuracy'], 'green', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Accuracy Curves [{opti:}]'.format(opti=currentOptimizer), fontsize=12)
        dataPath = pathlib.Path(os.path.join(sys.argv[2], 'plotsBNET'))
        dataPath.mkdir(parents=True, exist_ok=True)
        writeFile.write("Results for optimizer [{opti:}] are: TestAcc:{testValue:}, TestLoss: {testLoss:}, Accuracy:{Acc:}/AccLoss:{accLoss:}, ValAcc:{valAcc:}/ValLoss{valLoss:}".format(opti=currentOptimizer,                                                                                        testValue=testData, testLoss=testLoss, Acc=hist.history['accuracy'][-1], accLoss=hist.history['loss'][-1], valAcc=hist.history['val_accuracy'][-1], valLoss=hist.history['val_loss'][-1]))
        plt.savefig(os.path.join(dataPath, 'plotBNET_{num:}.png'.format(num=count)))
    writeFile.close()

def charlieNet(trainingData, validationData, selectedOptimizers, inputLossFunction, numberOfEpochs, testData):
    print("Starting CNet")
    count = 0
    loggerPath = pathlib.Path(os.path.join(sys.argv[2]))
    writeFile = open(os.path.join(loggerPath, "resultsCNet"), "w")
    for currentOptimizer in selectedOptimizers:
        plt.clf()
        hist, testData, testLoss = cnnNetworkCharlie(trainingData, validationData, currentOptimizer, inputLossFunction,
                                           numberOfEpochs, testData, loggerPath)
        count += 1
        plt.plot(hist.history['accuracy'], 'red', linewidth=3.0)
        plt.plot(hist.history['val_accuracy'], 'green', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Accuracy Curves [{opti:}]'.format(opti=currentOptimizer), fontsize=12)
        dataPath = pathlib.Path(os.path.join(sys.argv[2], 'plotsCNET'))
        dataPath.mkdir(parents=True, exist_ok=True)
        writeFile.write("Results for optimizer [{opti:}] are: TestAcc:{testValue:}, TestLoss: {testLoss:}, Accuracy:{Acc:}/AccLoss:{accLoss:}, ValAcc:{valAcc:}/ValLoss{valLoss:}".format(opti=currentOptimizer,                                                                                        testValue=testData, testLoss=testLoss, Acc=hist.history['accuracy'][-1], accLoss=hist.history['loss'][-1], valAcc=hist.history['val_accuracy'][-1], valLoss=hist.history['val_loss'][-1]))
        plt.savefig(os.path.join(dataPath, 'plotCNET_{num:}.png'.format(num=count)))
    writeFile.close()

# The following three functions are the actual networks. Where they are compiled, trained, evaluated
# and a confusion matrix is created for each of them.
def cnnNetworkAlpha(trainingData, validationData, currentOptimizer, inputLossFunction, numberOfEpochs, testData, path):
    model = tf.keras.Sequential([
        layers.Normalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)
    ])

    print("Compiling ANet model with optimizer [{opti:}]".format(opti=currentOptimizer))
    model.compile(optimizer=currentOptimizer, loss=inputLossFunction, metrics=['accuracy'])
    history = model.fit(trainingDataset, epochs=numberOfEpochs, validation_data=validationDataset)

    test_loss, test_acc = model.evaluate(testDataset, verbose=2)
    print("The test accuaracy", test_acc)

    labels = np.array([])
    testDataset2 = datasetCreationTest(sys.argv[3], 1)
    for x,y in testDataset2:
       if y.numpy() == 0:
           labels = np.hstack([labels, 0])
       else:
          labels = np.hstack([labels, 1])
    
    prediciton = model.predict(testDataset, batch_size=1)
    prediciton = prediciton.argmax(axis=1)

    dispLabels = ['dehydrated', 'hydrated']

    cm = confusion_matrix(prediciton, labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dispLabels)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    fig.set_figwidth(3)
    fig.set_figheight(3)
    disp.plot(ax=ax)
    plt.savefig(os.path.join(path, 'confusionMatrix_Anet_{opti:}.png'.format(opti=currentOptimizer)), bbox_inches='tight', pad_inches=1.0)
    

    return history, test_acc, test_loss


def cnnNetworkBeta(trainingData, validationData, currentOptimizer, inputLossFunction, numberOfEpochs, testData, path):
    # Based on https://ieeexplore-ieee-org.db.ub.oru.se/document/9716555
    model = tf.keras.Sequential([
        layers.Normalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 32, 3)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(2, activation='softmax')
    ])

    print("Compiling BNet model with optimizer [{opti:}]".format(opti=currentOptimizer))
    model.compile(optimizer=currentOptimizer, loss=inputLossFunction, metrics=['accuracy'])
    history = model.fit(trainingDataset, epochs=numberOfEpochs, validation_data=validationDataset)

    test_loss, test_acc = model.evaluate(testDataset, verbose=2)
    print("The test accuaracy", test_acc)

    labels = np.array([])
    testDataset2 = datasetCreationTest(sys.argv[3], 1)
    for x,y in testDataset2:
       if y.numpy() == 0:
           labels = np.hstack([labels, 0])
       else:
          labels = np.hstack([labels, 1])
    
    prediciton = model.predict(testDataset, batch_size=1)
    prediciton = prediciton.argmax(axis=1)

    dispLabels = ['dehydrated', 'hydrated']

    cm = confusion_matrix(prediciton, labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dispLabels)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    fig.set_figwidth(3)
    fig.set_figheight(3)
    disp.plot(ax=ax)
    plt.savefig(os.path.join(path, 'confusionMatrix_Bnet_{opti:}.png'.format(opti=currentOptimizer)), bbox_inches='tight', pad_inches=1.0)


    return history, test_acc, test_loss


def cnnNetworkCharlie(trainingData, validationData, currentOptimizer, inputLossFunction, numberOfEpochs, testData, path):
    # Based on https://ieeexplore-ieee-org.db.ub.oru.se/document/9362375
    model = tf.keras.Sequential([
        layers.Normalization(),
        layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu',
                      input_shape=(32, 32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu',
                      input_shape=(32, 32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu',
                      input_shape=(64, 16, 16, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu',
                      input_shape=(64, 16, 16, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu',
                      input_shape=(128, 8, 8, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu',
                      input_shape=(128, 8, 8, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(2, activation='softmax')
    ])

    print("Compiling CNet model with optimizer [{opti:}]".format(opti=currentOptimizer))
    model.compile(optimizer=currentOptimizer, loss=inputLossFunction, metrics=['accuracy'])
    history = model.fit(trainingDataset, epochs=numberOfEpochs, validation_data=validationDataset)

    test_loss, test_acc = model.evaluate(testDataset, verbose=2)
    print("The test accuaracy", test_acc)

    labels = np.array([])
    testDataset2 = datasetCreationTest(sys.argv[3], 1)
    for x,y in testDataset2:
       if y.numpy() == 0:
           labels = np.hstack([labels, 0])
       else:
          labels = np.hstack([labels, 1])
    
    prediciton = model.predict(testDataset, batch_size=1)
    prediciton = prediciton.argmax(axis=1)

    dispLabels = ['dehydrated', 'hydrated']

    cm = confusion_matrix(prediciton, labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dispLabels)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    fig.set_figwidth(3)
    fig.set_figheight(3)
    disp.plot(ax=ax)
    plt.savefig(os.path.join(path, 'confusionMatrix_Cnet_{opti:}.png'.format(opti=currentOptimizer)), bbox_inches='tight', pad_inches=1.0)


    return history, test_acc, test_loss


# The following two functions are for creating the dataset for training/validation and
# then another one for testing 
def dataSetCreation(filePathForTraining):
    imageHeight = 32
    imageWidth = 32
    batchSize = 32

    dataPath = pathlib.Path(filePathForTraining).with_suffix('')
    print(dataPath)
    trainingDataset, validationDataset = tf.keras.utils.image_dataset_from_directory(
        dataPath,
        validation_split=0.1,
        subset="both",
        shuffle=True,
        seed=123,
        image_size=(imageHeight, imageWidth),
        batch_size=batchSize
    )
    return trainingDataset, validationDataset


def datasetCreationTest(filePathForTest, bs):
    imageHeight = 32
    imageWidth = 32
    batchSize = bs

    testDataset = tf.keras.utils.image_dataset_from_directory(
        filePathForTest,
        shuffle=False,
        #seed=123,
        image_size=(imageHeight, imageWidth),
        batch_size=batchSize
    )
    return testDataset

# Small function for sorting items in directories in alphanumerical order
def sortedAlpha(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alK = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alK)

# Main function for the program, here we set the memory limit for the program,
# create the datasets and starts each network for a defined number of epochs.
# 
# The paths for the folders needed for creating the datasets are passed when running
# the program via the terminal
if __name__ == '__main__':
    numberOfEpochs = 100
    MEMLimit = 4096
    print("Welcome to the testing suite")
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=MEMLimit)])
    trainingDataset, validationDataset = dataSetCreation(sys.argv[1])
    testDataset = datasetCreationTest(sys.argv[3], 32)
    alphaNet(trainingDataset, validationDataset, selectedOptimizersA, inputLossFunctionTrue, numberOfEpochs, testDataset)
    betaNet(trainingDataset, validationDataset, selectedOptimizersB, inputLossFunctionFalse, numberOfEpochs, testDataset)
    charlieNet(trainingDataset, validationDataset, selectedOptimizersC, inputLossFunctionFalse, numberOfEpochs, testDataset)
    print("This concludes the tests")
