# detecting-dehydration
Exam work for 2023 for trying to detect dehydration in basil plants

# libraries used:
- tensorflow/keras
- pathlib
- matplotlib.pyplot
- sklearn
- sys
- is
- re
- numpy

# explanation on how to use the code provided in this repository

#CropperModule
1. Provide path to folder containing images that shall be cropped to “path”
2. Provide path to folder to put cropped images to “path_rect”
3. Provide desired width and height as “widthCrop” and “heightCrop”
4. Run the cropper via the terminal without any additional arguments

#QuadrantImagePatcher
IMPORTANT: For the folders input into the program, they need to following the structure: of one 
folder containing 2 folders where both classes are in their seperate folder. This is because for
the method we use for creating the dataset the labels are inferred from the folder names

1. Provide indexes for the following variables
  1a. INDEXLIST = list of the files reserved for testing
  1b. FILEINDEX = list of the files to be patched in selected folder
2. Provide path to folder containing images that shall be patched to “path”
3. Provide path to folder to put patched images to “path_rect_train”. This will save all but the lower right corner of the original image into the folder.
4. Provide path to folder for the remaining quadrant to “path_rect_test”. The reason for this is that this quadrant is exclusively used for testing, and not in training.
5. The patcher is set to create 32x32 patches, using strides of 24 pixels, resulting in an overlap of 8 pixels.
6. Run the image patcher via the terminal without any additional arguments

#CNNTestingSuite
1. Provide selected optimizers for ANet, BNet and CNet in the “selectedOptimzers” variables.
2. Lossfunctions can be changed via the inputLossFunctionTrue and inputLossFuncitonFalse, however, only the pre-provided one has been tested for this implementation.
3. Enter how many epochs is wanted and how much GPU memory is wanted to be used via the variables: numberOfEpochs and MEMLimit
4. Run the program via the terminal with three additional arguements, which are:
  - Path to the folder containing the training data
  - Path to the folder where you want to save the data from the networks (graphs, C-Matrixes and accuracies)
  - Path to the folder containing the test data
