===============
Segment Classifier
===============
Introduction
------------------
| The Segment classifier is a pytorch based neural network which is based on pretrained ResNet18.
| Segment classifier is used to classify document segments or words into handwritten or printed.
| It is one of the most important networks at the initial stage of the OCR pipeline.
| The Segment classifier is capable of distributing data across multiple GPUs using data parallelism

Usage
-----------------
| The Segment classifier is a class which needs to initialised
| For example:-
| ``sc=Segment_Classifier()``
| The constructor of the class loads the model and gets it ready for data parallelism.
| The class has two functions preprocess which takes the image folder containing the segments as input as applies preprocessing.
| This is further followed by the test function which starts the classification upon multiple gpus if present

Class methods
-----------------
| ``__init__`` :- The constructor initialises the ResNet18 model with the state dictionaries with initialisation of other class variables
| ``preprocess(folder)``:- This function is accepts the folder for the datasets and applies transforms into the dataset.
| ``test()``:- This is called after the preprocessing function to execute the datasets over the model and returns the binary classified prediction matrix as a numpy object
