## Project 4: Follow Me
Project submission for the Udacity Robotics Software Engineer Nanodegree

Jonathan Georgino
February 4 2018

---

### This project writeup is based on the sample template provided in the base repo provided by Udacity.

---

[//]: # (Image References)
[trainingdatacollection]: ./trainingdatacollection.png
[networkarchitecture]: ./networkarchitecture.png


![trainingdatacollection]

This project entailed building a FCN and training it such that it could achieve an accuracy >= 40% (using the IoU metric). With a fair amount of effort, I was ultimately able to achieve a FCN which scored a 0.44016659591 for my submission for the final project of term 1. The completed Jupyter Notebook for this model training can be found here: [model_training.ipynb](./notebooks/success/model_training.md) and the 'model_weights.h5' file can be found here: [model_weights.h5](./data/weights/model_weights.h5)

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points

---
### Writeup / README

#### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.

This document is intended to fullfil the requirement of the Writeup / Readme.

#### 2. The write-up conveys the an understanding of the network architecture.

![networkarchitecture]

Note that I was able to generate this model by following the posts in the #udacity_follow_me slack channel. Special thanks to @ross_lloyd, @safdar, @kalpitsmehta for their comments on this topic.

#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.


#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.


#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

### Model

#### 1. The model is submitted in the correct format. 

The 'model_weights.h5' generated via training on AWS instance file can be found here: [model_weights.h5](./data/weights/model_weights.h5)

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.

Per the project rubric, the neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric in order for a passing grade. Thankfully after the multiple attempts detailed above, I was able to achieve an IoU of 0.44016659591 for my submission. The associated training and validation images can be found in the completed Jupyter Notebook here: [model_training.ipynb](./notebooks/success/model_training.md)