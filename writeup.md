## Project 4: Follow Me
Project submission for the Udacity Robotics Software Engineer Nanodegree

Jonathan Georgino
February 5 2018

---

### This project writeup is based on the sample template provided in the base repo provided by Udacity.

---

[//]: # (Image References)
[trainingdatacollection]: ./trainingdatacollection.png
[networkarchitecture]: ./networkarchitecture.png


![trainingdatacollection]

This project entailed building a FCN and training it such that it could achieve an accuracy >= 40% (using the IoU metric). With a fair amount of effort, I was ultimately able to achieve a FCN which scored a 0.44016659591 for my submission for the final project of term 1. The completed Jupyter Notebook for this model training can be found here: [model_training.ipynb](./notebooks/success/model_training.md) and the model and weights files in '.h5' file format can be found here: [model_weights.h5](./data/weights/model_weights.h5) and [config_model_weights.h5](./data/weights/config_model_weights.h5)

Note that the training data has been removed from the repository due to the restrictions on number of files included in a repo for Udacity's submissions system.

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points

---
### Writeup / README

#### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.

This document is intended to fullfil the requirement of the Writeup / Readme.

#### 2. The write-up conveys the an understanding of the network architecture.

The FCN model that I implemented to solve the Semantic Segmentation consists of an encoder stage of 3 encoding blocks, and 1x1 convolution layer, and a decoder stage of 3 decoding blocks. The following python function implements the model as previously described.

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    enc_layer1 = encoder_block(inputs, 32, 2)
    enc_layer2 = encoder_block(enc_layer1, 64, 2)
    enc_layer3 = encoder_block(enc_layer2, 128, 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(enc_layer3, 256, 1, 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    dec_layer3 = decoder_block(conv_layer, enc_layer2, 128)
    dec_layer2 = decoder_block(dec_layer3, enc_layer1, 64)
    dec_layer1 = decoder_block(dec_layer2, inputs, 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(dec_layer1)

```


##### Encoder Blocks

As mentioned above, the model begins with a series of 3 encoder blocks. The goal of the encoder is to extract features from the image. Each encoder block includes a separable convolution layer, which is a technique that reduces the number of parameters needed (improves efficiency) and has the effect of preserving spatial information from the image. I personally prefer the full name for this technique introduced in the lesson, 'depthwise separable convolution', as it makes understanding what is actually happening more intuitive / easy to remember. In short, a separable convolution is a convolution performed on each input channel of the input layer followed up with a 1x1 convolution that takes the output from the previous step and combines them into a single output layer. A positive side effect of using fewer parameters is that separable convolutions also reduce overfitting.

The encoder is further optimized by using a technique known as Batch Normalization. In practice, this means that instead of just normalizing the inputs to the network, all of the inputs to the layers within the network are also being normalized. It's the same normalization technique using the mean and variance of the values, but applied at every layer. The lesson on this topic presents four notable benefits of this technique, which include faster network training, higher learning rates, creation of deeper networks, and provides some regularization (working as well as dropout technique).

The following snippet shows the python implementation of a single encoder block using a provided function which performs the separable convolution and batch normalization.
```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```


##### 1x1 Convolution Layers

 This 1x1 Convolution layer replaces a traditional fully connected layer. This results in the output value where the tensor will remain 4D instead of being flattened to 2D, which means spatial information will be retained. This technique basically creates a mini Neural network running over the patch instead of a linear classifier. Mixing in 1x1 convolutions is very inexpensive way to make models deeper and have more parameters without completely changing network structure. This technique is considered to be inexpensive because a 1x1 convolution ends up really just being matrix multiplications. Note that this is implemened with regular convolution, not the separable convolution technique discussed above. This layer makes up the final stage of the encoding portion of the FCN.


##### Decoder Blocks

The model ends with a series of 3 decoder blocks. The goal of the decoder is to upscale the output from the encoder such that it's the same size as the original image. Each decoder block includes a bilinear upsampling layer (factor = 2), layer concatenation, and a seperable convolution layer (with batch normalization, just as in the encoder block). I've already discussed separable convolutions and batch normalization in the Encoder Block section above, so here I will focus on the two new concepts.

Bilinear upsampling is a technique that takes the weighted average of four nearest (diagonally) known pixels to estimate a new pixel intensity. Although this method does not contribute to the network as a learnable layer, and it also is prone to lose some level of detail, it contributes to the speed of the network.

Layer Concatenation is a method of combining an upsampled layer with a layer which contains more special information than the upsampled layer and provides an equivalent performance benefit to the network as a skipped connection. Skipped connections allow the network to use information from multiple resolution scales, which enables more precise segmentation decisions. As such, the end result is that there is a prediction for each single pixel contained in the original image.

The following snippet shows the python implementation of a single decoder block using the provided functions to perform each of the steps described above.
```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    output_layer = BilinearUpSampling2D((2,2))(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([output_layer, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_layer, filters, 1)
    
    return output_layer
```


##### The Full Monty

The figure below (generated from the code) illustrates the FCN network architecture.
![networkarchitecture]

Note that I was able to generate this model by following the posts in the #udacity_follow_me slack channel. Special thanks to @ross_lloyd, @safdar, @kalpitsmehta for their comments on this topic.


##### Development of the FCN Model

A significant amount of experimentation went into the development of the model described above, as evident by the number of notebooks included in the `notebooks` directory of this repository. In my [first attempt](./notebooks/run1/model_training.md), I used a similar model as described above however with different parameters and supplied no additional training data. This resulted in a overall score of 0.364738134492 -- not bad for first pass. This also gave me the opportunity to realize that a single run would take several hours on my laptop, so I then took the time to setup the AWS instance for future attempts. [note: I really liked this aspect of the project which went through all the steps of actually performing the computations on a cloud instance. Very practical for actual work in industry.] For the [next pass](./notebooks/run2/model_training.md), I added a 4th layer to the encoding and decoding blocks, which improved the score, but only by 0.4%. At this point I left the model unchanged at 4 layers and experimented with different parameters (described in the next section) however I was unable to acheive satisfactory performance -- got as close as 39%.

At [attempt #7](./notebooks/run7/model_training.md), I reduced the model to three layers and changed some of the parameters, and at the same time, I supplemented the training data with additional data captured myself in the QuadSim Unity Similator. Ultimately, I ended up providing two runs of custom training and validation because my first capture did not capture data with enough of a crowd to be effective. In run 2 of data collection, I was more generous in placing the number of spawn points for non-hero characters. It was after including the training data that I collected that I was able to boost ther performance of the network up to 0.44016659591 on the 9th attempt, as shown [here](./notebooks/success/model_training.md).


#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.

##### Hyperparameter Tuning

In order to achieve a successful solution, I landed on the following set of hyperparameters:

| Hyperparameter   | Value |
|------------------|-------|
| learning_rate    | 0.005 |
| batch_size       | 32    |
| num_epochs       | 20    |
| steps_per_epoch  | 250   |
| validation_steps | 50    |
| workers          | 4     |

I tweaked these parameters several times across my 9 attempts, but ultimately felt that the final set of parameters were reasonable. I observed that the default of 10 epochs appeared to be too low and that there was still some performance benefits to be achieved through additional training. For this reason I extended my run to 20 epochs. I also reduced the learning rate to half of the default value since the course lessons indicated that a lower learning rate would ultimately lead to a better trained system in most cases. I picked a batch size of 32 somewhat arbitrarily, because it's a nice power of 2, however I don't believe there's any inheret performance advantage of selecting numbers which are powers of two in this implementation. [This line of thinking also coincided with the suggestion of the lab to use 32 or 64 ;-)]

Throughout my 9 attempts, I was also able to make some observations regarding how the amount of time it takes to train the network can vary significantly with the values for these parameters. As such, I was mindful about keeping things as realistic as possible. I experimented with 300 steps_per_epoch but found that the additional compute needed did not make it worth repeating. I also increase the validation_steps for a run, but I also did not see any a reason to keep it elevated. When it came to workers, I wasn't sure what would be the optimal value for running on the AWS cloud instance, and I'm not so sure that there was a meaningful differnce between 2 and 4 workers, as I made other parameter changes at the same time, so I could not draw any conclusions regarding the direct impact on compute time.

##### Model Parameters

Also important to discuss is the selection of the other model parameters which were critical to arriving at a successful implementation. In my original attempts at developing a model, I used filters for the first layer of the encoder / last layer of decoder to have a depth of 16, and then increasing power of 2s on up, stopping at 64 for a 3 x encoder block implementation or 128 for a 4 x encoder block implementation. As the numerous trials show, I was not able to get good performance, so I decided to remove the 16 depth layer and instead opted for a 3 x encoder block implementation beginning at 32 and increasing power of twos for each layer. For all strides, I used a value of 2, and for the bilinear upsampling, I also left it at the suggested value of two.


#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

I believe that the commentary and discussion in the above sections fulfills this requirement.

#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

I believe that the commentary and discussion in the above sections fulfills this requirement.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

##### Limitations

Based on observations of my several trial runs prior to supplementing the dataset with additional images I captured myself, I believe that the biggest weakness in the data was the performance of recognizing the hero when in a crowd of people. For this reason, all of the additional data I captured in both training runs involved the hero walking around crowds of many non-hero people.

When thinking of this model in real-world applications, it can be obvious that there will be a weakness such that the simulated hero was made to be significantly different than non-hero persons by means of coloring the entire body a shade of red. In reality, there would be very few cases in which a human in a crowd could be so fully differentiated from other persons in the crowd. This leads me to believe that for real world applications, the model would need to be much deeper and require much more training data in order to achieve a similar level of performance.

The project rubric asks specifically if this model AND data would work well for following other types of objects such as a dog, cat, or car, however I believe this question has some ambiguity. If the data used to train the model is for a human hero, then of course it will not be able to recognize and follow another object. However if the model is kept the same but it is trained for a similar dataset containing dogs instead of humans, or cats instead of humans, then I do not see any reason why the model would not work at a similar level.

##### Opportunities For Improvement / Future Enhancements

With a IoU score of just 44%, clearly there is room for improvement. The easiest way to make improvements would be to compile a larger and more complete set of training data. There are some pains in collecting data from the simulator, as it is a manual process of defining patrol paths, spawn points, etc, and there is no way to save the paths which are created. Collecting data would be much easier if it were possible to create the test scenarios programatically. In this fashion it would involve much fewer man-hours being dedicated to the task of data collection.

Another opportunity for improvement would be to continue to experimentally tweak the hyperparameters one-by-one in order to experimentally tune the FCN to achieve max performance with the given dataset. Again, this would be much easier to achieve with additional scripting such that it wouldn't require a human to manually make the adjustment and re-run the experiment and record the result. I achieved success on my first run after adding in the two training/validation runs of my own collected data, so I am certain that the investment of additional trials with parameter tweaking could achieve even better results with the current data.

### Model

#### 1. The model is submitted in the correct format. 

The 'model_weights.h5' generated via training on AWS instance file can be found here: [model_weights.h5](./data/weights/model_weights.h5)
The 'config_model_weights.h5' generated via the training on AWS instance file can be found here: [config_model_weights.h5](./data/weights/config_model_weights.h5)

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.

Per the project rubric, the neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric in order for a passing grade. Thankfully after the multiple attempts detailed above, I was able to achieve an IoU of 0.44016659591 for my submission. The associated training and validation images can be found in the completed Jupyter Notebook here: [model_training.ipynb](./notebooks/success/model_training.md)