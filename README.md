# CV Project: Comparison of FCNN & CNN for Image Classification

## Project Goals

Our team is relatively new to the space of machine learning and we selected learning goals to help give us an overview of neural networks.  Our learning goals were to:

- Familiarize ourselves  with neural nets
- Understand the difference between FCNN & CNNs
- Become familiar with the challenges of image classification
- Practice selecting, organizing, and using a dataset

## Implementation 
To achieve our learning goals, we implemented a fully connected neural net and a convolution neural net to classify road signs. The dataset we used is from the publicly available portion of the Chinese Traffic Sign Recognition database [1].  It contains 5998 images making up 58 different categories.  The images only contain single traffic signs and the road signs are (mostly) centered in the image. 

!["Dataset"](https://github.com/vscheyer/computer_vision/blob/main/documentation/dataset.PNG)  
*Fig. 1: A few examples of the type of images in our Dataset*
 

We based our FCNN and CNN off of a tensorflow tutorial [2].  Our goal was to generate simple neural nets that could be easily modified to facilitate our understanding of the differences between FCNN and CNNs.  After building our initial models, we compared how different modifications changed the perfoamce of each neural net.

### Fully Connected Neural Net
!["Dataset"](https://github.com/vscheyer/computer_vision/blob/main/documentation/four-layer_fcnn_58_classes.png)  
*Fig. 2: 1 fully connected layer vs. 4 fully connected layer NN*

To start, we built a simple FCNN with one hidden layer. We wanted to explore how adding layers impacted the accuracy. We were surprised to see that there was not very pronounced improvement when we increased the number of layers to 4. We had expected that adding multiple fully connected layers would either clearly improve the model or lead to overfitting. We only went up to 4 layers due to runtime challenges. In the future we could run this on a server and observe networks with many more layers to see if our results change.
 

### Convolutional Neural Net
!["Dataset"](https://github.com/vscheyer/computer_vision/blob/main/documentation/four-layer_cnn_58_layers.png)   
*Fig. 3: 1 convolutional layer vs. a 4 convolutional layers CNN*

To gain understanding of CNNs, we started with a very simple model that only contained 1 convolutional layer.  In an attempt to improve the performace, we modified the model to contain 4 convolutional layers.  For a dataset of this size, however, it appears that adding additional layers doesn't improve performance. This is likely due to overfitting; adding additional layers wouldn't necessarily improve the model because images only contain so many meaningful features. It is also possible that the simplicity of the images in this dataset could contribute to overfitting as well.


## Design Decisions
In our project, we intentionally implemented the neural net that is considered to be a poor choice for the application of image classification. In doing our initial project research we learned why FCNNs stuggle with image classification.  This is because the FCNN treats each individual pixel as a "feature." The higher the image resolution, the more the input layer grows since each single pixel is an input.  All of those inputs have weights in the first hidden layer, and the network then has many paramters.  A network with a large number of parameter takes longer to train, and FCNNs used for image classification have larger error than CNNs.

Convolutional neural nets are able to do feature extraction by grouping relevant pixels togther and training a fewer number of weights to describe the smae number of pixels as an FCNN.  This means they require less memory and are able to train on larger netorks.  

We wanted to witness the differences in performance and implemention of FCNN and CNNs first hand for image classification which led us to implement both neural nets using tensorflow to be able to compare the networks' performance.

## Challenges

**Picking the right resource**:  Having never used tensorflow before, we relied heavily on tutorials and stack overflow.  While there are many resoucres out there, we struggled to find ones that were accessible while tackling many similiar problems that we were encountering.  We did use Tensorflow documentation quite a bit but many built-in functions have some many parameters than can be tuned so it was a challenge to determine what was relevant for our project. As we learned more and more about neural nets and became more familiar with tensorflow, we were able to interpret more of the resources but there was intially a steep learning curve.

**Dataset organizing**:  Building off of experiences in prior courses, we thought that organizing our selected dataset and separating it into testing and training data would be a good first step.  Once we selected the Chinese Traffic Sign Recognition dataset, we wrote a script to separate training and testing data into four categories the user specified.  This seemed like a helpful tool to have before we even began to work with the library.  However, after digging into some tensorflow examples, we learned that the library already had built-in functions to help manage and separate datasets. The way we orgnaized our data was incompatible with the existing dataset loading features, so we ended up modifying our original script significantly.  While the final implementation was simpler, we did do unnecessary work that could have been avoided if we had looked closley at tensorflow examples prior to jumping in.

## Next Steps/Takeaways
To improve our project we would want to explore additonal ways to improve the accuracy of our basic neural nets.  For the FCNN, we could implement feature extraction prior to feeding the images into the neural net as a way to decrease the number of parameters in the network. For the CNN, we would want to explore ways of improving its accuracy like trainign for a larger number of epochs and/or including a dropout layer.

With more time, we would also like to explore incorporating neato simulation with our image classifier. We didn't begin to explore the challenges of feature extraction for camera streams in this project, so this would be a substantial add-on. Doing this extension would build our understanding of the industry challenges in implementing sign recognition in vehicles.

The timing of this project overlapped considerably with major national events and a period of increased uncertainty. When we were planning for getting work done, we brainstormed how to manage this uncertainty and decided to try out a more asynchornous workflow. We were able to minimize group meetings while increasing our autonomy to work when each of us was able to. This is a valuable insight for programming partner projects and it gave us useful tools to manage work during periods of increased stess outside of the project.

## Sources
**[1]** https://www.kaggle.com/dmitryyemelyanov/chinese-traffic-signs  
The dataset is a converted version of publicly available Chinese Traffic Sign Recognition Database.
This work is supported by National Nature Science Foundation of China(NSFC) Grant 61271306  
Credits:  
LinLin Huang,Prof.Ph.D  
School of Electronic and Information Engingeering,  
Beijing Jiaotong University,  
Beijing 100044,China  
Email: huangll@bjtu.edu.cn  
All images originally collected by camera under nature scenes or from BAIDU Street View  
 

**[2]** https://www.tensorflow.org/tutorials/images/classification#create_a_dataset
