# CV Project: Comparison of FCNN & CNN for Image Classification

## Project Goals

Our team is relatively new to the space of machine learning and we selected learning goals to help give us an overview of neural networks.  Our learning goals were to:

- Familiarize ourselves  with neural nets
- Understand the difference between FCNN & CNNs
- Become familiar with the challenges of image classification
- Practice selecting, organizing, and using a dataset

## Implementation 
To achieve our learning goals, we implemented a fully connect neural net and a convolution neural net to classify road signs. The dataset we used is from the publiclly avaliable portion of the Chinese Traffic Sign Recognition database [1].  It contains 5998 images making up 58 different categories.  The images only contain single traffic signs and the road signs are (mostly) centered in the image. 

!["Dataset"](https://github.com/vscheyer/computer_vision/blob/main/documentation/dataset.PNG)  
*Fig. 1: A few examples of the type of images in our Dataset*
 

We based our FCNN and CNN off of a tensorflow tutorial [2].  Our goal was to great simple neural nets that could be easily modfied to facilitae our understanding of the differnces beteen FCNN and CNNs.  After building our intial models, we compared how different modifications changed the perfoamce of each neural net

### Fully Connected Neural Net
!["Dataset"](https://github.com/vscheyer/computer_vision/blob/main/documentation/four-layer_fcnn_58_classes.png)  
*Fig. 3: blah blah*

To start, we build a basic 
 

### Convolutional Neural Net
!["Dataset"](https://github.com/vscheyer/computer_vision/blob/main/documentation/four-layer_cnn_58_layers.png)   
*Fig. 3: 1 convolutional layer vs. a 4 convolutional layer CNN*

To gain understanding of CNNs, we started with a very simple model that only contained 1 convolutional layer.  To improve the performace we modified the model to contained 4 convolutional layers.  For a dataset of this size, adding a few more layers allows the model to extract more features and improve performance. Adding many more additional layers wouldn't necessarily improve the model though because images only contain so many meaningful features.
 
### Comparison


## Design Decisions
In our project, we intentionally implemented the neural net that is considered to be a poor choice for the application of image classification. In doing our intial project research we learned why FCNNs stuggle with image classification.  This is because the FCNN treats each individual pixel as a "feature." The higher resolution the image, the more input layer grows were each input is a single pixel from the image.  All of those inputs have weights in the first hidden layer and the network then has many paramters.  A network with a large number of parameter takes longer to train and FCNN used for image classification have larger error than CNNs.

Convolutional neural nets are able to do feature extraction by grouping relevant pixels togther and training a fewer number of weights to describe the smae number of pixels as an FCNN.  This means they require less memory and are able to training on larger netorks.  

We wanted to witness the difference in performance and implemention of FCNN and CNNs first hand for image classification which led us to implement both neural nets using tensorflow to be able to compare the networks performance.

## Challenges

**Picking the right resource**:  Having never used tensorflow before, we relied heavily on turorials and stack overflow.  While there are many resoucres out there, we struggled to find ones that were both accessible and tackling similiar problems that we were encountering.  We did use Tensorflow documentation queit a bit but many built-in functions have some many parameters than can be tuned it was a challenge to determine what was relevant for our project. As we learned more and more about neural nets and became more familiar with tensorflow, we were able to interpert more of the resources but there was intially a steep learning curve.

**Dataset organizing**:  Building off of experiences in prior courses, we thought that orgnaizing our selected dataset and seperating it into testing and training data.  Once we selected the Chinese Traffic Sign Recognition dataset, we wrote a script to seperate training and testing data for categories the user specified.  This seemed like a helpful tool to have before we even began to work with the library.  However, after digging into some tesnroflow examples, we learned that the library already had built-in function to help manage and seperate datasets. The way we orgnaized our data was incompatible with the existing way to load datasets so we ended up modifying our original script signifigatly.  While the final implementation was simpler, we did do uneceddary work that could have been avoided if we had looked closley at tensorflow examples prior to jumping in.

## Next Steps/Takeaways
To improve our project we would want to explore additonal ways to improve the accuracy of our basic neural nets.  For the FCNN, we could implement feature extraction prior to feeding the images into the neural net as a way to decrease the number of parameters in the network. For the CNN, we would want to explore ways of improving it's accuracy like trainign for a larger number of epochs and/or including a dropout layer.

With more time, we would also like to explore incorporating neato simulation with our image classifier. We didn't begin to explore the challenges of feature extraction for camera stream in this project so this would be a substainal add-on. Doing this extension would build our understanding of the industry challenges in implemnting sign recognition in vehicles.

The timing of this project overlapped consierablly with major national events and a period of increasede uncreatinity. In planning for getting work done, we brainstormed how to manage the project this uncertainity and decided to try out more asynchornous work. We were able to minimize group meetings while increasing our automoy to work when each of us was able to. This is a valuable insight for programming partner projects and how to manage them during periods of increase stess outside of the project.

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
