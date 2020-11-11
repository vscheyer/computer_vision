# CV Project: Comparison of FCNN & CNN for Image Classification

## Project Goals

Our team is relatively new to the space of machine learning and we selected learning goals to help give us an overview of neural nets.  Our learning goals were to:

- Orient ourselves to neural nets
- Understand the difference between FCNN & CNNs
- Become familiar with the challenges of image classification
- Practice selecting, organizing, and using a dataset

## Implementation 
To achieve our learning goals, we implemented a fully connect neural net and a convolution neural net to classify road signs. The dataset we used is from the publiclly avaliable portion of the Chinese Traffic Sign Recognition database [1].  It contains 5998 images making up 58 different categories.  The images only contain single traffic signs and the road signs are (mostly) centered in the image. 
<p align="center">
  <img width="400" height="400" src="https://github.com/vscheyer/computer_vision/blob/main/documentation/dataset.PNG">
</p>
We based our FCNN and CNN off of a tensorflow tutorial [2].  Our goal was to great simple neural nets that could be easily modfied to facilitae our understanding of the differnces beteen FCNN and CNNs.  After building our intial models, we compared how different modifications changed the perfoamce of each neural net

### Fully Connected Neural Net
**Scenario 1:**    
**Scenario 2:**

### Convolutional Neural Net
**Scenario 1:**  
**Scenario 2:**

### Comparison
**Scenario 1:**    
**Scenario 2:**

## Design Decisions
In our project, we intentionally implemented the neural net that is considered to be a poor choice for the application of image classification. In doing our intial project research we learned why FCNNs stuggle with image classification.  This is because the FCNN treats each individual pixel as a "feature." The higher resolution the image, the more input layer grows were each input is a single pixel from the image.  All of those inputs have weights in the first hidden layer and the network then has many paramters.  A network with a large number of parameter takes longer to train and FCNN used for image classification have larger error than CNNs.

Convolutional neural nets are able to do feature extraction by grouping relevant pixels togther and training a fewer number of weights to describe the smae number of pixels as an FCNN.  This means they require less memory and are able to training on larger netorks.  

We wanted to witness the difference in performance and implemention of FCNN and CNNs first hand for image classification which led us to implement both neural nets using tensorflow to be able to compare the networks performance.

## Challenges

**Picking the right resource**:  Having never used tensorflow before, we relied heavily on turorials and stack overflow.  While there are many resoucres out there, we struggled to find ones that were both accessible and tackling similiar problems that we were encountering.  We did use Tensorflow documentation queit a bit but many built-in functions have some many parameters than can be tuned it was a challenge to determine what was relevant for our project. As we learned more and more about neural nets and became more familiar with tensorflow, we were able to interpert more of the resources but there was intially a steep learning curve.

**Dataset organizing**:  Building off of experiences in prior courses, we thought that orgnaizing our selected dataset and seperating it into testing and training data.  Once we selected the Chinese Traffic Sign Recognition dataset, we wrote a script to seperate training and testing data for categories the user specified.  This seemed like a helpful tool to have before we even began to work with the library.  However, after digging into some tesnroflow examples, we learned that the library already had built-in function to help manage and seperate datasets. The way we orgnaized our data was incompatible with the existing way to load datasets so we ended up modifying our original script signifigatly.  While the final implementation was simpler, we did do uneceddary work that could have been avoided if we had looked closley at tensorflow examples prior to jumping in.

## Next Steps/Takeaways
What would you do to improve your project if you had more time?

Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.

In the case of image classification, this is desirable because less memory is needed for application is self-driving vehicles

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
