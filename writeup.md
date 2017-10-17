***Traffic Sign Recognition***

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kevinwestern/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

2. Include an exploratory visualization of the dataset.

I plotted a random sample of training images and also plotted a histogram over the number of examples per label.

The histogram lead me down the path of image augmentation. I noticed there were labels with far fewer training examples than others. I found the average number of examples across the labels and generated images for labels that were fewer than the average.


**Design and Test a Model Architecture**

1. Describe how you preprocessed the image data.

I first ran the the images unprocessed on the LaNet model and saw around an 86% valid accuracy. I used the hints from the notebook and converted the images to grayscale and, after some Googling, found the skimage equalist_hist to equalize the images. I applied a combination of equalizing and grayscaling as my preprocess setp.

I also made a histogram of the number of examples per label as mentioned above. I augmented the dataset by adding new images with a random rotation applied.

In retrospect I could have spent more time applying flips, rotations, blurs and zooms on the images. I decided not to for the sake of decreasing assignment load on myself. It's already tough learning Tensorflow and additional python libraries.

I came up with the architecture mostly by chance. I had forgotten to set the keep_probability to 1.0 during validation and running with the test set, so I spent two days adding and removing random layers to my architecture. I finally realized my error with the current architecture, and that had a 99.1% validation accuracy at the end of training which I decided was fine :).


2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input	| 32x32x1 Grayscale image   							| 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x25 |
| RELU ||
| Max pooling	| 2x2 stride,  outputs 14x14x25 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x50 |
| RELU ||
| Max pooling	| 2x2 stride,  outputs 5x5x50 |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 2x2x100 |
| RELU ||
| Max pooling | 2x2 stride,  outputs 2x2x100 |
| Dropout | 50%
| Fully Connected	| 400 -> 100 |
| RELU ||
| Dropout | 50% |
|	Fully Connected	| 100 -> 84	|
|	RELU | |
| Dropout | 50% |
| Fully Connected | 84 -> number of labels |


3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the defualt LaNet optimizer from the course material.

I used the provided sigma and mu values. I ended up using the following:
 * learning rate: 0.001
 * batch_size: 128
 * epochs: 50

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 99.1% 
* test set accuracy of 96.8%

The first architecture was the one used in the course material.

As mentioned above, the final architecture was found by accident. I had originally hard coded a dropout value of 50%. I forgot that dropout shouldn't be applied during validation or when trunning the test data. But I realized that after two days of randomly changing the architecture in hopes of improving accuracy past 94%.

I saw others had gotten a higher accuracy in the slack channel so I figured I must have been doing something wrong. Suggestions were made like adding dropout and an additional convolution layer which I already had.

After changing many numbers back and forth (and multiple copies made :)) I realized my problem, fixed it, and saw the validation accuracy shoot to 99.1% and decided to stick with what I had.

Unfortunately I can't say 'why' I chose any of these values. It was mainly trial and error.

I did try and larger and smaller learning rate and noticed that, with both, it never starte to converge.
 

**Test a Model on New Images**

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

See the python notebook for sample images.

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 120 km/h | 30 km/h | 
| No Entry | No Entry |
| Right of way | Pedestrians |
| Road Work	| Road Work	|
| Beware of ice/snow | Beware of ice/snow |
| Stop | Stop |


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66%. This is lower than the accuracy on the test set of 96.8%.

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 14% | 120 km/h | 
| 99% | No Entry |
| 6.6% | Right of way |
| 97%	| Road Work	|
| 63% | Beware of ice/snow |
| 98% | Stop |


### Final Thoughts
I'd like to spend more time learning numpy, skimage, scikit and pandas. I felt like I was really slow in the exploratory part of the lab. I'll try to dedicate a few nights per week putting together my own library or notebook so I can perform some quicker data analysis.

Reading the Tensorflow documentation is also..rough. I got to the point where I started experimenting with Keras because it was faster. I'd like to spend more time in Keras and use that for the next project, if possible.

I had fun with this project by applying what we've learned in class on something other than mnist. I find the mnist examples to be very boring and they never encouraged me to trying writing my own classifier. I'm looking forward to the next project.