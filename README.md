# Data-603-Platforms-for-Big-Data-Processing
The goal of this course is to introduce methods, technologies, and computing platforms for performing data analysis at scale. 

Topics include the theory and techniques for data acquisition, cleansing, aggregation, management of large heterogeneous data collections, processing, information and knowledge extraction. Students are introduced to map-reduce, streaming, and external memory algorithms and implementations using Hadoop and its eco-system (MapReduce, HBase, Spark, and others). Students will gain practical experience in analyzing large existing databases.


            Classification of Big Cats from Google Image Dataset DATA 603 Platforms for Big Data Processing University of Maryland 

Introduction
A semester-long project on big data processing doing a simple classification model. The binary classification will be on predicting an image to be a big cat or not. The list of big cats within the Google Image Dataset includes Jaguar, Lynx, Tigers, Lions, Leopards, and Cheetahs.
Dataset

The Google Open Images V5 features segmentation masks for 2.8 million object instances in 350 categories. Unlike bounding-boxes, which only identify regions in which an object is located, segmentation masks mark the outline of objects, characterizing their spatial extent to a much higher level of detail. These masks cover a broader range of object categories and a more significant total number of instances than any previous dataset.
The segmentation masks on the training set (2.68M) have been produced by an interactive segmentation process, where professional human annotators iteratively correct the output of a segmentation neural network. This is considered efficient than manual drawing alone, while at the same time delivering accurate masks. In addition to the masks, 6.4M new human-verified image-level labels are present, reaching a total of 36.5M over nearly 20,000 categories.

Why ResNet50?

ResNet50 is a 50 layer deep, pre-trained Deep learning model for image classification of the Convolutional Neural Network, mostly used to analyze visual imagery. A pre-trained model is a more practical approach than collecting loads of data and training the model ourselves. It is trained on a million images of 1000 categories from the ImageNet database. The model has over 23 million trainable parameters, which indicates a deep architecture that makes it better for image recognition and classification. Also, ResNet50 has excellent generalization performance with fewer error rates than other pre-trained models like AlexNet, GoogleNet, or VGG19. The network has an image input size of 224-by-224.
Further, in a deep convolutional neural network like VGG16, several layers are stacked and trained to the task, and the network learns several low/mid/high-level features at the end of its layers. Whereas in residual learning, instead of trying to learn some features, we try to learn some residual by subtracting the feature learned from the input of the layer. This is done by directly connecting the input of the nth layer to some (n+x) the layer. Training these kinds of networks is more accessible than training simple deep convolutional neural networks, and the problem of degrading accuracy is also resolved.

MOBILENET

MobileNet is a separable convolution (conv) module which is composed of depth wise and point wise conv. Depth wise conv is performed independently for every input channel of the image, this helps in reducing the computational cost by omitting conversation in channel.
Mobile net independently performs conv. MobileNet is efficiently used to predict the category we want if available in ImageNet.
Advantages of using MobileNet:
• MobileNets is easy to apply as it is light weight.

• Uses depth wise separable conv, which is a streamlined architecture and human
understandable.

• Reduces size of the network

• Reduces number of parameters, resulting in availability of more space.

• Small Convolutional Neural Network, therefore, it does not forget the weights at large
scale and gives out reliable predictions.

• Fast in performance as compared to few other pre-trained models.


 
