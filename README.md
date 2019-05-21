[//]: # (Image References)

[image1]: ./images/02-guide-how-transfer-learning-v3-01.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

### Deep-Learning-Nanodegree
This repository contains material related to Udacity's Deep Learning Nanodegree Foundation program.

#### Projects

* [Predicting Bike Sharing Data](https://github.com/fpcarneiro/Deep-Learning-Nanodegree/tree/master/projects/first-neural-network)
* [Dog Breed Classifier](https://github.com/fpcarneiro/Deep-Learning-Nanodegree/tree/master/projects/dog-project)
* [Generate TV Scripts](https://github.com/fpcarneiro/Deep-Learning-Nanodegree/tree/master/projects/tv-script-generation)
* [Generate Faces](https://github.com/fpcarneiro/Deep-Learning-Nanodegree/tree/master/projects/face_generation)
* [Teach a Quadcopter How to Fly](https://github.com/fpcarneiro/Deep-Learning-Nanodegree/tree/master/projects/RL-Quadcopter)

___

### Convolutional Neural Network

#### Really Cool Resources on Visualizing CNN

If you would like to know more about interpreting CNNs and convolutional layers in particular, you are encouraged to check out these resources:
* Here's a [section](http://cs231n.github.io/understanding-cnn/) from the Stanford's CS231n course on visualizing what CNNs learn.
* Check out this [demonstration](https://aiexperiments.withgoogle.com/what-neural-nets-see) of a cool [OpenFrameworks](http://openframeworks.cc/) app that visualizes CNNs in real-time, from user-supplied video!
* Here's a [demonstration](https://www.youtube.com/watch?v=AgkfIQ4IGaM&amp;t=78s) of another visualization tool for CNNs. If you'd like to learn more about how these visualizations are made, check out this [video](https://www.youtube.com/watch?v=ghEmQSxT6tw&amp;t=5s).
* Read this [Keras blog post](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html) on visualizing how CNNs see the world. In this post, you can find an accessible introduction to Deep Dreams, along with code for writing your own deep dreams in Keras. When you've read that:
  * Also check out this [music video](https://www.youtube.com/watch?v=XatXy6ZhKZw) that makes use of Deep Dreams (look at 3:15-3:40)!
  * Create your own Deep Dreams (without writing any code!) using this [website](https://deepdreamgenerator.com/).
* If you'd like to read more about interpretability of CNNs:
  * Here's an [article](https://blog.openai.com/adversarial-example-research/) that details some dangers from using deep learning models (that are not yet interpretable) in real-world applications.
  * There's a lot of active research in this area. These [authors](https://arxiv.org/abs/1611.03530) recently made a step in the right direction.
  
#### Transfer Learning

Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.

Depending on both:

*   the size of the new data set, and
*   the similarity of the new data set to the original data set

the approach for using transfer learning will be different. There are four main cases:

1.  new data set is small, new data is similar to original training data
2.  new data set is small, new data is different from original training data
3.  new data set is large, new data is similar to original training data
4.  new data set is large, new data is different from original training data

[

![Sample Output][image1]

Four Cases when Using Transfer Learning

](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

A large data set might have one million images. A small data could have two-thousand images. The dividing line between a large data set and small data set is somewhat subjective. Overfitting is a concern when using transfer learning with a small data set.

Images of dogs and images of wolves would be considered similar; the images would share common characteristics. A data set of flower images would be different from a data set of dog images.

Each of the four transfer learning cases has its own approach. In the following sections, we will look at each case one by one.

### Demonstration Network

To explain how each situation works, we will start with a generic pre-trained convolutional neural network and explain how to adjust the network for each case. Our example network contains three convolutional layers and three fully connected layers:

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80ae2_02-guide-how-transfer-learning-v3-02/02-guide-how-transfer-learning-v3-02.png)

General Overview of a Neural Network







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

Here is an generalized overview of what the convolutional neural network does:

*   the first layer will detect edges in the image
*   the second layer will detect shapes
*   the third convolutional layer detects higher level features

Each transfer learning case will use the pre-trained convolutional neural network in a different way.

### Case 1: Small Data Set, Similar Data

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80b0b_02-guide-how-transfer-learning-v3-03/02-guide-how-transfer-learning-v3-03.png)

Case 1: Small Data Set with Similar Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is small and similar to the original training data:

*   slice off the end of the neural network
*   add a new fully connected layer that matches the number of classes in the new data set
*   randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
*   train the network to update the weights of the new fully connected layer

To avoid overfitting on the small data set, the weights of the original network will be held constant rather than re-training the weights.

Since the data sets are similar, images from each data set will have similar higher level features. Therefore most or all of the pre-trained neural network layers already contain relevant information about the new data set and should be kept.

Here's how to visualize this approach:

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80b31_02-guide-how-transfer-learning-v3-04/02-guide-how-transfer-learning-v3-04.png)

Neural Network with Small Data Set, Similar Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Case 2: Small Data Set, Different Data

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80b55_02-guide-how-transfer-learning-v3-05/02-guide-how-transfer-learning-v3-05.png)

Case 2: Small Data Set, Different Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is small and different from the original training data:

*   slice off most of the pre-trained layers near the beginning of the network
*   add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set
*   randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
*   train the network to update the weights of the new fully connected layer

Because the data set is small, overfitting is still a concern. To combat overfitting, the weights of the original neural network will be held constant, like in the first case.

But the original training set and the new data set do not share higher level features. In this case, the new network will only use the layers containing lower level features.

Here is how to visualize this approach:

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80b82_02-guide-how-transfer-learning-v3-06/02-guide-how-transfer-learning-v3-06.png)

Neural Network with Small Data Set, Different Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Case 3: Large Data Set, Similar Data

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80ba3_02-guide-how-transfer-learning-v3-07/02-guide-how-transfer-learning-v3-07.png)

Case 3: Large Data Set, Similar Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is large and similar to the original training data:

*   remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
*   randomly initialize the weights in the new fully connected layer
*   initialize the rest of the weights using the pre-trained weights
*   re-train the entire neural network

Overfitting is not as much of a concern when training on a large data set; therefore, you can re-train all of the weights.

Because the original training set and the new data set share higher level features, the entire neural network is used as well.

Here is how to visualize this approach:

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80bc3_02-guide-how-transfer-learning-v3-08/02-guide-how-transfer-learning-v3-08.png)

Neural Network with Large Data Set, Similar Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Case 4: Large Data Set, Different Data

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80bf7_02-guide-how-transfer-learning-v3-09/02-guide-how-transfer-learning-v3-09.png)

Case 4: Large Data Set, Different Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

If the new data set is large and different from the original training data:

*   remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
*   retrain the network from scratch with randomly initialized weights
*   alternatively, you could just use the same strategy as the "large and similar" data case

Even though the data set is different from the training data, initializing the weights from the pre-trained network might make training faster. So this case is exactly the same as the case with a large, similar data set.

If using the pre-trained network as a starting point does not produce a successful model, another option is to randomly initialize the convolutional neural network weights and train the network from scratch.

Here is how to visualize this approach:

[

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/April/58e80c1c_02-guide-how-transfer-learning-v3-10/02-guide-how-transfer-learning-v3-10.png)

Neural Network with Large Data Set, Different Data







](https://classroom.udacity.com/nanodegrees/nd101/parts/b9c4c3c3-b524-427b-8832-9d0748f14a2e/modules/cb574ac4-7144-4ba5-97b9-1c1265525ff8/lessons/37492b45-ee34-4dc3-a2e8-076149f92562/concepts/8c202ff3-aab5-46c3-8ed1-0154fa7b566b#)

### Optional Resources

*   Check out this [research paper](https://arxiv.org/pdf/1411.1792.pdf) that systematically analyzes the transferability of features learned in pre-trained CNNs.
*   Read the [Nature publication](http://www.nature.com/articles/nature21056.epdf?referrer_access_token=_snzJ5POVSgpHutcNN4lEtRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuP9jVts1q2g1KBbk3Pd3AelZ36FalmvJLxw1ypYW0UxU7iShiMp86DmQ5Sh3wOBhXDm9idRXzicpVoBBhnUsXHzVUdYCPiVV0Slqf-Q25Ntb1SX_HAv3aFVSRgPbogozIHYQE3zSkyIghcAppAjrIkw1HtSwMvZ1PXrt6fVYXt-dvwXKEtdCN8qEHg0vbfl4_m&tracking_referrer=edition.cnn.com) detailing Sebastian Thrun's cancer-detecting CNN!

*   Search or ask questions in [Knowledge](https://knowledge.udacity.com/).
*   Ask peers or mentors for help in [Student Hub](https://study-hall.udacity.com/).

Next

\_linkedin\_partner\_id="406745";window.\_linkedin\_data\_partner\_ids=window.\_linkedin\_data\_partner\_ids||\[\];window.\_linkedin\_data\_partner\_ids.push(\_linkedin\_partner\_id);(function(){var b=document.getElementsByTagName("script")\[0\],a=document.createElement("script");a.type="text/javascript";a.async=!0;a.src="https://snap.licdn.com/li.lms-analytics/insight.min.js";b.parentNode.insertBefore(a,b)})();

![](https://dc.ads.linkedin.com/collect/?pid=406745&fmt=gif)

!function(b,e,f,g,a,c,d){b.fbq||(a=b.fbq=function(){a.callMethod?a.callMethod.apply(a,arguments):a.queue.push(arguments)},b.\_fbq||(b.\_fbq=a),a.push=a,a.loaded=!0,a.version="2.0",a.queue=\[\],c=e.createElement(f),c.async=!0,c.src=g,d=e.getElementsByTagName(f)\[0\],d.parentNode.insertBefore(c,d))}(window,document,"script","https://connect.facebook.net/en\_US/fbevents.js");fbq("init","2197623860455254");fbq("track","PageView");

![](https://www.facebook.com/tr?id=2197623860455254&ev=PageView&noscript=1)

!function(b,e,f,g,a,c,d){b.fbq||(a=b.fbq=function(){a.callMethod?a.callMethod.apply(a,arguments):a.queue.push(arguments)},b.\_fbq||(b.\_fbq=a),a.push=a,a.loaded=!0,a.version="2.0",a.queue=\[\],c=e.createElement(f),c.async=!0,c.src=g,d=e.getElementsByTagName(f)\[0\],d.parentNode.insertBefore(c,d))}(window,document,"script","https://connect.facebook.net/en\_US/fbevents.js");fbq("init","823398221086645");fbq("track","PageView");

![](https://www.facebook.com/tr?id=823398221086645&ev=PageView
&noscript=1)
  
### RNN

#### LSTM
Now that you've gone through the Recurrent Neural Network lesson, I'll be teaching you what an LSTM is. This stands for Long Short Term Memory Networks, and are quite useful when our neural network needs to switch between remembering recent things, and things from long time ago. But first, I want to give you some great references to study this further. There are many posts out there about LSTMs, here are a few of my favorites:

* [Chris Olah's LSTM post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [Edwin Chen's LSTM post](http://blog.echen.me/2017/05/30/exploring-lstms/)
* [Andrej Karpathy's lecture on RNNs and LSTMs from CS231n](https://www.youtube.com/watch?v=iX5V1WpxxkY)

If you would like to deepen your knowledge even more, go over the following [tutorial](https://skymind.ai/wiki/lstm). Focus on the overview titled: Long Short-Term Memory Units (LSTMs).

#### Sources & References

If you want to learn more about hyperparameters, these are some great resources on the topic:

*   [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) by Yoshua Bengio
    
*   [Deep Learning book - chapter 11.4: Selecting Hyperparameters](http://www.deeplearningbook.org/contents/guidelines.html) by Ian Goodfellow, Yoshua Bengio, Aaron Courville
    
*   [Neural Networks and Deep Learning book - Chapter 3: How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters) by Michael Nielsen
    
*   [Efficient BackProp (pdf)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCun
    

More specialized sources:

*   [How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523) by Siwei Lai, Kang Liu, Liheng Xu, Jun Zhao
*   [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228) by Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas
*   [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Andrej Karpathy, Justin Johnson, Li Fei-Fei
