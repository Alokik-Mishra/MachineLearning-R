# Machine Learning (R)
### Spring 2018 - Data Science Institute, Columbia University
### Prof. John Paisely

## Overview
1. [Course description](#desc)
2. [Tech/framework](#tech)
3. [Assignment 1](#as1)
4. [Assignment 2](#as2)
4. [Assignment 3](#as3)
4. [Assignment 4](#as4)
4. [Assignment 5](#as5)


<a name="desc"></a>
## Course Description
ELEN 4903 was a 1 semester class on machine learning theory. The course was almost fully theoretical, focusing on probability, optimization, and analysis. We covered most of the algorithms in use for the span of machine learning models, in a method grounded in rigorous mathematics. The goal of the class was to understand what is happening under the hood when using these models. The assignments however did involve some questions related to coding. All of the coding in this repo is done in R, the models are coded from scratch and all pre-made ML packages (caret, etc) are avoided. The topics covered in the course inclde:

* Regression
* Maximum Liklihood and Maximum a posteriori
* Regularization and the bias-variance tradeoff
* Classficiation
  * Naive bayes
  * K-nn
  * Perceptron and Logistic
  * Laplace approxmiation and Bayesian logistic
* Feature expansions and kernals
  * Kernalized perceptron
  * Kernalized knn
  * Gaussian processes
* SVMs
* Trees, bagging, and random forests
* Boosting
* The Expectation Maximization (EM) algorithm
* Clustering and Gaussian mixture models (GMMs)
* Matrix factorization and recommender systems
* Topic modelling and the LDA algorithm
* Nonnegative matrix factorization
* Principal components analysis (PCA)
* Markov chains
* Hidden Markov models (HMMs) and the Kalman filter

<a name="tech"></a>
## Tech/Framework
All of the coding is done in R. Some packages are used for data wrangling and visualization but the actual algorithms are written from scratch. A function is included in the headers for all the code that install and/or loads the required packages.

<a name="as1"></a>
## Assignment 1
<img src="/Assignment1/Images/L2_reg_dflambda.png" width=400 align="right" height="300"> The first part of assignment 1 involves a theoretical answer which is not provided in this repo. The seocnd part involves the coding of ridge regression (OLS with L2 penalty) and comparing various hyperparameter values. The dataset used for training and testing contains features about various cars used to predict miles per gallon. The graph shows the variation of the weights with different values of the df(<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a>). Higher values of lambda (and thus lower df(<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a>) pulls the weights (coefficients) to 0 by increasing the penalty. One important thing to notice is that for some values the not only the magnitudes, but the signs on the weights change as well. 
\
\
\
\
\
Next we turn to both the validation of the model using RMSE on the testing set along with trying out a polynomial model with L2 regularization. 

<img src="/Assignment1/Images/RMSE_lambda.png" width=400> <img src="/Assignment1/Images/RMSE_lambda_poly.png" width=400> 

The results above are fairly intutitive. The OLS model seems to perform best without any regularization, however once we add more parameters to the model through polynomials, some regularization helps in the prediction on the testing set by reducing overfitting. The best model on the RMSE metric seems to be a second order polynomial with a value for the hyperparameter <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> being around 15.



<a name="as2"></a>
## Assignment 2
The second assignment shifts focus from regression to classification. The first problem involves a theoretical ML derivation of the naive Bayes classifier, and this part of the solution is not included in this repo. The second part of the assignment <img src="/Assignment2/Images/Nbayes.png" width=400 align="right" height="300"> involves coding various classifiers on a dataset consisting spam and nonspam emails with words as features.

The first classifier is the naive bayes we derived in the first part of the assignment. This chart shows the class conditional weights for the different features. A value of 1 indicates the email is labelled spam and 0 nonspam. The different weights represent the different conditional liklihood of the different classes for each feature. For example, the features 16 and 52 represent the words "free" and "!" respectively. It makes sense that given a high presence of these words, the email is more likely to b spam (y = 1) than not.
 \
 \
 \
 Next I implement a non-probabalistic classfier, he K nearest neighbors (k-nn) classifier. The performance of the model on the testing set with varying values of k are shown below. 
 
<img src="/Assignment2/Images/KNN.png" width=400 align="center" height="300">

The last classifer I implement, is one of the most commonly used binary classifiers, the logistic regression. I implement the regression using two methods, the normal stochastic gradient decsent (sdg) method and Newton's method. The log liklihood of each as a function of iterations is shown below.


<img src="/Assignment2/Images/Log_liklihood.png" width=400> <img src="/Assignment2/Images/Log_liklihood_newton.png" width=400> 

<a name="as3"></a>
## Assignment 3
Assignment 3 involves two coding problems, the first part is a gaussian process, and the second is the implmentation of a boosting algorithm on a simple OLS model. In the first part I use the same dataset as that in the first assignment, using various car characteristics as features to predict miles per gallon.


<a name="as4"></a>
## Assignment 4

<a name="as5"></a>
## Assignment 5
