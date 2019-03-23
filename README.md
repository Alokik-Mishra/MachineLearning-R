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
<img src="/Assignment1/Images/L2_reg_dflambda.png" width=354 align="right" height="300"> The first part of assignment 1 involves a theoretical answer which is not provided in this repo. The seocnd part involves the coding of ridge regression (OLS with L2 penalty) and comparing various hyperparameter values. The dataset used for training and testing contains features about various cars used to predict miles per gallon. The graph shows the variation of the weights with different values of the df(<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a>). Higher values of lambda (and thus lower df(<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a>) pulls the weights (coefficients) to 0 by increasing the penalty. One important thing to notice is that for some values the not only the magnitudes, but the signs on the weights change as well. 
\
\
\
\
\
Next we turn to both the validation of the model using RMSE on the testing set along with trying out a polynomial model with L2 regularization. 
<img src="/Assignment1/Images/RMSE_lambda.png" width=354> <img src="/Assignment1/Images/RMSE_lambda_poly.png" width=354> 




<a name="as2"></a>
## Assignment 2

<a name="as3"></a>
## Assignment 3

<a name="as4"></a>
## Assignment 4

<a name="as5"></a>
## Assignment 5
