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
Assignment 3 involves two coding problems, the first part is a gaussian process, and the second is the implmentation of a boosting algorithm on a simple OLS model. In the first part I use the same dataset as that in the first assignment, using various car characteristics as features to predict miles per gallon. The second questions uses an room occupancy dataset found [here](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)

The gaussian process algorithm involves creating kernal matrices using the radial basis function and then using that to predict values based on the hyperparameters _b_ and <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^2" title="\sigma^2" /></a> . Below I show a scatter plot with the predicted mean based on just using one of the features (car weight).

<img src="/Assignment3/Images/gaussian.png" width=400>

<img src="/Assignment3/Images/boost_sampledist.png" width=300 align="right">The second part of the assignment involved using the Adaboost algorithm on a simple OLS model. I used 1500 iterations for the boosting. The distribution of the sampled observations can be seen on the right. It is clear that the sampling was highly uneven with some of the observations having a alrge influence on weight optimization. Below I show <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_t" title="\alpha_t" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon_t" title="\epsilon_t" /></a> as a function of _t_.

<img src="/Assignment3/Images/boost_AandE.png" width=400>


<a name="as4"></a>
## Assignment 4
In assignment 4 we switch gears from supervised learning to unsupervised learning. In the first part of the assignment I first generate three Gaussians on **R<sup>2</sup>** with different mean and mixing weights but the same covariance matrices. I then implement the k-means clustering algorithms for various _k_ clusters which allows us to test how well the unsupervised algorithm works whilst knowing the true data generating process. 

<img src="/Assignment4/Images/Clusters1.png" width=400> <img src="/Assignment4/Images/Clusters2.png" width=400>

Above, I show both the k-means objective function as a function of iterations (left) and the final cluster assignment after 20 iterations of the data sample for two different _k_ (right). From the left we can see that increasing the number of clusters will montonically decrease the loss function, however this should not be used as the only metric, as we know the true _k_ is 3  not 5, which can be seen from the image on the right as well as our prior information on the data generating process.

In the second part of the assignment I implement a matrix factorization algorithm on a user-movie dataset. The log liklihood as a function of iterations for different intitalizations can be seen below.

<img src="/Assignment4/Images/Matrix_Factor.png" width=400>

All the iterations are pretty similar, however I use the highest one (trial 7) and then examine the movies "Star Wars", "My Fair Lady", and "Goodfellas". The tables with the closest movies (recommendations) to each can be seen below.


| Star Wars                                 | My Fair Lady                                     | GoodFellas                             |
|-------------------------------------------|--------------------------------------------------|----------------------------------------|
| Empire Strikes Back, The (1980)           | Return of Martin                                 | Casino (1995)                          |
| Raiders of the Lost Ark (1981)            | Guerre, The (Retour de Martin Guerre, Le) (1982) | Full Metal Jacket (1987)               |
| Return of the Jedi (1983)                 | Carrington (1995)                                | Bonnie and Clyde (1967)                |
| Usual Suspects, The (1995)                | Snow White and the Seven Dwarfs (1937)           | Good, The Bad and The Ugly, The (1966) |
| Indiana Jones and the Last Crusade (1989) | Victor/Victoria (1982)                           | Apocalypse Now (1979)                  |
| Day the Earth Stood Still, The (1951)     | Fantasia (1940)                                  | Carlito's Way (1993)                   |
| My Man Godfrey (1936)                     | Charade (1963)                                   | Swingers (1996)                        |
| Prefontaine (1997)                        | Only You (1994)                                  | Godfather, The (1972)                  |
| Blues Brothers, The (1980)                | Mary Poppins (1964)                              | Godfather: Part II, The (1974)         |
| Back to the Future (1985)                 | Singin' in the Rain (1952)                       | People vs. Larry Flynt, The (1996)     |

Based on a cursory familiarity with some of the movies above it seems the algorithm was implemented with a fair degree of success using only collaborative filtering and no movie of user characteristics.

<a name="as5"></a>
## Assignment 5
In the final assignment I implment a markov chain model to rank college football teams based on their performances vis-a-vis each other. The second part of the assignment involves using nonnegative matrix factorization to do topic modelling on 8447 articles from the new york times. 

<img src="/Assignment5/Images/Markov.png" width=300 align="right"> In the first part of the assignment I implement markov chain model looking the performance of 763 college football teams from the 2017 season and attempt to rank them. The chart on the right shows the difference between the weight at iteration _t_ and the steady state weight as a function of t. The table below shows the rankings by the markov chain algorithm and the rankings from [AP poll](https://www.sbnation.com/college-football/2018/1/9/16867316/college-football-rankings-final-2017-top-25). The outcomes are quite similar.

| Algo Rank | Team           | Voting Rank |
|-----------|----------------|-------------|
| 1         | Alabama        | 1           |
| 2         | Georgia        | 2           |
| 3         | OhioState      | 5           |
| 4         | Clemson        | 4           |
| 5         | Oklahoma       | 3           |
| 6         | Wisconsin      | 7           |
| 7         | CentralFlorida | 6           |
| 8         | Auburn         | 10          |
| 9         | PennState      | 8           |
| 10        | NotreDame      | 11          |


In the second part of the assignment, I use non-negative martix factorization (NMF) to do topic modelling on NYT articles. Below I show the objective function value as a function of iterations.

<img src="/Assignment5/Images/NNMF_Divergence.png" width=300>

Next I show the top 10 words from 5 topics. As can be seen there is some coherence to the topics. The first one seems to be about logistics, the second is media, the third is business, the fourth is social/community, and the fifth is policy.

| Topic 1     | Topic 2     | Topic 3   | Topic 4  | Topic 5    |
|-------------|-------------|-----------|----------|------------|
| agreement   | television  | company   | father   | money      |
| plan        | medium      | business  | mrs      | state      |
| meeting     | advertising | industry  | son      | pay        |
| agree       | magazine    | executive | graduate | budget     |
| issue       | network     | product   | daughter | program    |
| negotiation | news        | customer  | mother   | tax        |
| decision    | broadcast   | service   | marry    | cost       |
| proposal    | video       | president | receive  | government |
| meet        | newspaper   | base      | retire   | bill       |
| official    | commercial  | sell      | degree   | law        |

