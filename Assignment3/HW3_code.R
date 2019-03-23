#### Author: Alokik Mishra
#### Assignment: Machine Learning Homework 3
#### Date: 03/18/2018
##################################
## Packages loading/installation
auto_load <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# usage
packages <- c("readr", "tidyverse", "ggplot2", "ggthemes")
auto_load(packages)


###################################
#### Question 1: Gaussian Processes
###################################
## Importing Data
X_test <- read_csv("Data/gaussian_process/X_test.csv", col_names = FALSE)
Y_test <- read_csv("Data/Y_test.csv", col_names = FALSE)
X_train <- read_csv("Data/gaussian_process/X_train.csv", col_names = FALSE)
Y_train <- read_csv("Data/gaussian_process/Y_train.csv", col_names = FALSE)


## Setting up function to generate Kernel matrix
Gauss_Kernel <- function(x, X, b){
  K <- matrix(NA, nrow = nrow(x), ncol = nrow(X))
  for(i in 1:nrow(x)){
    for(j in 1:nrow(X)) {
      K[i,j] <- exp((-1/b)*((sqrt(sum((x[i,]-X[j,])^2)))^2))
    }
  }
  return(as.matrix(K))
}

## Setting up function to generate  predicted values (mu_x) based on Kernel 
Gauss_Kernel_predict <- function(x, X, sigma2, b, Y){
  mu_x <- c(rep(NA, nrow(x)))
  KernX <- Gauss_Kernel(X,X,b)
  Kern_inv <- solve(as.matrix(sigma2*diag(1, nrow(X)) + KernX))
  for(i in 1:nrow(x)){
    x_tr <- x[i,]
    mu_x[i] <- (as.matrix(Gauss_Kernel(x_tr,X,b))) %*% Kern_inv %*% as.matrix(Y)
  }
  return(mu_x)
}

## Calculating RMSE from predicted values
Gauss_Kernel_predict_RMSE <- function(x, X, sigma2, b, Y, y){
  y_pred <- Gauss_Kernel_predict(x, X, sigma2, b, Y)
  RMSE <- sqrt((sum((y_pred-y)^2))/nrow(x))
  return(RMSE)
}


## Initiating vectors with b and sigma2 and matrix for RMSE comparision
B <- c(seq(5, 15, by = 2))

Sigma2 <- c(seq(0.1, 1, by = 0.1))

RMSE_comp <- matrix(data = NA, nrow = length(B), ncol = length(Sigma2), dimnames = list(B, Sigma2))

## Running a loop to fill in values for RMSE with different parameters
## This loop takes approximately 90 minutes to run.
for(i in 1:length(B)) {
  for(j in 1:length(Sigma2)){
    B_tr = B[i]
    Sig_tr = Sigma2[j]
    RMSE_comp[i,j] <- Gauss_Kernel_predict_RMSE(x = X_test, X = X_train, sigma2 = Sig_tr, b = B_tr, Y = Y_train, y = Y_test )
  }
}

RMSE_comp

## Rerunning algorithm with only 4th dim of X
X_train_4 <- X_train[,4]
y_pred <- Gauss_Kernel_predict(x = X_train_4, X = X_train_4, b = 5, sigma2 = 2, Y = Y_train)

plot_df <- cbind(X_train_4, y_pred, Y_train)

## Ploting
ggplot(plot_df) +
  geom_point(aes(plot_df$X4, plot_df$X1, col = "Y")) +
  geom_line(aes(plot_df$X4, plot_df$y_pred, col = "Predictive Mean - Gaussian")) +
  xlab("Car Weight") +
  ylab("Actual MPG") +
  theme_tufte() +
  ggtitle("Gaussian Process") +
  theme(legend.position = "bottom")
ggsave("Images/gaussian.png")


###################################
#### Question 2: Boosting
###################################

## Creating Function for OLS
rm(list = ls())

OLS_pred_class <- function(x, X, Y){
  beta <- solve(as.matrix(t(X)) %*% as.matrix(X)) %*% as.matrix(t(X)) %*% as.matrix(Y)
  pred_OLS <- sign(as.matrix(x) %*% as.matrix(beta))
  list(pred = pred_OLS, b = beta)
}
  
## Importing Data
  
  X_test <- read_csv("Data/boosting/X_test.csv", col_names = FALSE)
  Y_test <- read_csv("Data/boosting/Y_test.csv", col_names = FALSE)
  X_train <- read_csv("Data/boosting/X_train.csv", col_names = FALSE)
  Y_train <- read_csv("Data/boosting/Y_train.csv", col_names = FALSE)
  
  const_tr <- c(rep(1, nrow(X_train)))
  const_test <- c(rep(1, nrow(X_test)))
  
  X_train_int <- cbind(const_tr, X_train)
  X_test_int <- cbind(const_test, X_test)
  
  
## Creating Boosting function
## Parameters - Traning and Testing data, Number of Interations
## Outputs -  Epsilon(T), Alpha(T), Boosted Training and Testing Errors, T-specific Training and Testing
##            Error, Names of Observations at each T, Predicted Y Training and Testing Values at each
##            T.
  
BoostOLS <- function(Rounds, testx, testy, trainx, trainy){
    
    n <- nrow(trainx)
    d <- ncol(trainx)
    n_test <- nrow(testx)
    #W <- matrix(NA, nrow = n, ncol = Rounds+1)
    E <- c(rep(NA, Rounds))
    Alpha <- c(rep(NA, Rounds))
    TR_Error <- c(rep(NA, Rounds))
    Test_Error <- c(rep(NA, Rounds))
    Names <- matrix(NA, nrow = n, ncol = Rounds)
    Beta <- matrix(NA, nrow = d, ncol = Rounds)
    Pred_y_tr <- matrix(NA, nrow = n, ncol = Rounds)
    Pred_y_test <- matrix(NA, nrow = n_test, ncol = Rounds)
    Pred_y_tr_boosted <- matrix(NA, nrow = n, ncol = Rounds)
    Pred_y_test_boosted <- matrix(NA, nrow = n_test, ncol = Rounds)
    TR_Error_boosted <- c(rep(NA, Rounds))
    Test_Error_boosted <- c(rep(NA, Rounds))
    
    
    W <- c(rep(1/n, n))  ## Uniform starting weights
    names <- c(1:length(W)) ## Naming the observations by index
    All <- cbind(names, trainx, trainy)
    
    for(i in 1:Rounds) {
      if(i < Rounds){
        All_Sampled <- All[sample(names, size = n, 
                                  replace = TRUE, prob = W),]  ## Drawing a distr according to weights
        Names[,i] <- All_Sampled[,1]
        last_row <- ncol(All_Sampled)
        Y <- All_Sampled[,last_row]
        X <- All_Sampled[,3:last_row-1]
        f_t <- lm(Y ~ as.matrix(X) - 1)     ## OLS
        Beta[,i] <- coefficients(f_t)       ## Storing Coefficients
        Pred_y_tr[,i] <- sign(as.matrix(trainx) %*% Beta[,i])   ## Predicting using f_t
        E[i] <- sum(W*(as.matrix(trainy) != Pred_y_tr[,i]))  ## Calculating Epsilon
        if(E[i] >= 0.5) {                         ## Adjusting eplison
          Beta[,i] <- -1 * Beta[,i]
          Pred_y_tr[,i] <- sign(as.matrix(trainx) %*% Beta[,i])
          E[i] <- sum(W*(trainy != Pred_y_tr[,i]))
        }
        Alpha[i] <- 0.5*log((1 - E[i])/E[i])
        Pred_y_tr_boosted[,i] <- sign(as.matrix(Pred_y_tr[,1:i]) %*% (Alpha[1:i]))  ## Calculating Boosted Prediction
        e_power <- -Alpha[i] * trainy * Pred_y_tr[,i]
        W <- W*sapply(e_power, exp)     ## Generating new Weight
        W <- W / sum(W)       ## Normalzing the Weight
        TR_Error[i] <- sum(as.matrix(trainy) != Pred_y_tr[,i]) / length(Y)  ## T-specific Training Error
        Pred_y_test[,i] <- sign(as.matrix(testx) %*% Beta[,i])  
        Test_Error[i] <- sum(testy != Pred_y_test[,i]) / nrow(testy)  ## T-specific Testing Error
        Pred_y_test_boosted[,i] <- sign(as.matrix(Pred_y_test[,1:i]) %*% (Alpha[1:i]))
        TR_Error_boosted[i] <- sum(trainy != Pred_y_tr_boosted[,i]) / length(Y) ## Boosted Training Error
        Test_Error_boosted[i] <- sum(testy != Pred_y_test_boosted[,i]) / nrow(testy)  ## Boosted Testing Error
      } else{       ## For last iteration just replicating values so 'i+1' doesnt cause problems
        E[i] <- E[i-1]
        Alpha[i] <- Alpha[i-1]
        Test_Error[i] <- Test_Error[i-1]
        TR_Error[i] <- TR_Error[i-1]
        TR_Error_boosted[i] <- TR_Error_boosted[i-1]
        Test_Error_boosted[i] <- Test_Error_boosted[i-1]
      }
    print(paste("Round", i, "complete"))
    }
    
    
    list(E = E, Alpha  = Alpha, Train_err = TR_Error, Test_err = Test_Error, 
         W = W, Pred_y_train = Pred_y_tr, Pred_y_test = Pred_y_test, Samples = Names, 
         Boost_test_error = Test_Error_boosted, Boost_train_error = TR_Error_boosted)
    
  }


## Running Function for T = 1500 with data given (~ 1 minute runtime)

BOOST <- BoostOLS(Rounds = 1500, testx = X_test_int, trainx = X_train_int, trainy = Y_train, testy = Y_test)


## Plotting Boosted Training and Tested Error

T <- c(1:1500)
tr_boost_err <- BOOST$Boost_train_error
test_boost_err <- BOOST$Boost_test_error

error_comp <- cbind(T, tr_boost_err, test_boost_err)
error_comp <- data.frame(error_comp)

ggplot(error_comp) +
  geom_line(aes(T, tr_boost_err, col = "Boosted Training Error")) +
  geom_line(aes(T, test_boost_err, col = "Boosted Testing Error")) +
  xlab("Iteration") +
  ylab("Misclassification %") + 
  theme_tufte() +
  ggtitle("Boosting Error") +
  theme(legend.position = "bottom")
ggsave("Images/boost_error.png")

## Plotting Upper Bound

upper_bound <- c(rep(NA, 1500))
e_t <- BOOST$E

for(i in 1:1500){
  upper_bound[i] = exp(-2 * sum((0.5 - e_t[1:i])^2))
}

upper_calc <- data.frame(cbind(T, upper_bound))

ggplot(upper_calc)+
  geom_line(aes(T, upper_bound, col = "Upper Bound")) +
  ylab("Upper Bound") +
  xlab("Iteration") +
  theme_tufte() +
  ggtitle("Boosting : Upper Bound") +
  theme(legend.position = "bottom")
ggsave("Images/boost_upperbound.png")

## Plotting Distribution of Observations

Observations <- data.frame(BOOST$Samples)

Observations <- Observations %>%
  gather(Temp, 1:1500)

Observations <- data.frame(Observations[,2])

#hist(Observations, breaks = 200, xlab = "Observation Number")

ggplot(Observations, aes(x=Observations...2.)) +
  geom_histogram() + 
  ylab("Count") +
  xlab("Sample Number") +
  theme_tufte() +
  ggtitle("Boosting : Sample Distribution") +
  theme(legend.position = "bottom")
ggsave("Images/boost_sampledist.png")

## Plotting Alpha and Epsilon as function of T

alpha_t <- BOOST$Alpha


Error <- data.frame(cbind(T, e_t))
Alpha <- data.frame(cbind(T, alpha_t))

Combined <- left_join(Error, Alpha)

ggplot(Combined) +
  geom_line(aes(T, e_t, col = "Epsilon")) +
  geom_line(aes(T, alpha_t, col = "Alpha")) +
  ylab("Value") +
  xlab("Iteration") +
  theme_tufte() +
  ggtitle("Boosting : Alpha and Epsilon") +
  theme(legend.position = "bottom")
ggsave("Images/boost_AandE.png")


ggplot(Error) +
  geom_line(aes(T, e_t, col = "Epsilon")) +
  ylab("Epsilon") +
  xlab("Iteration") 


ggplot(Alpha) +
  geom_line(aes(T, alpha_t, col = "Alpha")) +
  ylab("") +
  xlab("Iteration")
