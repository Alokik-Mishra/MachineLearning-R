library("tidyverse")
library("readr")
library("ggplot2")
setwd("/Users/alokikmishra/Desktop/Classes/Sem2/ML/HW/HW1")

## Reading in the files and converting to matrices

X_test <- read_csv("hw1-data/X_test.csv", col_names = FALSE)
X_train <- read_csv("hw1-data/X_train.csv", col_names = FALSE)
y_test <- read_csv("hw1-data/y_test.csv", col_names = FALSE)
y_train <- read_csv("hw1-data/y_train.csv", col_names = FALSE)


X_test <- as.matrix(X_test)
X_train <- as.matrix(X_train)
y_test <- as.matrix(y_test)
y_train <- as.matrix(y_train)


## Initialzing the matrcies and creating the svd
df_lam <- c(rep(0,5000))
w <- matrix(nrow = 7, ncol = 5000)
X_tr_svd <- svd(X_train)

## Looping through to create matrix of coeffieicnts and df(lamba) for different values of lambda

for(j in 1:7){
  for(i in 1: 5000) {
   w[j,i] <- (solve((diag(i,7) + (t(X_train) %*% (X_train)))) %*% t(X_train) %*% y_train)[j]
   df_lam[i] <- sum((X_tr_svd$d)^2/(i + (X_tr_svd$d)^2))
  }
}

## Creating joint dataframe for easier navigation with ggplot

w_t <- as_data_frame(t(w))
final_1 <- cbind(w_t, df_lam)


## Plotting the value of coefficients as function of df(lamba)

ggplot(final_1) +
  geom_line(aes(df_lam, V1, color = "W1"), show.legend = TRUE) +
  geom_line(aes(df_lam, V2, color = "W2"), show.legend = TRUE) +
  geom_line(aes(df_lam, V3, color = "W3"), show.legend = TRUE) +
  geom_line(aes(df_lam, V4, color = "W4"), show.legend = TRUE) +
  geom_line(aes(df_lam, V5, color = "W5"), show.legend = TRUE) +
  geom_line(aes(df_lam, V6, color = "W6"), show.legend = TRUE) +
  geom_line(aes(df_lam, V7, color = "W7"), show.legend = TRUE) +
  ylab("Coefficient value") +
  xlab("df(lambda)") +
  theme(legend.position = "bottom")
  

## Predicting and plotting RMSE of testing data with lambda = 1:50

w_2 <- w[,1:50]

RMSE = c(rep(0,50))
lambda <- c(1:50)

for(i in 1:50) {
  y_hat <- X_test %*% w[,i]
  RMSE[i] <- (sum((y_test - y_hat)^2)/42)^0.5
}
 
plot(lambda, RMSE) 

# Ploynomial Regression Modification

df_lam <- c(rep(0,500))

## Creating function to output coefficients based on p polynomial specification

poly_ridge <- function(k) {
  if(k == 1) {
  w <- matrix(nrow = 7, ncol = 500)
  X_tr_svd <- svd(X_train)
  

  for(j in 1:7) {
    for(i in 1: 500) {
      w[j,i] <- (solve((diag(i,7) + (t(X_train) %*% (X_train)))) %*% t(X_train) %*% y_train)[j]
      df_lam[i] <- sum((X_tr_svd$d)^2/(i + (X_tr_svd$d)^2))
    }
  }
  }
 else {
   w <- matrix(nrow = 6*k + 1, ncol = 500)
   X_train_original <- X_train[,-7]
    for(z in 2:k) {
      X_train <- cbind(X_train, X_train_original^z)
    }
   X_tr_svd <- svd(X_train)

   for(j in 1:(6*k + 1)) {
     for(i in 1: 500) {
       w[j,i] <- (solve((diag(i,6*k+1) + (t(X_train) %*% (X_train)))) %*% t(X_train) %*% y_train)[j]
       df_lam[i] <- sum((X_tr_svd$d)^2/(i + (X_tr_svd$d)^2))
     }
   }
   
 }
outcome <- list("W" = w, "DF_Lam" = df_lam, "X_t" = X_train)
return(outcome)
}

w_1 <- poly_ridge(1)
RMSE_1 <- c(1:500)
w_2 <- poly_ridge(2)
RMSE_2 <- c(1:500)
w_3 <- poly_ridge(3)
RMSE_3 <- c(1:500)
lambda_500 <- c(1:500)


for(i in 1:500) {
  y_hat <- X_test %*% w_1$W[,i]
  RMSE_1[i] <- (sum((y_test - y_hat)^2)/42)^0.5
}

for(i in 1:500) {
  y_hat <- cbind(X_test, X_test[,-7]^2) %*% w_2$W[,i]
  RMSE_2[i] <- (sum((y_test - y_hat)^2)/42)^0.5
}

for(i in 1:500) {
  y_hat <- cbind(X_test, X_test[,-7]^2, X_test[,-7]^3) %*% w_3$W[,i]
  RMSE_3[i] <- (sum((y_test - y_hat)^2)/42)^0.5
}

Total_2 <- as.data.frame(cbind(lambda_500, RMSE_1, RMSE_2, RMSE_3))

ggplot(Total_2) +
  geom_line(aes(lambda_500, RMSE_1, color = "p = 1"), show.legend = TRUE) +
  geom_line(aes(lambda_500, RMSE_2, color = "p = 2"), show.legend = TRUE) +
  geom_line(aes(lambda_500, RMSE_3, color = "p = 3"), show.legend = TRUE) +
  ylab("RMSE") +
  xlab("lambda") +
  theme(legend.position = "bottom")
