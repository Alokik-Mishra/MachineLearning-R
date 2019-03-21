library(tidyverse)
library(locfit)

testX <- read.csv("hw2-data/X_test.csv", header = FALSE)
testY <- read.csv("hw2-data/Y_test.csv", header = FALSE)
trainX <- read.csv("hw2-data/X_train.csv", header = FALSE)
trainY <- read.csv("hw2-data/Y_train.csv", header = FALSE)

## Naive Bayes

pi_pred <- sum(trainY)/length(t(trainY))

trainX_y1 <- trainX[(trainY == 1),]
trainX_y0 <- trainX[(trainY == 0),]

theta_y1 <- c(rep(NA, 57))
theta_y0 <- c(rep(NA, 57))

for(i in 1:57) {
  if(i < 55) {
    theta_y1[i] = sum(trainX_y1[,i])/ length(t(trainX_y1[,i]))
    theta_y0[i] = sum(trainX_y0[,i])/ length(t(trainX_y0[,i]))
  }
  else {
    theta_y1[i] = length(t(trainX_y1[,i]))/sum(log(trainX_y1[,i]))
    theta_y0[i] = length(t(trainX_y0[,i]))/sum(log(trainX_y0[,i]))
  }
}

y_pred <- c(rep(NA, 93)) 

for (i in 1:93) {
  temp1 <- c(rep(NA, 57))
  temp0 <- c(rep(NA, 57))
  for(j in 1:57) {
    if (j <= 54) {
      temp1[j] <- (theta_y1[j]^(testX[i,j])) * (1 - theta_y1[j])^(1 - testX[i,j])
      temp0[j] <- (theta_y0[j]^testX[i,j]) * (1 - theta_y0[j])^(1 - testX[i,j])
    }
    else {
      temp1[j] <- theta_y1[j]*((testX[i,j])^(-(theta_y1[j] + 1)))
      temp0[j] <- theta_y0[j]*((testX[i,j])^(-(theta_y0[j] + 1)))
    }
  }
  y_pred[i] = as.numeric((log(pi_pred) + sum(log(temp1)) > (log(1 - pi_pred) + sum(log(temp0)))))
}

table(y_pred, as.vector(t(testY)), dnn = c("Pred Y", "Real Y"))


bern_only_y0 <- theta_y0[1:54]
bern_only_y1 <- theta_y1[1:54]

x <- 1:54
y <- cbind(bern_only_y0, bern_only_y1)

df <-  data.frame(y = c(bern_only_y0, bern_only_y1),x=rep(x,2), class=factor(rep(c(0,1),each=length(x))))

library(ggplot2)
p <- ggplot(aes(group=class, col=class, shape=class), data=df) + 
  geom_hline(aes(yintercept=0)) +
  geom_segment(aes(x,y,xend=x,yend=y-y)) + 
  geom_point(aes(x,y),size=3) +
  xlab("Theta") +
  ylab("Value of theta") 
rm(y, i, j, pi_pred, temp0, temp1, x, y_pred)
p



### KNN

set.seed(112233)
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

diff <- c(NA, 4508)
temp_pred <- matrix(ncol = 20, nrow = 93)
for (i in 1:93) {
  for(j in 1:4508) {
    diff[j] = sum(abs(testX[i,] - trainX[j,]))
  }
  diff_indexed <- cbind(as.vector(diff), c(1:4508))
  diff_indexed <- diff_indexed[order(diff_indexed[,1]),]
  for(k in 1:20) {
    diff_indexed_k <- (diff_indexed[1:k,2])
    neighbors <- trainY[diff_indexed_k,]
    outcome <- sum(neighbors)/length(as.vector(neighbors))
    if(outcome != 0.5){
      temp_pred[i,k] = as.numeric((outcome > 0.5))
    }
    else{
      out_arb <- sample(0:1, 1)
      temp_pred[i,k] = out_arb
    }
  }
}


correct <- c(rep(NA,20))
for(i in 1:20){
  correct[i] <- 1 - sum(abs(temp_pred[,i] - testY))/93
}

knn <- cbind(correct, c(1:20))
knn <- data.frame(knn)

ggplot(data = knn, aes(x = V2, y = correct)) +
  geom_point() +
  geom_line() +
  xlab("k") +
  ylab("% Prediction Accuracy")



## Logistic


trainX_2 <- cbind(c(rep(1, 4508)), trainX)
trainY_2 <- replace(trainY, trainY == 0, -1)
trainX_2 <- as.matrix(trainX_2)
trainY_2 <- unname(unlist(trainY_2))
testX_2 <- cbind(crep(1,93), testX)


logis <- function(x, y, w) {
  a <- matrix(0, nrow = 1, ncol = nrow(w))
  for (i in 1:nrow(x)){
    if(y[i] == 1){
      b <- as.vector((1 - expit(x[i,]%*%w)))*y[i]*x[i,]
    }
    else{
      b <- as.vector(1 - (1 - expit(x[i,]%*%w)))*y[i]*x[i,]
    }
    a <- rbind(a,b)
  }
  a <- colSums(a)
  a <- as.matrix(a)
  return(a)
}

logis_likely <- function(x, y, w) {
  a <- c(rep(0, nrow(x)))
  for(i in 1:nrow(x)) {
    if(y[i] == 1) {
      a[i] <- log((expit(t(as.matrix(x[i,]))%*%as.matrix(w[,1]))))
    }
    else{
      a[i] <- log(1 - (expit(t(as.matrix(x[i,]))%*%as.matrix(w[,1]))))
    }
    a <- sum(a)
    return(a)
  }
}


W <- matrix(0, ncol = 10000, nrow = 58)
liklihood <- c(rep(NA, 10000))

for(i in 2:500) {
  eta <- (1 / ((10^5)*(i + 1)^0.5))
  w <- as.matrix(W[,i-1])
  W[,i] <- W[,i-1] + eta*(logis(x = trainX_2, y = trainY_2, w ))
  liklihood[i-1] <- logis_likely(x = trainX_2, y = trainY_2, w )
  
}


logis_Newton <- function(x, y, w) {
  a <- matrix(0, nrow = 1, ncol = nrow(w))
  d <- matrix(0, nrow = nrow(x), ncol = nrow(w))
  for (i in 1:nrow(x)){
    if(y[i] == 1){
      b <- as.vector((1 - expit(x[i,]%*%w)))*y[i]*x[i,]
      c <- as.vector(expit(x[i,]%*%w))*as.vector((1 - expit(x[i,]%*%w)))*x[i,]*t(x[i,])
    }
    else{
      b <- as.vector(1 - (1 - expit(x[i,]%*%w)))*y[i]*x[i,]
      c <- as.vector(1 -expit(x[i,]%*%w))*as.vector((1 - 1 - expit(x[i,]%*%w)))*x[i,]*t(x[i,])
    }
    a <- rbind(a,b)
    d <- d + c
  }
  d <- solve(d)
  a <- colSums(a)
  a <- as.matrix(a)
  final <- d%*%a
  return(final)
}

W <- matrix(0, ncol = 100, nrow = 58)
liklihood <- c(rep(NA, 100))

for(i in 2:100) {
  eta <- (1 / ((i + 1)^0.5))
  w <- as.matrix(W[,i-1])
  W[,i] <- W[,i-1] - eta*(logis_Newton(x = trainX_2, y = trainY_2, w ))
  liklihood[i-1] <- logis_likely(x = trainX_2, y = trainY_2, w )
}

y_pred <- expit(X_test %*% W[,41])

table(y_pred, testY, dnn = c("Predicted Y", "Actual Y"))
