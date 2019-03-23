## Packages loading/installation
auto_load <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# usage
packages <- c("readr", "tidyverse", "ggplot2", "ggthemes", "reshape2")
auto_load(packages)


## Question 1

## Setup

rm(list = ls())
Scores<-read.csv("Data/CFB2017_scores.csv", head=FALSE)

Scores2 <- Scores %>%
  rename(Team_A = V1, 
         Team_A_Points = V2, 
         Team_B = V3, 
         Team_B_Points = V4)


TotalGames <- nrow(Scores2)
Games_Mat <- matrix(0, 763, 763)

## Generating Transition Matrix

for (num in 1:TotalGames) {
  TeamA <- Scores2$Team_A[num]
  TeamB <- Scores2$Team_B[num]
  Points_A <- Scores2$Team_A_Points[num]
  Points_B <- Scores2$Team_B_Points[num]
  A_pct <- Points_A/(Points_A + Points_B)
  B_pct <- Points_B/(Points_A + Points_B)
  
  if (Points_A > Points_B) {
    Games_Mat[TeamA,TeamA] <- Games_Mat[TeamA,TeamA] + 1 + A_pct
    Games_Mat[TeamB,TeamA] <- Games_Mat[TeamB,TeamA] + 1 + A_pct
    Games_Mat[TeamB,TeamB] <- Games_Mat[TeamB,TeamB] + 0 + B_pct
    Games_Mat[TeamA,TeamB] <- Games_Mat[TeamA,TeamB] + 0 + B_pct
  }
  if (Points_B > Points_A) {
    Games_Mat[TeamA,TeamA] <- Games_Mat[TeamA,TeamA] + 0 + A_pct
    Games_Mat[TeamB,TeamA] <- Games_Mat[TeamB,TeamA] + 0 + A_pct
    Games_Mat[TeamB,TeamB] <- Games_Mat[TeamB,TeamB] + 1 + B_pct
    Games_Mat[TeamA,TeamB] <- Games_Mat[TeamA,TeamB] + 1 + B_pct
  }
}



for (i in 1:763)
{
  Games_Mat[i,]<-Games_Mat[i,] / (sum(Games_Mat[i,]))
}

w0 <- as.matrix(c(rep(1/763, 763)))

State <- function(t,W0,M) {
  for (step in 1:t){
    W0 <- W0%*%M
  }
  W0 <- W0/(sum(W0))
  return(W0)
}

Teams <- readLines("Data/TeamNames.txt")
Teams<-data.frame(Teams)
colnames(Teams)<-"team"


Rank <- function(W, DF, t, band) {
  Order <- order(W, decreasing = TRUE)
  Band <- Order[1:band]
  Band_Val <- W[Band]
  Band_Teams <- data.frame(DF[Band,])
  Result <- cbind(Band, Band_Teams , Band_Val)
  colnames(Result)<-c("Index", "Team", "Wt")
  Title <- paste(t , ".csv", sep="")
  write.csv(Result,file = Title)
  a <- assign(Title, Result)
  return(a)
}


## A

Output <- c(10, 100, 1000, 10000)
for (O in Output) {
  w_10 <- State(t = O,W0 = t(w0),M = Games_Mat)
  result <- Rank(W = w_10,DF = Teams, t = O, band = 25)
}

## B

E <- eigen(t(Games_Mat))
E_1 <- as.matrix(E$vectors[,1])
E_1 <- Re(E_1)
W_stationary <- t(E_1/sum(E_1))

W_new <- t(w0)
D <- rep(NA,10000)

for (i in 1:10000) {
  W_new <- W_new%*%Games_Mat
  D[i] <- sum(abs(W_new - W_stationary))
}
t <- seq(1:10000)

Data <- data.frame('W-W_stationary' = D, Time = t)

ggplot(data = Data) +
  geom_line(aes(x = Time, y = W.W_stationary)) +
  ylab(bquote(~abs( W[0] - W[inf] ))) +
  ggtitle("Markov Chain: Stable State") + theme_tufte()
ggsave("Images/Markov.png")


## Question 2

## Setup

rm(list = ls())

Name <- "Data/nyt_data.txt"
NYT <- file(Name,open="r")
LineN <- readLines(NYT)
X <- matrix(0,nrow=3012, ncol=8447)

for (i in 1:length(LineN)){
  Line <- LineN[i]
  Y1 <- strsplit(Line,",")[[1]]
  for (j in 1:length(Y1)) {
    Y2 <- strsplit(Y1[j], ":")[[1]]
    Z1 <- as.numeric(Y2[1])
    Z2 <- as.numeric(Y2[2])
    X[Z1,i] <- Z2
  }
}

close(NYT)

N <- nrow(X)
M <- ncol(X)
K <- 25
W<-matrix(runif(N*K,min=1, max=2), nrow=N, ncol=K)
H<-matrix(runif(K*M, min=1, max=2), nrow=K, ncol=M)
Penalty <- c(rep(NA,100))

## Algo for Divergence Penalty

for (i in 1:100) {
  Y <- W%*%H
  Y <- Y+10^(-16)
  Z <- X/Y
  W_t <- t(W)
  W_t <-t(apply(W_t, 1, function(x) x/sum(x)))
  H <- H*(W_t%*%Z)
  Y <- W%*%H
  Y <- Y+10^(-16)
  Z <- X/Y 
  H_t <- t(H)
  H_t <- (apply(H_t, 2, function(x) x/sum(x)))
  W <- W*(Z%*%H_t)
  Combined <- W%*%H
  Combined <- Combined + 10^(-16)
  Combined2 <- log(Combined) 
  Final <- X*Combined2 - Combined
  Penalty[i] <- -(sum(Final))
  print(paste("Iter", i, "Complete"))
}

## A

Iteration <- seq(1:100)
Data <- data.frame(Penalty, Iteration)
ggplot(data = Data)+
  geom_line(aes(x = Iteration, y = Penalty)) +
  theme_tufte() + ggtitle("Penalty Convergence")
ggsave("Images/NNMF_Divergence.png")

## B

W <- (apply(W, 2, function(x) x/sum(x)))
Name <- "Data/nyt_vocab.dat"
NYT <- file(Name, open="r")
Bag <-readLines(NYT)
close(NYT)

Output <- data.frame(matrix(NA,nrow=10))

for (k in 1:ncol(W)) {
  K <-W[,k]
  Selected <- order(K, decreasing = TRUE)[1:10]
  Select_Terms <- Bag[Selected]
  Output[,paste0("Term",k)] <- Select_Terms
  Weight <- K[Selected]
  Output[,paste0("Weights",k)] <- Weight
}

Output <- Output[,-1]
write.csv(Output, file = "Topics_2B.csv")