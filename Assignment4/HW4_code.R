## Packages loading/installation
auto_load <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# usage
packages <- c("readr", "tidyverse", "ggplot2", "ggthemes", "mvtnorm", "reshape2")
auto_load(packages)


set.seed(1234)

############################
## Question 1
############################


### First we create a function for K-means clustering


KM_Cluster<-function(k, X, N, n_iter)
{
  c <- rep(0,N)
  Mu <- matrix(runif(2*k), ncol=k) 
  Iter <- n_iter
  Outcome <-c() 
  k <- k
  for (t in 1:Iter)
  {
    for (j in 1:N)
    {
      DMat <- t(Mu)
      DMat <- rbind(DMat,X[j,])
      DMat2 <- as.matrix(dist(DMat))
      DMat2 <- DMat2[k+1,1:k]
      c[j] <- which.min(DMat2)
    }
    class_counts <- table(c)
    Y <- matrix(0,nrow = N, ncol = k)
    for (i in 1:N)
    {
      for (j in 1:k)
      {
        if (c[i] == j)
        {
          Y[i,j] = 1
        }
      }
    }
    mat1 <- t(X)%*%Y
    mat2 <- mat1
    for (j in 1:k)
    {
      num <- class_counts[j]
      mat2[,j] <- (1/num)*mat2[,j]
    }
    for(j in 1:k)
    {
      Mu[,j] <- mat2[,j]
    }
    Diff <- rep(0,k)
    mat1 <- t(X)
    for (i in 1:N)
    {
      u_1 <- as.matrix(mat1[,i])
      for (j in 1:k)
      {
        if (c[i] == j)
        {
          u_2 <- as.matrix(u_1-Mu[,j])
          Diff_2 <- as.numeric(t(u_2)%*%u_2)
          Diff[j] <- Diff[j] + Diff_2
        }
      }
    }
    Outcome[t] <- sum(Diff)
  }
  Output <- list(Outcome, c)
  return(Output)
}

## Creating the data

Mu_1 <- as.vector(c(0,0))
Mu_2 <- as.vector(c(3,0))
Mu_3 <- as.vector(c(0,3))
Sigma_1 <- matrix(c(1,0,0,1), nrow=2, ncol=2)
Sigma_2 <- Sigma_1
Sigma_3 <- Sigma_1
Sigma <- matrix(c(4,2,2,3), ncol=2)

Pi<-c(0.2,0.5,0.3)

N <- 500
Samples = runif(N)

Data_sim = matrix(NA, nrow=N, ncol=2)

for(i in 1:N){
  if(Samples[i] < Pi[1]){
    Data_sim[i,] = rmvnorm(1, mean = Mu_1, sigma = Sigma_1)
  } else if(Samples[i] < Pi[2]){
    Data_sim[i,] = rmvnorm(1, mean = Mu_2, sigma = Sigma_2)
  } else {
    Data_sim[i,] = rmvnorm(1, mean = Mu_3, sigma = Sigma_3)
  }
}

### Running Data through K-means function
Clusters <-data.frame(nrow=20, ncol=5)

for (i in 1:5)
{
  C_vals <- KM_Cluster(k = i, N = 500, X = Data_sim, n_iter = 20)[1]
  Clusters<-cbind(Clusters, C_vals)
}

### Reformatting data to plot (a)

Clusters <- Clusters[,4:7]
colnames(Clusters)<-c("2", "3", "4", "5")

Trial <- c(1:20)
Clusters <- cbind(Clusters , Trial)

Clusters <- melt(Clusters, id="Trial")

colnames(Clusters)[2] <- "k"

plot1<-ggplot(data = Clusters, aes(x = Trial, y = value, colour = k)) +
  geom_line() + 
  labs(title = "K-means Clusters", x = "Iterations", y = "L", color = "# of Clusters") + theme_tufte() +
  theme(legend.position = "bottom")
ggsave("Images/Clusters1.png", plot = plot1)

  
plot1


### Plotting (b)

Class3 <- KM_Cluster(k = 3, N = 500, X = Data_sim, n_iter = 20)[2]
Class5 <- KM_Cluster(k = 5, N = 500, X = Data_sim, n_iter = 20)[2]

Data_sim3 <- data.frame(Data_sim)
Data_sim3 <- cbind(Data_sim3, Class3)
colnames(Data_sim3)<-c("x", "y", "cluster")
Data_sim5 <- data.frame(Data_sim)
Data_sim5 <- cbind(Data_sim5, Class5)
colnames(Data_sim5)<-c("x", "y", "cluster")
Combined <- rbind(Data_sim3, Data_sim5)

Total_K <- as.vector(rep(3, 500))
Total_K <- append(Total_K, c(rep(5, 500)))

Combined <- cbind(Combined, Total_K)

plot2<-ggplot(Combined, aes(x,y))+
  geom_point(aes(color=factor(cluster)))+ facet_grid(~Total_K) +
  labs(xlab="x")+
  labs(ylab="y")+
  labs(title = "Cluster Visualization", x = "X", y = "Y", color = "# of Clusters") +
  theme_tufte() +
  theme(legend.position = "bottom")

plot2
ggsave("Images/Clusters2.png", plot = plot2)



############################
## Question 2
############################

rm(list= ls())


Train <- read.csv("Data/ratings.csv", header = FALSE)
colnames(Train) <- c("user_id", "movie_id", "rating")
Train <-spread(Train,movie_id, rating)[,-1]

Test <- read.csv("Data/ratings_test.csv", header = FALSE)
colnames(Test) <- c("user_id", "movie_id", "rating")
Test <-spread(Test,movie_id, rating)[,-1]


Lam <- 1
Rank <- 10
Sig_2 <- 0.25
Trial <- 10
Iter <- 100
RMSE<-rep(NA, Trial)  
U_all <- list()
V_all <- list()
L_f <- as.data.frame((matrix(nrow = Iter , ncol = Trial)))
Users <- nrow(Train)
Items <- ncol(Train)
U <- matrix(NA, nrow = Users, ncol = Rank)
V <- matrix(NA, nrow = Rank, ncol = Items)

for(t in 1:Trial)
{
  print(t)
  Mu <- as.vector(rep(0, Rank))
  Sig <- Lam*diag(Rank)
  Mnorm <- function(vector)
  {
    vector = rmvnorm(1, mean = Mu, sigma = Sig)
  }
  U <- t(apply(U, 1, Mnorm))
  V <- (apply(V, 2, Mnorm))
  Train_mat <- as.matrix(Train)
  L_f_f <- c()
  for(num in 1:Iter)
  {
    paste("inner loop #", num)
    for (i in 1:nrow(U))
    {
      Users_Omega <- as.matrix(Train_mat[i,])
      Users_Omega <- t(Users_Omega)
      Rated_j <- which(!is.na(Users_Omega))
      V_temp <- V[,Rated_j]
      X_1 <- V_temp %*% t(V_temp)
      X_2 <- Lam * Sig_2 * diag(Rank)
      X_inv <- solve(X_1 + X_2)
      Rated_j_2 <- as.matrix(t(Users_Omega[Rated_j]))
      X_3 <- V_temp %*% t(Rated_j_2)
      X_final <- X_inv %*% X_3
      U[i,]<- t(X_final)
    }
    for (j in 1:ncol(V))
    {
      Items_Omega <- as.matrix(Train_mat[,j])
      Rated_i <- which(!is.na(Items_Omega))
      U_temp <- as.matrix(U[Rated_i,])
      if (length(Rated_i)>1)
      {
        U_temp<-t(U_temp)
      }
      Y_1 <- U_temp %*% t(U_temp)
      Y_2 <- Lam * Sig_2 * diag(Rank)
      Y_inv <- solve(Y_1 + Y_2)
      Rated_i_2 <- as.matrix(t(Items_Omega[Rated_i]))
      Y_3 <- (U_temp) %*% t(Rated_i_2)
      Y_final <- Y_inv %*% Y_3
      V[,j] <- Y_final
    }
    Pred_rate <- U%*%V
    Miss <- (sum(is.na(Train_mat)))
    All <- nrow(Train_mat) * ncol(Train_mat)
    Temp <- All - Miss
    E <- Pred_rate - Train_mat
    E <- E^2
    L_pt1 <- (sum(E, na.rm=TRUE))/(2*Sig_2)
    Miss_2 <- (sum(is.na(E)))
    U1 <- as.matrix(c(t(U)))
    L_pt2 <- (t(U1) %*% U1) * (Lam/2)
    V1 <- as.matrix(c((V)))
    L_pt3 <- (t(V1) %*% V1) * (Lam/2)
    L <- -(L_pt1 + L_pt2 + L_pt3)
    L_f_f[num]<-L
  }
  U_all[[t]] <- U
  V_all[[t]] <- V
  L_f[,t] <- L_f_f
  Test_mat <- as.matrix(Test)
  Pred_rate <- U%*%V
  Miss <- (sum(is.na(Test_mat)))
  Pred_2 <- Pred_rate[1:nrow(Test_mat),1:ncol(Test_mat)]
  E <- Pred_2 - Test_mat
  All <- nrow(Test_mat)*ncol(Test_mat)
  Num2 <- All - Miss
  E <- E^2
  Mean_sq_err <- (sum(E, na.rm=TRUE))/Num2
  Mean_sq_err_root <- sqrt(Mean_sq_err)
  RMSE[t] <- Mean_sq_err_root
}

N_iter <- c(1:100)
L_f2 <- cbind(L_f, N_iter)
colnames(L_f2) <- c("1","2", "3", "4", "5", "6","7","8","9","10", "iterations")
L_f2 <- L_f2[-1,]

Outcome <- melt(L_f2, id="iterations")

colnames(Outcome)[2] <- "code_run_number"

### Plotting 2(a)

plot1<-ggplot(data = Outcome, aes(x=iterations, y=value, colour=code_run_number)) +
  geom_line()+ ylim(-95000, -90000) + 
  labs(title = "Matrix Factorization and Log Liklihood", x = "Iterations", y = "Ln L", color = "Trial") + theme_tufte()

ggsave("Images/Matrix_Factor.png", plot = plot1)

### Writing table for 2(a)

Final_iters <- as.data.frame(t(L_f[100,]))
Run <- c(seq(1,10))
Final_iters  <- cbind(Run, Final_iters, RMSE)
colnames(Final_iters) <- c("Trial", "Objective Function Value", "RMSE")
Final_iters <- Final_iters[order(Final_iters[,2],decreasing=TRUE),]
write.csv(Final_iters, "Answer2.csv")



### Writing tables for 2(b)

Max_Obj_V <- as.matrix(unlist(V_all[[7]]))
Dist <- as.matrix(dist(t(Max_Obj_V), method="euclidean", p=2))

Movie_List_all <-data.frame(readLines("hw4-data/movies.txt", n=-1))                        
Movie_List <- c(50,485,182)
Dist <- Dist[Movie_List,]

#### Star Wars
SW <- data.frame(Dist[1,])
SW <- cbind(movie_id = rownames(SW), SW)
colnames(SW)[2]<-"distances"
SW <- SW[order(SW$distances),]
SW <- SW[1:11,]
SW$movie_id <- as.numeric(levels(SW$movie_id))[SW$movie_id]
SWids <- SW$movie_id
SWnames <- Movie_List_all[SWids,]
SWfinal <- cbind(as.character(SWnames), SW)
colnames(SWfinal)[1]<-"Movie"
SWfinal <- SWfinal[c(2,1,3)]
write.csv(SWfinal, "SWfinal.csv")

#### My Fair Lady
MFL <- data.frame(Dist[2,])
MFL <- cbind(movie_id = rownames(MFL), MFL)
colnames(MFL)[2]<-"distances"
MFL <- MFL[order(MFL$distances),]
MFL <- MFL[1:11,]
MFL$movie_id <- as.numeric(levels(MFL$movie_id))[MFL$movie_id]
MFLids <- MFL$movie_id
MFLnames <- Movie_List_all[MFLids,]
MFLfinal <- cbind(as.character(MFLnames), MFL)
colnames(MFLfinal)[1]<-"Movie"
MFLfinal <- MFLfinal[c(2,1,3)]
write.csv(MFLfinal, "MFLfinal.csv")

#### Goodfellas
GF <- data.frame(Dist[3,])
GF <- cbind(movie_id = rownames(GF), GF)
colnames(GF)[2]<-"distances"
GF <- GF[order(GF$distances),]
GF <- GF[1:11,]
GF$movie_id <- as.numeric(levels(GF$movie_id))[GF$movie_id]
GFids <- GF$movie_id
GFnames <- Movie_List_all[GFids,]
GFfinal <- cbind(as.character(GFnames), GF)
colnames(GFfinal)[1]<-"Movie"
GFfinal <- GFfinal[c(2,1,3)]
write.csv(GFfinal, "GFfinal.csv")
