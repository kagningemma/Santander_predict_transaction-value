# ou are provided with an anonymized dataset containing numeric feature variables, the numeric target column, and a string ID column.

# The task is to predict the value of target column in the test set.

# Importing the dataset
library(data.table)
library(tibble)
library(dplyr)
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "8G")
http://localhost:54321
training_set<-read.csv("training_set.csv")
test_set <-read.csv("test_set.csv")

#Preprocessing 
 
as.tibble(training_set[1])
training_set2<- training_set[-1] #remove the id column
as.tibble(training_set2[1])
as.tibble(test_set[1]) # remove the id column
test_set2<- test_set[-1]

#check olumns with missig val
# Check missing values . Identify the columns which have missing values 
mvc = 0
for (i in 1:ncol(b))
{
  m = sum(is.na(b[,i]))
  print(paste("Column ",colnames(b[i])," has ",m," missing values"))
  if(m>0){
    mvc = mvc+1
  }
  else{
    mvc
  }
}  
print(paste("Dataset has overall ",mvc," columns with missing values"))

#Remove duplicate columns
b<- training_set2 %>% 
  setNames(make.names(names(.), unique = TRUE)) %>% 
  select(-matches("*\\.[1-9]+$"))
dim(b) # b has 4981 columns

b<- b[, -(which(colSums(b)==0))]
dim(b) #

b<- Filter(function(x) sd(x) != 0, b) # filter all columns with 0 variance

dim(b)# has 4725 colunms now

anyDuplicated(b) # no duplicate


# all columns are near zero columns.

# remove near zero variance predictors
library(caret)
nzv <- nearZeroVar(b, saveMetrics= TRUE)
nzv2 <- nearZeroVar(b)
nzv[nzv$nzv,][1:10,]
dim(b)
nzv <- nearZeroVar(as.data.frame(b))
filteredDescr <- b[, -nzv]
dim(filteredDescr) # it is not working for my dataset

# Removing highly correlated predictors
descrCor <-  cor(b)
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999)

training_set3<- as.h2o(b)
test_set3<- as.h2o(test_set2)

# GBM
## We only provide the required parameters, everything else is default
gbm <- h2o.gbm(y = "target", 
               training_frame = training_set3,
               max_depth = 4)
## Show a detailed model summary
gbm

predict.gbm <- as.data.frame(h2o.predict(gbm, test_set3))
sub_gbm <- data.frame(ID = test_set$"'ID", target=predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm2.csv", row.names = F)



#Tuning GBM
## Depth 10 is usually plenty of depth for most datasets, but you never know
hyper_params = list( max_depth = c(4,6,8,12,16,20) )
grid <- h2o.grid(
  hyper_params = hyper_params,
    ## full Cartesian hyper-parameter search
  search_criteria = list(strategy = "Cartesian"),
    ## which algorithm to run
  algorithm="gbm",
  y = "target",
  training_frame = training_set3, 
  validation_frame = test_set3,
  stopping_metric = "RMSLE")

  
  grid <- h2o.grid("gbm", y = "target", 
                   training_frame = training_set3,
                   stopping_metric = "RMSLE",
                   grid_id = "depth_grid",
                 hyper_params = list( max_depth = c(4,6,8,12,16,20)))
  
  sortedGrid <- h2o.getGrid("depth_grid", sort_by="RMSLE", decreasing = TRUE)    
  sortedGrid # problem i should not increase max depth
  
  
## by default, display the grid search results sorted by increasing logloss (since this is a classification task)

grid                                                                       

## sort the grid models by decreasing AUC
sortedGrid <- h2o.getGrid("depth_grid", sort_by="auc", decreasing = TRUE)    
sortedGrid

gbm

predict.gbm <- as.data.frame(h2o.predict(gbm, test_set3))
sub_gbm <- data.frame(ID = test_set$"'ID", target=predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm2.csv", row.names = F)


#tuning with caret 
library(doParallel); cl <- makeCluster(detectCores()); registerDoParallel(cl)

library(caret)
library(randomForest)

names(getModelInfo()) # to see all model you can build with caret

tune_rf <- train(x=b[-1], y=b$target, ntree=5, method = "rf")
tune_rf
tune_rf$bestTune
set.seed(1234)
regressor = randomForest(x=b[-1], y=b$target, ntree=100, mtry = 2)
regressor
 
# Predicting a new result with Random Forest Regression
y_pred.rf = predict(regressor, newdata=test_set2, type = "response")
sub_rf <- data.frame(ID = test_set$"'ID", target=y_pred.rf)
write.csv(sub_gbm, file = "sub_gbm2.csv", row.names = F)



tune_svm = train(form = target ~ ., data = b, method = 'svmRadial')

classifier # to see the un-tuned classifier 
classifier$bestTune #

# stop the parallel processing and register sequential front-end
stopCluster(cl); registerDoSEQ();






#GLM regression
regression.model <- h2o.glm( y = "target",training_frame = training_set3, 
                             standardize = T,
                             family = "gaussian")

h2o.performance(regression.model) # RMSLE:  2.071767
summary(regression.model)


#So, after we print the model results, we see that regression gives a poor R² value.
#It means that only 32.6% of the variance in the dependent variable is explained by independent variable and rest is unexplained. This shows that regression model is unable to capture non linear relationships.

# Out of curiosity, let's check the predictions of this model. 
#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test_set3))
sub_reg <- data.frame(ID = test_set$"'ID", target=predict.reg$predict)
write.csv(sub_reg, file = "sub_reg.csv", row.names = F)
write.table(sub_reg, file="sub_reg2.csv", sep = ",", row.names = T, append = T)
fwrite(sub_reg, file = "sub_reg.csv", row.names = T)
write.xlsx(sub_reg, file="sub_reg2.csv")

#RMSLE:  2.071766
#Mean Residual Deviance :  6.778193e+13
#R^2 :  9.956263e-05
#It seems, we can do well if we choose an algorithm which maps non-linear relationships well. Random Forest is our next bet. Let's do it.
#try random forest

#Random Forest
training_set4<- training_set3[-1]
training_set4<- h2o.scale((training_set4), center = T, scale = T)
training_set4<- h2o.cbind(training_set4,training_set3$target)
test_set4<- h2o.scale((test_set3), center = T, scale = T)
system.time(rforest.model <- h2o.randomForest(y = "target", training_frame = training_set4,
                                              ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122))

h2o.performance(rforest.model) # check performance of rf model
h2o.r2(rforest.model, train = T, valid = FALSE, xval = FALSE) #Rsqare=0.07669645
# with scaling predictors and centering predictors 
#MSE:  0.9230978
#RMSE:  0.9607798
#MAE:  0.6987088
#RMSLE:  0.8419327
#Mean Residual Deviance :  0.9230978

#check variable importance
h2o.varimp(rforest.model)
#making predictions on unseen data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test_set4)))
#writing submission file
sub_rf <- data.frame(ID = test_set$"'ID", target=predict.rforest$predict)
write.csv(sub_rf, file = "sub_rf.csv", row.names = F)

#This gave a slight improvement on leaderboard, 
#but not as significant as expected. May be GBM, a boosting algorithm can help us.

#GBM

system.time(
  gbm.model <- h2o.gbm(y = "target", training_frame = training_set3,ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)
)
h2o.performance (gbm.model)
h2o.r2(gbm.model, train = T, valid = FALSE, xval = FALSE) # Rsqare GBM = 0.47
#MSE:  0.5221815
#RMSE:  0.7226213
#MAE:  0.502061
#RMSLE:  0.6690803
#Mean Residual Deviance :  0.5221815
#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test_set4))
sub_gbm <- data.frame(ID = test_set$"'ID", target=predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm.csv", row.names = F)


system.time(predict.dl<- h2o.deeplearning(y="target", training_frame = training_set3))
predict.dl
h2o.performance(gbm2)
h2o.r2(predict.dl, train = T, valid = F,xval = F)

