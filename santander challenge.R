# ou are provided with an anonymized dataset containing numeric feature variables, the numeric target column, and a string ID column.

# The task is to predict the value of target column in the test set.

# Importing the dataset
library(data.table)
library(tibble)
library(dplyr)
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16G")

training_set<- fread("training_set.csv")

test_set <- fread("test_set.csv")

#Preprocessing 

training_set <- training_set %>% select(-"'ID") # remove the id column
test_set <- test_set %>% select(-"'ID") # select the variables of interest
Nb# testset does not contain the outcome variable
last(colnames(training_set))# check position of dependent variable
training_set <- training_set[,c(2:4992,1)] # reordering columns
last(colnames(training_set))#
#Remove duplicate columns
b<- training_set %>% 
  setNames(make.names(names(.), unique = TRUE)) %>% 
  select(-matches("*\\.[1-9]+$"))
dim(b)
tt<- log(training_set)
df <- subset(df, select = -c(a, c))

#Feature Scaling

test_set<- h2o.scale(as.h2o(test_set), center = F, scale = TRUE)
#build model
model = h2o.deeplearning(y = 'target',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(6,6),
                         epochs = 100,
                         standardize=T,
                         train_samples_per_iteration = -2)
# View the AutoML Leaderboard
model@model
# Predicting the Test set results (predict if a customer will leave the bank)
y_pred = h2o.predict(model, newdata = as.h2o(test_set))
test_set$target<- y_pred
dim(y_pred)

write.csv(y_pred, 'y_pred.csv')
# Once model is trained, we can calculate its performance on a new (unseen) dataset by using h2o.performance.
performance <- h2o.performance(model, newdata = test_set)
performance # model 1 without CV: RSME = 0.005002065


#model2 with GBM
#build GBM default model
class<- h2o.gbm(y = 'target', training_frame = as.h2o(training_set3))
# Predicting the Test set results (predict if a customer will leave the bank)
y_pred2 = h2o.predict(class, newdata = as.h2o(test_set))
test_set$target<- NULL
test_set$target<- y_pred2

# Once model is trained, we can calculate its performance on a new (unseen) dataset by using h2o.performance.
performance <- h2o.performance(class, newdata = test_set)
performance # model 2 GBM RSME : 0.1171116

#model3 with autoML
#build GBM default model
class_automl<- h2o.automl(y = 'target', 
                          training_frame = as.h2o(training_set))

# Predicting the Test set results (predict if a customer will leave the bank)
y_pred3 = h2o.predict(class_automl, newdata = as.h2o(test_set))
test_set$target<- NULL
test_set$target<- y_pred3

# Once model is trained, we can calculate its performance on a new (unseen) dataset by using h2o.performance.
performance <- h2o.performance(class, newdata = test_set)
performance # model 3 default autoML  RSME : 101767.4

class_automl2<- h2o.automl(y = 'target', 
                          training_frame = as.h2o(training_set), 
                          nfolds = 10)

# Predicting the Test set results (predict if a customer will leave the bank)
y_pred4 = h2o.predict(class_automl2, newdata = as.h2o(test_set))
test_set$target<- NULL
test_set$target<- y_pred4

# Once model is trained, we can calculate its performance on a new (unseen) dataset by using h2o.performance.
performance <- h2o.performance(class, newdata = test_set)
performance # model 3 default autoML2  RSME : 101767.4

# model 4: xgboost 

my_xgb1 <- h2o.xgboost(y="target",training_frame = as.h2o(training_set),
                       ntrees = 50,
                       max_depth = 3,
                       min_rows = 2,
                       learn_rate = 0.2,
                       nfolds = 5,
                       fold_assignment = "Modulo",
                       keep_cross_validation_predictions = TRUE,
                       seed = 1,
                       backend = "gpu")

# Predicting the Test set results (predict if a customer will leave the bank)
y_pred5 = h2o.predict(my_xgb1, newdata = as.h2o(test_set))
test_set$target<- NULL
test_set$target<- y_pred5

# Once model is trained, we can calculate its performance on a new (unseen) dataset by using h2o.performance.
performance <- h2o.performance(my_xgb1, newdata = test_set)
performance # model 3 default autoML2  RSME : 101767.4



# Calculate performance measures at threshold that maximizes precision
library(h2o)
h2o.init()
prosPath <- system.file("extdata", "prostate.csv", package="h2o")
prostate.hex <- h2o.uploadFile(path = prosPath)
prostate.hex$CAPSULE <- as.factor(prostate.hex$CAPSULE)
prostate.gbm <- h2o.gbm(3:9, "CAPSULE", prostate.hex)
h2o.performance(model = prostate.gbm, newdata=prostate.hex)

## If model uses balance_classes
## the results from train = TRUE will not match the results from newdata = prostate.hex
prostate.gbm.balanced <- h2o.gbm(3:9, "CAPSULE", prostate.hex, balance_classes = TRUE)
h2o.performance(model = prostate.gbm.balanced, newdata = prostate.hex)
h2o.performance(model = prostate.gbm.balanced, train = TRUE)

 



variable_importances # click on he model, and its elements to see that


h2o.shutdown()


# Automatic Machine Learning ---------------------------------------------------------------------

aml <- h2o.automl(y = 'target',
                  training_frame = as.h2o(training_set),
                  max_runtime_secs = 60)

# View the AutoML Leaderboard
aml@leader
lb <- aml@leaderboard
as.tibble(lb)

# Predicting the Test set results (predict target) using the leaderboard

pred <- h2o.predict(aml, as.h2o(test_set))  # predict(aml, test) and h2o.predict(aml@leader, test) also work
head(pred>0.5)
head(test_set[11])

perf <- h2o.performance(aml@leader, test)


#plotting a best model
library(h2o)
localH2O = h2o.init()

# Run GBM classification on prostate.csv
prosPath = system.file("extdata", "prostate.csv", package = "h2o")
prostate.hex = h2o.importFile(localH2O, path = prosPath, key = "prostate.hex")
prostate.gbm = h2o.gbm(y = 2, x = 3:9, data = prostate.hex)


# Calculate performance measures at threshold that maximizes precision
prostate.pred = h2o.predict(prostate.gbm)
prostate.perf = h2o.performance(prostate.pred[,3], prostate.hex$CAPSULE, measure = "precision")

plot(prostate.perf, type = "cutoffs")     # Plot precision vs. thresholds
plot(prostate.perf, type = "roc")         # Plot ROC curve

library(plotly)
