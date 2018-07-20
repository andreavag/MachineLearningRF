###Machine Learning Work

First, we will load all the required packages for work and read the training csv file. For convenience we will convert it to tbl_df format.

```r
library(dplyr)
library(ggplot2)
library(caret)
setwd("C:\\Users\\barca\\Desktop\\MachineLearningExam")
train_full <- read.csv("pml-training.csv", header = TRUE, sep = ",", na.strings = c("NA", ""), stringsAsFactors = TRUE)
train_full <- tbl_df(train_full)
```

After studying the data it becomes clear, that many variables have *more than 50%* missing values rate (mostly even more than 90%), so it is more reasonable to exclude this variables, then trying to substitute to some logical values, is missing values are actually more than normal.

```r
cols <- ncol(train_full)
rows <- nrow(train_full)
knockout <- character(0)
for(i in 1:cols) {
       if(sum(is.na(train_full[,i])) < rows*0.50)
          {knockout <- c(knockout, names(train_full[i]))} }
```

We will also exclude *time series variables*, because we are not going to use time-series forecasting, as well as we will include "X" variable, because it is just ordering from 1 to N.

```r
train_2 <- train_full %>% select(knockout)
train_3 <- train_2 %>% select(-c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
```


For validation we will use the following two techinques.

- We will split the training set into 75% training_final and 25% testing_final set, and leave the real testing set with 20 variables as validation test.
- We will use cross-validation on train_final with 3 folds, as a first step to estimate out-of-sample error, before trying the model on testing. We will only proceed to train_final, if it shows accuracy rate > 95%. 


```r
set.seed(7)
splits <- createDataPartition(train_3$classe, p = 0.75, list = F)
train_final <- train_3[splits,]
test_final <-  train_3[-splits,]
train_options <- trainControl(method = "cv", number = 3, verboseIter = FALSE) ## Cross-validation
```


As this model is going to work as a black-box prediction (we have more than 50 variables predictor), we will first try to use random forest algorithm.

```r
model1 <- train(classe ~ ., data = train_final, method = "rf", trControl = train_options)
model1$results
```

```
##   mtry  Accuracy     Kappa   AccuracySD      KappaSD
## 1    2 0.9905556 0.9880516 0.0018284689 0.0023139791
## 2   30 0.9957195 0.9945851 0.0005400742 0.0006834413
## 3   59 0.9917107 0.9895134 0.0030938717 0.0039142271
```

```r
confusionMatrix(predict(model1), train_final$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

As the cross-validation results show, predicted Out-of-Sample Accuracy is more than 99%. So we can proceed and predict with this model on test_final dataset.



```r
confusionMatrix(predict(model1, newdata = test_final), test_final$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  946    1    0    0
##          C    0    2  854    0    0
##          D    0    0    0  804    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9992          
##                  95% CI : (0.9979, 0.9998)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.999           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9988   1.0000   1.0000
## Specificity            0.9997   0.9997   0.9995   1.0000   1.0000
## Pos Pred Value         0.9993   0.9989   0.9977   1.0000   1.0000
## Neg Pred Value         1.0000   0.9992   0.9998   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1929   0.1741   0.1639   0.1837
## Detection Prevalence   0.2847   0.1931   0.1746   0.1639   0.1837
## Balanced Accuracy      0.9999   0.9983   0.9992   1.0000   1.0000
```
The resulting accuracy is great, thus we don't need algorithms and can proceed to validation set.



```r
validation20 <- train_full <- read.csv("pml-testing.csv", header = TRUE, sep = ",", na.strings = c("NA", ""), stringsAsFactors = TRUE)
predict(model1, newdata = validation20)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
All the 20 predicted values were correct according to the test quiz.
