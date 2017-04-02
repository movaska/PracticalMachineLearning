# Practical machine learning: classifying proper/incorrect execution of dumbbell curls
M. Ovaska  
2 April 2017  



# Summary
The goal of this study is to use data from sensors to classify whether
individuals are performing dumbbell curls correctly (class A), or in one
of four incorrect ways (classes B-E), such as throwing the elbows to the front or
lifting the dumbbell only halfway. The sensors were attached to the individuals'
arms, forearms, waist, and to the dumbbell. 

52 predictors are identified and selected, and a Random Forest algorithm is trained
on a set of 13737 training examples. The optimal number of variables considered
at each tree split (mtry) is determined by considering the out-of-bag error
for a range of values on forests of 100 trees. The final model is then trained
using 1000 trees. The overall misclassification error on a test set of 5885
observations is approximately 0.5%. 

For more information on the data, see the original paper: 
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.
http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201


# Loading and preprocessing the data

First we load up the data and check for missing data (NA's, empty strings) etc. 
The goal is to find variables that are useful for training a machine learning 
algorithm.


```r
full_data <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
dim(full_data)
```

```
## [1] 19622   160
```

```r
# Get number of NA's in numeric columns, and "" in character-type columns:
missing_data <- sapply(1:length(full_data), 
                       function(x){
                           if(is.numeric(full_data[, x])){
                               return(sum(is.na(full_data[, x])))
                           }
                           else if(is.character(full_data[, x])){
                               return(sum(full_data[, x] == ""))
                           }
                       })

table(missing_data)
```

```
## missing_data
##     0 19216 
##    60   100
```

100 variables consist mainly of missing data (19216/19622), while the
rest of the variables have all data present. Imputing is not sensible in this
case, so we remove the columns with any NA's or empty strings. We also drop the
first 7 columns, which contrain timestamps, identifiers etc. 


```r
remove_cols <- c(1:7, which(missing_data > 0))
data <- full_data[, -remove_cols]
names(data)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

We end up with 52 numeric variables and the ```classe``` variable, which we want
to predict. We turn ```classe``` into a factor variable and separate the data
into a training set (70 %) and a test set (30 %).


```r
data$classe <- factor(data$classe)
library(caret)
inTrain <- createDataPartition(data$classe, p=0.7, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
```

# Training and testing a Random Forest algorithm

We use the Random Forest algorithm in the ```randomForest``` package through the 
```caret``` package's ```train``` function. The main tunable parameters are the
number of trees to
train (typically more is better, but it takes more time), and the number of
variables that are considered for each split in a tree, labeled ```mtry```. 
For classification problems with $N$ variables, a typical value is 
mtry = $\sqrt N$. We scan the range ```mtry=1,4,7,...,52``` for trees of
size ```ntree=100``` to find the optimal value. For a random forest, we can
use the out-of-bag error ("oob") instead of normal cross-validation to assess
the model performance during training.


```r
set.seed(100)
rfGrid <- expand.grid(mtry=seq(1,52,3))
trCont <- trainControl(method="oob")
rfModel <- train(classe~., data=training, method="rf", tuneGrid=rfGrid,
               ntree=100, trControl=trCont)
rfModel
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    1    0.9855136  0.9816714
##    4    0.9922836  0.9902389
##    7    0.9922108  0.9901464
##   10    0.9932300  0.9914358
##   13    0.9925020  0.9905143
##   16    0.9927932  0.9908834
##   19    0.9928660  0.9909755
##   22    0.9922108  0.9901462
##   25    0.9919196  0.9897780
##   28    0.9915557  0.9893172
##   31    0.9914101  0.9891340
##   34    0.9904637  0.9879361
##   37    0.9898813  0.9871995
##   40    0.9892262  0.9863704
##   43    0.9900997  0.9874755
##   46    0.9901725  0.9875666
##   49    0.9891534  0.9862775
##   52    0.9891534  0.9862774
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 10.
```


```r
library(ggplot2)
ggplot(data=rfModel$results, aes(x=mtry, y=Accuracy)) +
    geom_point(size=4, col="firebrick3") + 
    geom_path(lwd=1, col="grey25") + 
    ggtitle("Out-of-bag prediction accuracy for 100 trees") + 
    theme_bw(base_size=14)
```

![](dumbbell_classification_files/figure-html/plot oob accuracy-1.png)<!-- -->

The optimal value is found to be ```mtry=10```, although there is little difference
between this and the adjacent values. Finally we train one larger forest with
```ntree=1000``` and test this with the test set that we haven't touched yet.


```r
set.seed(100)
rfGrid <- expand.grid(mtry=c(10))
trCont <- trainControl(method="oob")
rfModelFinal <- train(classe~., data=training, method="rf", tuneGrid=rfGrid,
               ntree=1000, trControl=trCont)

confusionMatrix(testing$classe, predict(rfModelFinal, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    1
##          B   10 1128    1    0    0
##          C    0    8 1018    0    0
##          D    0    0    9  953    2
##          E    0    0    1    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9946          
##                  95% CI : (0.9923, 0.9963)
##     No Information Rate : 0.286           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9931          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9941   0.9930   0.9893   1.0000   0.9972
## Specificity            0.9998   0.9977   0.9984   0.9978   0.9998
## Pos Pred Value         0.9994   0.9903   0.9922   0.9886   0.9991
## Neg Pred Value         0.9976   0.9983   0.9977   1.0000   0.9994
## Prevalence             0.2860   0.1930   0.1749   0.1619   0.1842
## Detection Rate         0.2843   0.1917   0.1730   0.1619   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9969   0.9953   0.9938   0.9989   0.9985
```

The overall misclassification error on the test set is approximately 0.5 %.
Let's have a look at the most important variables, with respect to the decrease
in Gini index:


```r
importance <- as.data.frame(rfModelFinal$finalModel$importance)
importance$variable <- rownames(importance)
ggplot(data=importance, aes(x=reorder(variable, MeanDecreaseGini), y=MeanDecreaseGini)) +
    geom_col(fill="firebrick3") +
    coord_flip() +
    xlab("") +
    ylab("Mean decrease in Gini index") +
    ggtitle("Relative importance of variables") +
    theme_bw(base_size = 14)
```

![](dumbbell_classification_files/figure-html/variable importance-1.png)<!-- -->

Finally we can also predict the class of the 20 additional testing cases:


```r
final_testing <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)
final_testing <- final_testing[, -remove_cols]
predict(rfModelFinal, final_testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


