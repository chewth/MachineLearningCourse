library(caret)
library(knitr)

set.seed(140819)

# Read in the training and testing data data
dat.train <- read.csv("./data/pml-training.csv", stringsAsFactors=FALSE)
dat.test <- read.csv("./data/pml-testing.csv", stringsAsFactors=FALSE)

# Function to filter the features
# Here, we just remove the features with any missing data
filterData <- function(idf) {
  # Since we have lots of variables, remove any with NA's
  # or have empty strings
  idx.keep <- !sapply(idf, function(x) any(is.na(x)))
  idf <- idf[, idx.keep]
  idx.keep <- !sapply(idf, function(x) any(x==""))
  idf <- idf[, idx.keep]
  
  # Remove the columns that aren't the predictor variables
  col.rm <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
              "cvtd_timestamp", "new_window", "num_window")
  idx.rm <- which(colnames(idf) %in% col.rm)
  idf <- idf[, -idx.rm]
  
  return(idf)
}

# Perform the filtering on the datasets
# The training data has classe so we need to
# convert it to a factor for the model building
dat.train <- filterData(dat.train)
dat.train$classe <- factor(dat.train$classe)

dat.test <- filterData(dat.test)

# Now create some prediction models on the training data
# Here, we'll use cross validation with trainControl to help optimize
# the model parameters
# Here, we'll do 5-fold cross validation
cvCtrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE, verboseIter = TRUE)
# We'll make 3 models that use different approaches and use a voting mechanism for the class predictions
m1 <- train(classe ~ ., data = dat.train, method = "rf", trControl = cvCtrl)

m2 <- train(classe ~ ., data = dat.train, method = "svmRadial", trControl = cvCtrl)

m3 <- train(classe ~ ., data = dat.train, method = "knn", trControl = cvCtrl)

# Make a data frame with the maximum accuracy values from the models obtained
# via the cross validation on the training data
acc.tab <- data.frame(Model=c("Random Forest", "SVM (radial)", "KNN"),
                     Accuracy=c(round(max(head(m1$results)$Accuracy), 3),
                     round(max(head(m2$results)$Accuracy), 3),
                     round(max(head(m3$results)$Accuracy), 3)))


table(acc.tab)

# Do the predictions
test.pred.1 <- predict(m1, dat.test)
test.pred.2 <- predict(m2, dat.test)
test.pred.3 <- predict(m3, dat.test)

# Make a table and check if they all agree
pred.df <- data.frame(rf.pred = test.pred.1, svm.pred = test.pred.2, knn.pred = test.pred.3)
pred.df$agree <- with(pred.df, rf.pred == svm.pred && rf.pred == knn.pred)
all.agree <- all(pred.df$agree)

colnames(pred.df) <- c("Random Forest", "SVM", "KNN", "All Agree?")
table(pred.df)

# Looks like they all do; let's write out the prediction files to submit
# This uses the code supplied by the class instructions
answers <- pred.df$rf.pred

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
