########### library ###########
library(caret)
library(xgboost)
library(pROC)
library(e1071)
library(MLmetrics)
library(MASS)
library(glmnet)
library(bimba)
library(klaR)

########## data import #########
train = read.csv('../project_code_9_reclean_gram_tfidf/train_tfidf_norm_v2.csv',header = T)
test = read.csv('../project_code_9_reclean_gram_tfidf/test_tfidf_norm_v2.csv',header = T)

train.toxic = train[,c(1:465)]
test.toxic = test[,c(1:465)]

########## assign weights ###########
fraction_0 <- 1 - sum(train.toxic$toxic == 0) / nrow(train.toxic)
fraction_1 <- 1 - sum(train.toxic$toxic == 1) / nrow(train.toxic)
# assign that value to a "weights" vector
weights <- numeric(nrow(train.toxic))
weights[train.toxic$toxic == 0] <- fraction_0
weights[train.toxic$toxic == 1] <- fraction_1

################# lasso on this new dataset
train.t.matrix =model.matrix(toxic~.,train.toxic)[,-465]
test.t.matrix = model.matrix(toxic~.,test.toxic)[,-465]

set.seed(1234)
lasso=cv.glmnet(train.t.matrix,train.toxic$toxic,
                alpha=1,family="binomial",
                weights = weights,
                nfolds = 3)

lasso.pred = predict(lasso, s = lasso$lambda.min, newx = test.t.matrix)
lasso.pred = ifelse(lasso.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.pred), as.factor(test.toxic$toxic))
AUC(as.factor(lasso.pred), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso.pred))
