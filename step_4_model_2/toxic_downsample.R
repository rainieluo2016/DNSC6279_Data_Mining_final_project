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
library(dplyr)
library(ggplot2)

########## data import #########
train = read.csv('../project_code_6_reclean/train_tfidf_norm.csv',header = T)
test = read.csv('../project_code_6_reclean/test_tfidf_norm.csv',header = T)

########## about dataset ######

col.max = sapply(train[,c(1:461)], max)
# col.max = sort(col.max, decreasing = T)
col.max %>% sort(decreasing = T)

col.min  = sapply(train[,c(1:461)], mean)
col.std = sapply(train[,c(1:461)],sd)

ggplot(train, aes(x = stupid)) + geom_density(kernel = 'gaussian')
ggplot(train, aes(x = log(stupid+1))) + geom_density(kernel = 'gaussian')
ggplot(train, aes(x = stupid^(1/3))) + geom_density(kernel = 'gaussian')
plot(density(train$stupid))

######## standardize ###########
normalized = function(x){
  return(x/max(x))
}
train.standized  = train[,c(1:461)]
train.standized = apply(train.standized[,c(1:461)],2,normalized)
train.standized = as.data.frame(train.standized)
sapply(train.standized, max)
ggplot(train.standized, aes(x = stupid)) + geom_density(kernel = 'gaussian')

################ TOXIC #######################
train.toxic = train[,c(1:461,463)]
test.toxic = test[,c(1:461,463)]



train.toxic.factor  =train.toxic
train.toxic.factor$toxic = as.factor(train.toxic.factor$toxic)
levels(train.toxic.factor$toxic) = make.names(unique(train.toxic.factor$toxic))


######### undersample #########################
set.seed(1000)
train.under = RUS(train.toxic.factor,perc_under = 50)

######### lasso - no weight ####################

# create an initial model
train.t.u.matrix =model.matrix(toxic~.,train.under)[,-462]
test.t.u.matrix = model.matrix(toxic~.,test.toxic)[,-462]

set.seed(1234)
lasso.t.u=cv.glmnet(train.t.u.matrix,train.under$toxic,
                     alpha=1,family="binomial",
                     # weights = weights,
                     nfolds = 3)
# plot(lasso,xvar = 'lambda')
# lambda = lasso$lambda.1se
# lasso.coef = coef(lasso, lasso$lambda.1se)@i
# lasso.coef = lasso.coef[-1]-1
# lasso.coef[435] = 462

lasso.pred.t.u = predict(lasso.t.u, s = lasso.t.u$lambda.min, 
                          newx = test.t.u.matrix)
lasso.pred.t.u = ifelse(lasso.pred.t.u > 0.5, 1,0)
confusionMatrix(as.factor(lasso.pred.t.u), as.factor(test.toxic$toxic))
AUC(as.factor(lasso.pred.t.u), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso.pred.t.u))


######################## hate #################

train.hate = train[,c(1:461,468)]
test.hate = test[,c(1:461,468)]

train.hate.factor = train.hate
train.hate.factor$identity_hate = as.factor(train.hate.factor$identity_hate)
levels(train.hate.factor$identity_hate) = make.names(unique(train.hate.factor$identity_hate))

# assign weights 
fraction_0.h <- 1 - sum(train$identity_hate == 0) / nrow(train)
fraction_1.h <- 1 - sum(train$identity_hate == 1) / nrow(train)
# assign that value to a "weights" vector
weights.h <- numeric(nrow(train))
weights.h[train$identity_hate == 0] <- fraction_0.h
weights.h[train$identity_hate == 1] <- fraction_1.h

# create an initial model
train.h.matrix =model.matrix(identity_hate~.,train.hate.factor)[,-462]
test.h.matrix = model.matrix(identity_hate~.,test.hate)[,-462]

set.seed(1234)
lasso.h.w = cv.glmnet(train.h.matrix,
                      train.hate.factor$identity_hate,
                      alpha=1,family="binomial",
                      weights = weights.h,
                      nfolds = 3)


lasso.h.pred = predict(lasso.h.w, 
                       s = lasso.h.w$lambda.min, 
                       newx = test.h.matrix)
lasso.h.pred = ifelse(lasso.h.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.h.pred), as.factor(test.hate$identity_hate))
AUC(as.factor(lasso.h.pred), as.factor(test.hate$identity_hate))
F1_Score(as.factor(test.hate$identity_hate), as.factor(lasso.h.pred))

##################### standardzied selection #############
train.standized$identity_hate = train$identity_hate

# create an initial model
train.h.s.matrix =model.matrix(identity_hate~.,train.standized)[,-462]
# test.h.matrix = model.matrix(identity_hate~.,test.hate)[,-462]

set.seed(1234)
lasso.h.w = cv.glmnet(train.h.s.matrix,
                      train.standized$identity_hate,
                      alpha=1,family="binomial",
                      weights = weights.h,
                      nfolds = 3)


lasso.h.pred = predict(lasso.h.w, 
                       s = lasso.h.w$lambda.min, 
                       newx = test.h.matrix)
lasso.h.pred = ifelse(lasso.h.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.h.pred), as.factor(test.hate$identity_hate))
AUC(as.factor(lasso.h.pred), as.factor(test.hate$identity_hate))
F1_Score(as.factor(test.hate$identity_hate), as.factor(lasso.h.pred))


########### logistic 
logistic.h.w = glm(identity_hate ~., 
                   data = train.hate.factor[,hate.l],
                   weights = weights.h,
                   family = binomial) 
logistic.h.pred = predict(logistic.h.w, 
                          test.hate[,hate.l],
                          type = 'response')
logistic.h.pred = ifelse(logistic.h.pred > 0.5, 1,0)

confusionMatrix(as.factor(logistic.h.pred), as.factor(test.hate$identity_hate))
AUC(as.factor(logistic.h.pred), as.factor(test.hate$identity_hate))
F1_Score(as.factor(test.hate$identity_hate), as.factor(logistic.h.pred))