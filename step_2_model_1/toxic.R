################### library #####################
library(bimba)
library(caret)
library(glmnet)
library(randomForestSRC)
library(fastAdaboost)
library(adabag)
library(plyr)
library(xgboost)
library(MLmetrics)
# install.packages('C50')
library(C50)
library(ggplot2)
# install.packages('ranger')
library(ranger)


################### dataset import ##############
# import the new dataset

train_new = read.csv('../project_code_2/train_new.csv',header = T)
test_new = read.csv('../project_code_2/test_new.csv',header = T)
train_label = read.csv('../data/train.csv', header = T)
test_label = read.csv('../data/test_labels.csv', header = T)
test_label = test_label[test_label$toxic != -1,]

# the original test contains 484 columns, train contains 505

# modify train and test new dataset

## intersection column (in case)
intersection_col = intersect(colnames(train_new),
                             colnames(test_new))
length(intersection_col)
# intersection is 466 co

## 
test_new = test_new[, intersection_col]
ncol(test_new) # now 466
train_new = train_new[, intersection_col]
ncol(test_new) # now 466

## remove id column (466) in both
match('id',names(test_new))
match('id',names(train_new))
train_new = train_new[,-466]
test_new = test_new[,-466]
train_new = train_new[,-1]
test_new = test_new[,-1]

toxic.prop = table(train_label$toxic)[2]/table(train_label$toxic)[1]

################### scale - toxic ############
# scl = function(x){(x-min(x))/(max(x)-min(x))}
# train_scl = as.data.frame(scale(train_new))

train_log = log(train_new+1)
test_log = log(test_new+1)

train.toxic = train_log
train.toxic$toxic = train_label$toxic
train.toxic$toxic = as.factor(train.toxic$toxic)
levels(train.toxic$toxic) = make.names(unique(train.toxic$toxic))

########################### train - raw xgb #######################

fitControl = trainControl(method = 'none',
                             summaryFunction = twoClassSummary,
                             classProbs = T)
fitControl.cv = trainControl(method = 'cv',
                             summaryFunction = twoClassSummary,
                             #seeds = 1000,
                             classProbs = T,number =3)

set.seed(1000)

xgbGrid <- expand.grid(nrounds = 200,
                       max_depth = 10,
                       eta = .1,#learning rage
                       gamma = 0.001, #regularization parameter.
                       colsample_bytree = .8,
                       min_child_weight = 1,
                       subsample = 0.75)

set.seed(1000)
xgb.t.base = train(toxic~., 
                  data = train.toxic, 
                  method = 'xgbTree',
                  metric = "ROC",
                  verbose = FALSE,
                  trControl = fitControl,
                  # tuneGrid =xgbGrid,
                  scale_pos_weight = toxic.prop)

xgb.pred = predict(xgb.t.base,test_log)
xgb.pred = ifelse(xgb.pred == 'X0','0','1')
confusionMatrix(as.factor(xgb.pred), as.factor(test_label$toxic))
summary(test_label$toxic)
F1_Score(as.factor(test_label$toxic),as.factor(xgb.pred))
AUC(as.factor(test_label$toxic),as.factor(xgb.pred))



###################### Tuning for xgboost ################
set.seed(1000)

xgbGrid.tune <- expand.grid(nrounds = 200,
                       max_depth = c(4,6,10),
                       eta = c(0.05,0.1,0.2),#learning rage
                       gamma = 0.001, #regularization parameter.
                       colsample_bytree = c(1),
                       min_child_weight = 1,
                       subsample = c(0.75,1))

set.seed(1000)
xgb.t.tune = train(toxic~., 
                   data = train.toxic, 
                   method = 'xgbTree',
                   metric = "ROC",
                   verbose = FALSE,
                   trControl = fitControl.cv,
                   tuneGrid =xgbGrid.tune,
                   scale_pos_weight = toxic.prop)

plot(xgb.t.tune)
ggplot(xgb.t.tune)
xgb.t.tune


xgb.pred.t = predict(xgb.t.tune,test_scl)
xgb.pred.t = ifelse(xgb.pred.t == 'X0','0','1')
confusionMatrix(as.factor(xgb.pred.t), as.factor(test_label$toxic))
summary(test_label$toxic)
F1_Score(as.factor(test_label$toxic),as.factor(xgb.pred.t))
AUC(as.factor(test_label$toxic),as.factor(xgb.pred.t))
xgb.imp  = varImp(xgb.t.tune, scale = F)
plot(xgb.imp, top = 10)


set.seed(1000)
xgbGrid.tune2 <- expand.grid(nrounds = 200,
                            max_depth = c(6),
                            eta = c(0.2),#learning rage
                            gamma = 0.001, #regularization parameter.
                            colsample_bytree = c(1/2,2/3,1),
                            min_child_weight = 1,
                            subsample = c(0.5,0.75))


xgb.t.tune2 = train(toxic~., 
                   data = train.toxic, 
                   method = 'xgbTree',
                   metric = "ROC",
                   verbose = FALSE,
                   trControl = fitControl.cv,
                   tuneGrid =xgbGrid.tune2,
                   scale_pos_weight = toxic.prop)

####################### Lasso ############################
train.matrix =model.matrix(toxic~.,train.toxic)[,-465]
# test_scl$toxic = test_label$toxic
# x.test=model.matrix(toxic~.,test_scl)[,-465] 

lasso=cv.glmnet(train.matrix,train.toxic$toxic,
                    alpha=1,family="binomial")
plot(lasso, xvar = 'lambda')
lasso.mod$lambda.min
lambda_final = lasso.mod$lambda.1se
coef(lasso.mod, lasso.mod$lambda.1se)

lasso=glmnet(x.train,train_down$toxic, 
             lambda = lasso.mod$lambda.1se,
             alpha=1,family="binomial")
lasso_prob = predict(lasso, newx = x.test)
lasso_pred = ifelse(lasso_prob > 0.5, 1,0)
confusionMatrix(as.factor(lasso_pred), as.factor(test_label$toxic))

######################## export #########
write.csv(train.toxic,'train_log_toxic.csv',row.names = F)
write.csv(test_log,'test_log.csv',row.names = F)


######################### RANDOM FOREST ############
train.toxic$toxic = train_label$toxic
train.toxic$toxic = as.factor(train.toxic$toxic)
levels(train.toxic$toxic) = make.names(unique(train.toxic$toxic))

set.seed(1000)
rf.t.base = ranger(toxic~.,data = train.toxic,
                   importance = 'impurity',
                   sample.fraction = c(toxic.prop,1),
                   class.weights = toxic.prop,
                   verbose = T)

rf.pred = predict(rf.t.base,test_log,type = 'prob')
rf.log = ifelse(rf2.p[,2] > 0.5, 1, 0) %>% factor
confusionMatrix(as.factor(rf.log), as.factor(test_label$toxic))