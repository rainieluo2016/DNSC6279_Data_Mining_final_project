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
# library(mlr)
library(MASS)
library(gbm)


################## import dataset ############
train.l.t = read.csv('../../project_code_4under/train_log_toxic.csv',header = T)
test.l = read.csv('../../project_code_4under/test_log.csv',header = T)
test.label = read.csv('../../data/test_labels.csv',header = T)
test.label = test.label[test.label$toxic != -1,]
test.l$toxic = test.label$toxic

########### weighted lasso example #######
fraction_0 <- 1 - sum(train.l.t$toxic == 'X0') / nrow(train.l.t)
fraction_1 <- 1 - sum(train.l.t$toxic == 'X1') / nrow(train.l.t)
# assign that value to a "weights" vector
weights <- numeric(nrow(train.l.t))
weights[train.l.t$toxic == 'X0'] <- fraction_0
weights[train.l.t$toxic == 'X1'] <- fraction_1
sum(weights)

# create an initial model
train.matrix =model.matrix(toxic~.,train.l.t)[,-465]
test.matrix = model.matrix(toxic~.,test.l)[,-465]

set.seed(1234)
lasso=cv.glmnet(train.matrix,train.l.t$toxic,
                alpha=1,family="binomial",
                weights = weights,
                nfolds = 3)
plot(lasso,xvar = 'lambda')
lambda = lasso$lambda.1se
lasso.coef = coef(lasso, lasso$lambda.1se)@i
lasso.coef = lasso.coef[-1]
lasso.coef = lasso.coef -1
lasso.coef[335] = 465

lasso.pred = predict(lasso, s = lasso$lambda.1se, newx = test.matrix)
lasso.pred = ifelse(lasso.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.pred), as.factor(test.label$toxic))

AUC(as.factor(lasso.pred), as.factor(test.label$toxic))
F1_Score(as.factor(test.label$toxic),as.factor(lasso.pred))

################ lasso no weights - worse ###################
set.seed(1234)
lasso.no.w=cv.glmnet(train.matrix,train.l.t$toxic,
                alpha=1,family="binomial",
                #weights = weights,
                nfolds = 3)
lasso.no.w.pred = predict(lasso.no.w, s = lasso.no.w$lambda.1se, newx = test.matrix)
lasso.no.w.pred = ifelse(lasso.no.w.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.no.w.pred), as.factor(test.label$toxic))
remove(lasso.no.w)
remove(lasso.no.w.pred)

################ weight 2 is a little worse than original weight ########


fraction2_0 <- 1 - sum(train.l.t$toxic == 'X0') / nrow(train.l.t)
fraction2_1 <- 1 - sum(train.l.t$toxic == 'X1')/2 / nrow(train.l.t)
remove(fraction2_0)
remove(fraction2_1)
# assign that value to a "weights" vector
weights2 <- numeric(nrow(train.l.t))
weights2[train.l.t$toxic == 'X0'] <- fraction2_0
weights2[train.l.t$toxic == 'X1'] <- fraction2_1
remove(weights2)
set.seed(1234)
lasso2=cv.glmnet(train.matrix,train.l.t$toxic,
                alpha=1,family="binomial",
                weights = weights2,
                nfolds = 3)
remove(lasso2)
lasso2.pred = predict(lasso2, s = lasso2$lambda.1se, newx = test.matrix)
lasso2.pred = ifelse(lasso2.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso2.pred), as.factor(test.label$toxic))
remove(lasso2.pred)

############# consider ridge ################
# create an initial model
train.matrix.l =model.matrix(toxic~.,train.l.t[lasso.coef])[,-335]
test.matrix.l = model.matrix(toxic~.,test.l[lasso.coef])[,-335]
ridge=cv.glmnet(train.matrix.l,train.l.t$toxic,
                 alpha=0,family="binomial",
                 weights = weights,
                 nfolds = 3)

ridge.pred = predict(ridge, s = ridge$lambda.min, newx = test.matrix.l)
ridge.pred = ifelse(ridge.pred > 0.5, 1,0)
confusionMatrix(as.factor(ridge.pred), as.factor(test.label$toxic))
AUC(as.factor(ridge.pred), as.factor(test.label$toxic))
F1_Score(as.factor(test.label$toxic),as.factor(ridge.pred))



####################ridge and lasso mix
l.r=cv.glmnet(train.matrix.l,train.l.t$toxic,
                alpha=0.5,family="binomial",
                weights = weights,
                nfolds = 3)

l.r.pred = predict(l.r, s = l.r$lambda.min, newx = test.matrix.l)
l.r.pred = ifelse(l.r.pred > 0.5, 1,0)
confusionMatrix(as.factor(l.r.pred), as.factor(test.label$toxic))

#################### gbm ########################
set.seed(99)
gbm <- train(toxic ~ .
                   , data=train.l.t[lasso.coef2]
                   , method="gbm"
                   , trControl=fitControl.cv
                   , verbose=T
                   #, tuneGrid=caretGrid
                   , bag.fraction=0.75
                   , weights = model.weights
)
gbm.pred = predict(gbm, test.l, type = 'prob')
gbm.pred = ifelse(gbm.pred$X0 > 0.5, 0,1)
confusionMatrix(as.factor(gbm.pred), as.factor(test.label$toxic))
AUC(as.factor(gbm.pred), as.factor(test.label$toxic))
F1_Score(as.factor(test.label$toxic),as.factor(gbm.pred))


#############pca - fail ###################
train_lass_pca = prcomp(train.l.t[lasso.coef2][-346],
                        center = T, scale = T)
length(train_lass_pca$sdev)
# variance
pr_var = ( train_lass_pca$sdev )^2 
# % of variance
prop_varex = pr_var / sum( pr_var )
# Plot
plot( prop_varex[1:30], xlab = "Principal Component", 
      ylab = "Proportion of Variance Explained", type = "b" )
# Scree Plot
plot( cumsum( prop_varex ), xlab = "Principal Component", 
      ylab = "Cumulative Proportion of Variance Explained", type = "b" )
remove(train_lass_pca)
remove(pr_var)
remove(prop_varex)


############lda - fail ###########################
f = paste(names(train.l.t)[465], '~', 
          paste(names(train.l.t)[lasso.coef2][-346], collapse = '+'))
lda = lda(as.formula(paste(f)),data = train.l.t)
lda.pred = predict(lda, test.l)
lda.class = lda.pred$class
lda.class = ifelse(lda.class == 'X0', 0,1)
confusionMatrix(as.factor(lda.class), as.factor(test.label$toxic))
remove(lda)
remove(lda.pred)
remove(lda.class)


########################### train - raw xgb #######################

fitControl = trainControl(method = 'none',
                          summaryFunction = twoClassSummary,
                          classProbs = T)
fitControl.cv = trainControl("repeatedcv", repeats = 1,number = 3,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE)




xgbGrid <- expand.grid(nrounds = 200,
                       max_depth = 10,
                       eta = .1,#learning rage
                       gamma = 0.001, #regularization parameter.
                       colsample_bytree = .8,#default .8
                       min_child_weight = 1, # default 1
                       subsample = 0.75)

set.seed(1000)
xgb.t.base = train(toxic~., data = train.l.t, 
                           method = 'xgbTree',
                           metric = "ROC",
                           verbose = T,
                           trControl = fitControl.cv,
                           tuneGrid =xgbGrid,
                           # scale_pos_weight = fraction_0/fraction_1,
                           alpha = 1, lambda = lambda)

xgb.lambda.pred = predict(xgb.t.base, test.l)
xgb.lambda.pred = ifelse(xgb.lambda.pred == 'X0',0,1)
confusionMatrix(as.factor(xgb.lambda.pred), as.factor(test.label$toxic))

AUC(as.factor(xgb.lambda.pred), as.factor(test.label$toxic))
F1_Score(as.factor(test.label$toxic),as.factor(xgb.lambda.pred))


#### xgb - tune ###############

set.seed(1000)

xgbGrid.tune <- expand.grid(nrounds = 200,
                            max_depth = c(10,12),
                            eta = c(0.05,.1,0.2),#learning rage
                            gamma = 0.001, #regularization parameter.
                            colsample_bytree = c(.8),#default .8
                            min_child_weight = c(1), # default 1
                            subsample = 0.75)



set.seed(1000)
xgb.t.tune = train(toxic~., data = train.l.t, 
                   method = 'xgbTree',
                   metric = "ROC",
                   verbose = FALSE,
                   trControl = fitControl.cv,
                   tuneGrid =xgbGrid.tune,
                   scale_pos_weight = fraction_0/fraction_1,
                   alpha = 1, lambda = lambda)

xgb.t.tune
plot(xgb.t.tune)

xgb.lambda.t.pred = predict(xgb.t.tune, test.l)
xgb.lambda.t.pred = ifelse(xgb.lambda.t.pred == 'X0',0,1)
confusionMatrix(as.factor(xgb.lambda.t.pred), as.factor(test.label$toxic))

AUC(as.factor(xgb.lambda.t.pred), as.factor(test.label$toxic))
F1_Score(as.factor(test.label$toxic),as.factor(xgb.lambda.t.pred))


xgb.imp = varImp(xgb.t.tune)
plot(xgb.imp)
nrow(xgb.imp$importance)

#####################logistic regression ##################
logist = glm(toxic~., data = train.l.t[lasso.coef], 
             weights = model.weights,
             family='binomial')
summary(logist)
logist.pred = predict(logist, test.l, type = 'response')
logist.fac = ifelse(logist.pred>0.5,1,0)
confusionMatrix(as.factor(logist.fac), as.factor(test.label$toxic))
AUC(as.factor(logist.fac), as.factor(test.label$toxic))
F1_Score(as.factor(test.label$toxic),as.factor(logist.fac))


#########################random forest######################
model.weights = ifelse(train.l.t$toxic == "X0",
                       (1/table(train.l.t$toxic)[1]) * 0.5,
                       (1/table(train.l.t$toxic)[2]) * 0.5)



set.seed(1000)

rf <- train(toxic ~ .,data = train.l.t[lasso.coef],
                      method = "ranger",
                      verbose = TRUE,
                      weights = model.weights,
                      metric = "ROC",
                      trControl = fitControl.cv,
                      importance = 'impurity')

# Use the same seed to ensure same cross-validation splits
# ctrl$seeds <- orig_fit$control$seeds