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

########## data import #########
train = read.csv('../project_code_6_reclean/train_tfidf_norm.csv',header = T)
test = read.csv('../project_code_6_reclean/test_tfidf_norm.csv',header = T)

train.toxic = train[,c(1:461,463)]
test.toxic = test[,c(1:461,463)]

########## assign weights ###########
fraction_0 <- 1 - sum(train.toxic$toxic == 0) / nrow(train.toxic)
fraction_1 <- 1 - sum(train.toxic$toxic == 1) / nrow(train.toxic)
# assign that value to a "weights" vector
weights <- numeric(nrow(train.toxic))
weights[train.toxic$toxic == 0] <- fraction_0
weights[train.toxic$toxic == 1] <- fraction_1
sum(weights)

########## lasso - original weight ####################
# create an initial model
train.t.matrix =model.matrix(toxic~.,train.toxic)[,-462]
test.t.matrix = model.matrix(toxic~.,test.toxic)[,-462]

set.seed(1234)
lasso=cv.glmnet(train.t.matrix,train.toxic$toxic,
                alpha=1,family="binomial",
                weights = weights,
                nfolds = 3)
plot(lasso,xvar = 'lambda')
lambda = lasso$lambda.1se
lasso.coef = coef(lasso, lasso$lambda.1se)@i
lasso.coef = lasso.coef[-1]-1
# lasso.coef[435] = 462
coef.lasso.final = as.data.frame(as.matrix(coef(lasso, lasso$lambda.min)))
coef.lasso.final = coef.lasso.final[c(3:462),]


### rank
lasso.rank = data.frame(colnames(train.toxic)[1:460],coef.lasso.final)
lasso.rank = lasso.rank[order(abs(lasso.rank$coef.lasso.final)),]
tail(lasso.rank,15)


lasso.pred = predict(lasso, s = lasso$lambda.min, newx = test.t.matrix)
lasso.pred = ifelse(lasso.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.pred), as.factor(test.toxic$toxic))
AUC(as.factor(lasso.pred), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso.pred))

######### lasso_severe_toxic #############

train.st = train[,c(1:461,464)]
test.st = test[,c(1:461,464)]

# assign that value to a "weights" vector
weights_st <- numeric(nrow(train.st))
weights_st[train.st$severe_toxic == 0] <- fraction_0
weights_st[train.st$severe_toxic == 1] <- fraction_1

# create an initial model
train.st.matrix =model.matrix(severe_toxic~.,train.st)[,-462]
test.st.matrix = model.matrix(severe_toxic~.,test.st)[,-462]

set.seed(1234)
lasso.st=cv.glmnet(train.st.matrix,train.st$severe_toxic,
                alpha=1,family="binomial",
                weights = weights_st,
                nfolds = 3)

### rank
coef.lasso.st = as.data.frame(as.matrix(coef(lasso.st, 
                                             lasso.st$lambda.min)))
coef.lasso.st = coef.lasso.st[c(3:462),]
lasso.rank.st = data.frame(colnames(train.st)[1:460],coef.lasso.st)
lasso.rank.st = lasso.rank.st[order(abs(lasso.rank.st$coef.lasso.st)),]
tail(lasso.rank.st,15)



########## lasso - no weight ####################
set.seed(1234)
lasso.no.w=cv.glmnet(train.t.matrix,train.toxic$toxic,
                alpha=1,family="binomial",
                # weights = weights,
                nfolds = 3)
# plot(lasso,xvar = 'lambda')
# lambda = lasso$lambda.1se
# lasso.coef = coef(lasso, lasso$lambda.1se)@i
# lasso.coef = lasso.coef[-1]-1
# lasso.coef[435] = 462

lasso.pred.no.w = predict(lasso.no.w, s = lasso.no.w$lambda.min, 
                          newx = test.t.matrix)
lasso.pred.no.w = ifelse(lasso.pred.no.w > 0.5, 1,0)
confusionMatrix(as.factor(lasso.pred.no.w), as.factor(test.toxic$toxic))
AUC(as.factor(lasso.pred.no.w), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso.pred.no.w))



###########lasso - tune the weights
########## lasso - original weight ####################
fraction2_1 <- 1 - sum(train.toxic$toxic == 1) * 2 / nrow(train.toxic)
# assign that value to a "weights" vector
weights2 <- numeric(nrow(train.toxic))
weights2[train.toxic$toxic == 0] <- fraction_0
weights2[train.toxic$toxic == 1] <- fraction2_1


# model
set.seed(1234)
lasso2=cv.glmnet(train.t.matrix,train.toxic$toxic,
                alpha=1,family="binomial",
                weights = weights2,
                nfolds = 3)

# lambda = lasso$lambda.1se
lasso2.coef = coef(lasso2, lasso2$lambda.1se)@i
lasso2.coef = lasso2.coef[-1]-1
# lasso.coef[435] = 462

lasso2.pred = predict(lasso2, s = lasso2$lambda.min, newx = test.t.matrix)
lasso2.pred = ifelse(lasso2.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso2.pred), as.factor(test.toxic$toxic))
AUC(as.factor(lasso2.pred), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso2.pred))

#####################xgboost - lambda vs variable selection ###############
train.toxic.factor  =train.toxic
train.toxic.factor$toxic = as.factor(train.toxic.factor$toxic)
levels(train.toxic.factor$toxic) = make.names(unique(train.toxic.factor$toxic))


fitControl = trainControl(method = 'none',
                          summaryFunction = twoClassSummary,
                          classProbs = T)
fitControl.cv = trainControl("cv",number = 3,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE)

xgbGrid <- expand.grid(nrounds = 300,
                       max_depth = 12,
                       eta = .1,#learning rage
                       gamma = 0.001, #regularization parameter.
                       colsample_bytree = .8,#default .8
                       min_child_weight = 1, # default 1
                       subsample = 0.75)

set.seed(1000)
xgb.t.base = train(as.factor(toxic)~., data = train.toxic.factor, 
                   method = 'xgbTree',
                   metric = "ROC",
                   verbose = T,
                   trControl = fitControl,
                   tuneGrid =xgbGrid,
                   scale_pos_weight = fraction_0/fraction_1,
                   alpha = 1, lambda = lambda)

xgb.lambda.pred = predict(xgb.t.base, test.toxic)
xgb.lambda.pred = ifelse(xgb.lambda.pred == 'X0',0,1)
confusionMatrix(as.factor(xgb.lambda.pred), as.factor(test.toxic$toxic))
AUC(as.factor(xgb.lambda.pred), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(xgb.lambda.pred))


set.seed(1000)
xgb.t.base2 = train(as.factor(toxic)~., data = train.toxic.factor[,c(lasso.coef,462)], 
                   method = 'xgbTree',
                   metric = "ROC",
                   verbose = T,
                   trControl = fitControl,
                   tuneGrid =xgbGrid,
                   scale_pos_weight = fraction_0/fraction_1,
                   #alpha = 1, lambda = lambda
                   )

xgb.lambda.pred2 = predict(xgb.t.base2, test.toxic[,lasso.coef])
xgb.lambda.pred2 = ifelse(xgb.lambda.pred2 == 'X0',0,1)
confusionMatrix(as.factor(xgb.lambda.pred2), as.factor(test.toxic$toxic))
AUC(as.factor(xgb.lambda.pred2), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(xgb.lambda.pred2))

xgb.imp = varImp(xgb.t.base2)
xgb.row = row.names(xgb.imp$importance)[which(xgb.imp$importance$Overall >1)] 

# apply selected row on lasso
train.t.matrix.xgb =model.matrix(toxic~.,train.toxic[,c(xgb.row,'toxic')])[,-100]
test.t.matrix.xgb = model.matrix(toxic~.,test.toxic[,c(xgb.row,'toxic')])[,-100]

set.seed(1234)
lasso.xgb=cv.glmnet(train.t.matrix.xgb,train.toxic$toxic,
                alpha=1,family="binomial",
                weights = weights,
                nfolds = 3)
plot(lasso,xvar = 'lambda')

lasso.xgb.pred = predict(lasso.xgb, 
                         s = lasso.xgb$lambda.min, 
                         newx = test.t.matrix.xgb)
lasso.xgb.pred = ifelse(lasso.xgb.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.xgb.pred), as.factor(test.toxic$toxic))
AUC(as.factor(lasso.xgb.pred), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso.xgb.pred))

#################### SMOTE ######################
smote.train = SMOTE(train.toxic, perc_min = 50, k = 3)
table(smote.train$toxic)

### lasso matrix

train.t.matrix.s =model.matrix(toxic~.,smote.train)[,-462]

# model
set.seed(1234)
lasso.smote=cv.glmnet(train.t.matrix.s,smote.train$toxic,
                 alpha=1,family="binomial",
                 nfolds = 3)

# lambda = lasso$lambda.1se
# lasso.coef = coef(lasso2, lasso2$lambda.1se)@i
# lasso2.coef = lasso2.coef[-1]-1
# lasso.coef[435] = 462

# prediction

lasso.smote.pred = predict(lasso.smote, 
                         s = lasso.smote$lambda.min, 
                         newx = test.t.matrix)
lasso.smote.pred = ifelse(lasso.smote.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.smote.pred), as.factor(test.toxic$toxic))
AUC(as.factor(lasso.smote.pred), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso.smote.pred))


#################### NB ###################
nb.grid = expand.grid(
  usekernel = c(TRUE),
  fL = 2,
  adjust = 3
)


nb = train(toxic~., data = train.toxic.factor, method = 'nb',
           trControl = fitControl, 
           tuneGrid = nb.grid,verbose = T)

nb.pred = predict(nb, 
             newx = test.toxic)
lasso.smote.pred = ifelse(lasso.smote.pred > 0.5, 1,0)
confusionMatrix(as.factor(lasso.smote.pred), as.factor(test.toxic$toxic))
AUC(as.factor(lasso.smote.pred), as.factor(test.toxic$toxic))
F1_Score(as.factor(test.toxic$toxic),as.factor(lasso.smote.pred))

#################### PCA ##################
train_lasso_pca = prcomp(train.toxic[,lasso.coef],
                        center = T, scale = F)
# length(train_lass_pca$sdev)
# variance
pr_var = ( train_lasso_pca$sdev )^2 
# % of variance
prop_varex = pr_var / sum( pr_var )
# Plot
plot( prop_varex[1:300], xlab = "Principal Component", 
      ylab = "Proportion of Variance Explained", type = "b" )
# Scree Plot
plot( cumsum( prop_varex ), xlab = "Principal Component", 
      ylab = "Cumulative Proportion of Variance Explained", type = "b" )
remove(train_lasso_pca)
remove(prop_varex)
