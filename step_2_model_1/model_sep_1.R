########### library ###########
library(DMwR)
# install.packages('neuralnet')
library(neuralnet )
# devtools::install_github("RomeroBarata/bimba")
library(bimba)
library(caret)
library(glmnet)
library(randomForestSRC)

###############################################
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

#########################################################

# multilabel
### create full dataset which contains both labels and features
train_full = cbind(train_new,train_label)
test_full =cbind(test_new,test_label)
### remove id from both dataset

match('id',names(test_full))
match('id',names(train_full))
train_full = train_full[,-466] # remove id
train_full = train_full[,-466] # remove comments
test_full = test_full[,-466] # remove id 
train_full = train_full[,-1] # [1:464] for all obs
test_full = test_full[,-1]
train_new = train_new[,-1]
test_new = test_new[,-1]
label = colnames(test_label[2:7])
pred_name = colnames(train_new)

#############pca - fail ###################
train_full_pca = prcomp(train_full[c(1:464)],
                        center = T, scale = T)
length(train_full_pca$sdev)
# variance
pr_var = ( train_full_pca$sdev )^2 
# % of variance
prop_varex = pr_var / sum( pr_var )
# Plot
plot( prop_varex[1:30], xlab = "Principal Component", 
      ylab = "Proportion of Variance Explained", type = "b" )
# Scree Plot
plot( cumsum( prop_varex ), xlab = "Principal Component", 
      ylab = "Cumulative Proportion of Variance Explained", type = "b" )
remove(train_full_pca)

############## scale-data ##########
scl = function(x){(x-min(x))/(max(x)-min(x))}
train_scl = as.data.frame(scale(train_new))
test_scl = as.data.frame(scale(test_new))

name = names(train_full_scl)



############ fit data with nn - fail - takes toooo long ########


f_nn = as.formula(paste("toxic + severe_toxic + obscene + threat + insult + identity_hate ~", 
                        paste(name[!name %in% label], 
                              collapse = " + ")))
nn <- neuralnet(f_nn,
                data = train_full_scl,
                hidden = c(464, 128,64,6),
                act.fct = "tanh", 
                linear.output = FALSE,
                learningrate = .1,
                lifesign = "minimal")


################### smote - too long  ############
train_1 = train_scl
train_1$toxic = as.factor(train_label$toxic)
over = trunc(summary(train_1$toxic)[1]/summary(train_1$toxic)[2])*100
train_1_smote = SMOTE(toxic ~., train_1, perc.over = over)
table(train_1_smote$toxic)
remove(train_1_smote)


################### random down sampling ########
train_down = RUS(train_1,perc_maj = 60)
table(train_down$toxic)


################### xgboost ###########
train_new$toxic = as.factor(train_label$toxic) 
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)


fitControl <- trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)
levels(train_1$toxic) = make.names(unique(train_1$toxic))
set.seed(13)
formula = toxic ~ .
xbgtree = train(formula, data = train_down,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = grid_default, na.action = na.pass,metric="ROC", 
              maximize=FALSE)

### prediction
xgbtree_pred = predict(xbgtree,test_new)
xgbtree_pred = ifelse(xgbtree_pred == 'X0','0','1')
levels(test_label$toxic) = make.names(unique(test_label$toxic))
confusionMatrix(as.factor(xgbtree_pred), as.factor(test_label$toxic))
summary(test_label$toxic)

#### tune
tune_grid <- expand.grid(
  nrounds = 100,
  eta = c(0.3,0.5,0.6),
  max_depth = c(2,3,5,10),
  gamma = c(0),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  verboseIter = FALSE, # no training log
  allowParallel = TRUE #
)

xgb_tune <- train(
  formula, data = train_down,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE,
)

plot(xgb_tune)

#### pick eta = 0.5, max = 5

grid_pick <- expand.grid(
  nrounds = 100,
  max_depth = 5,
  eta = 0.5,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

set.seed(13)
formula = toxic ~ .
xbgtree = train(formula, data = train_down,
                method = "xgbTree",trControl = fitControl,
                tuneGrid = grid_pick, na.action = na.pass,metric="ROC", 
                maximize=FALSE)
xgbtree_pred = predict(xbgtree,test_new)
xgbtree_pred = ifelse(xgbtree_pred == 'X0','0','1')
confusionMatrix(as.factor(xgbtree_pred), as.factor(test_label$toxic))
summary(test_label$toxic)


#################### lasso regression #############
x.train=model.matrix(toxic~.,train_down)[,-465]
test_scl$toxic = test_label$toxic
x.test=model.matrix(toxic~.,test_scl)[,-465] 

lasso.mod=cv.glmnet(x.train,train_down$toxic,
                    alpha=1,family="binomial")
plot(lasso.mod, xvar = 'lambda')
lasso.mod$lambda.min
lambda_final = lasso.mod$lambda.1se
coef(lasso.mod, lasso.mod$lambda.1se)

lasso=glmnet(x.train,train_down$toxic, 
                lambda = lasso.mod$lambda.1se,
                  alpha=1,family="binomial")
lasso_prob = predict(lasso, newx = x.test)
lasso_pred = ifelse(lasso_prob > 0.5, 1,0)
confusionMatrix(as.factor(lasso_pred), as.factor(test_label$toxic))

######### logistic ############
logis <- glm(toxic ~., data = train_down, family = binomial)
log_prob = predict(logis, test_scl, type = 'response')
log_pred = ifelse(log_prob > 0.5, 1,0)
confusionMatrix(as.factor(lasso_pred), as.factor(test_label$toxic))

######## random forest ########
fitControl = trainControl(method = 'cv', classProbs = TRUE, number = 3)
set.seed(1000)
rf = train(toxic ~., 
            data = train_down, 
            method ='rf', trControl = fitControl)