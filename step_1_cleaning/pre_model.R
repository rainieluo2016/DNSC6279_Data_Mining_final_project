# library
library(caret)
library(xgboost)
library(pROC)
library(e1071)
# install.packages('mldr')
library(mldr)
library(mlr)
# install.packages('RWeka')
library(RWeka)
# install.packages('ada')
library(ada)
# devtools::install_github("mlr-org/measures")
library(measures)
library(splitstackshape)
library(dplyr)
library(tibble)
# install.packages('bst')
library(bst)
library(rpart)
# install.packages('adabag')
library(adabag)

###############################################
# import the new dataset

train_new = read.csv('train_new.csv',header = T)
test_new = read.csv('test_new.csv',header = T)
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
train_full = train_full[,-1]
test_full = test_full[,-1]
### create mldf
match('toxic',names(test_full)) # 466 - 471
train_ml = mldr_from_dataframe(train_full, 
                               labelIndices = c(465:470),
                               name = 'multi-train')
summary(train_ml)
train_ml$labels

### remedial
train_remedial = remedial(train_ml)
train_remedial$labels

mldrGUI()

####plot
# number of labels per instance
plot(train_ml, type = "CH")
dev.new(width=5, height=4)
plot(train_ml, type = "LC",cex = 0.5)

##
train_br <- mldr_transform(train_ml, type = "BR")
weka_knn =  IBk(classLabel ~ ., data = train_br, 
                       control = Weka_control(K = 10))
# error - Error in terms.formula(formula, data = data) : 
# duplicated name 'abl' in data frame using '.'

###########################################################
#### mlr multilabel classification

############# prep #############
labels = colnames(train_full)[465:470]
# change into logical
train_log = train_full
train_log$toxic = as.logical(train_log$toxic)
train_log$severe_toxic = as.logical(train_log$severe_toxic)
train_log$obscene = as.logical(train_log$obscene)
train_log$threat = as.logical(train_log$threat)
train_log$insult = as.logical(train_log$insult)
train_log$identity_hate = as.logical(train_log$identity_hate)
n = dim(train_label)[1]

test_log = test_full
test_log$toxic = as.logical(test_log$toxic)
test_log$severe_toxic = as.logical(test_log$severe_toxic)
test_log$obscene = as.logical(test_log$obscene)
test_log$threat = as.logical(test_log$threat)
test_log$insult = as.logical(test_log$insult)
test_log$identity_hate = as.logical(test_log$identity_hate)

### remedial dataset
train.r_log = train_remedial$dataset[,1:470]
train.r_log$toxic = as.logical(train.r_log$toxic)
train.r_log$severe_toxic = as.logical(train.r_log$severe_toxic)
train.r_log$obscene = as.logical(train.r_log$obscene)
train.r_log$threat = as.logical(train.r_log$threat)
train.r_log$insult = as.logical(train.r_log$insult)
train.r_log$identity_hate = as.logical(train.r_log$identity_hate)

# create a task
task = makeMultilabelTask(data = train_log, target = labels)
task2 = makeMultilabelTask(data = train.r_log, target = labels)
# smote = smote(task, rate = 2, nn = 3, standardize = TRUE)

# subset for train
train.size = sample(n, size = n/3)
set.seed(1000)
train_sub_10 = train_full %>% rowid_to_column(.,'id')%>% group_by("toxic","severe_toxic","obscene","threat","insult","identity_hate") %>% 
  sample_frac(0.1, replace = F) 
train_sub_10 = c(train_sub_10$id)


##################### learner - rf  ###############
lrn.rfsrc = makeLearner("multilabel.randomForestSRC",
                        predict.type = 'prob')
rf = train(lrn.rfsrc, task, subset = 1:5000)
pred_rf = predict(rf, newdata = test_log)

names(as.data.frame(pred_rf))
performance(pred_rf, measures = list(multilabel.subset01, multilabel.hamloss, multilabel.acc,
                                   multilabel.f1, timepredict))
getMultilabelBinaryPerformances(pred_rf, measures = list(acc, mmce, auc))
calculateConfusionMatrix(pred_rf)

pred_rf_toxic = ifelse(pred_rf$data$response.toxic,1,0)
confusionMatrix(as.factor(pred_rf_toxic),
                as.factor(test_label$toxic))
MultilabelF1(pred_rf$data[1:6],pred_rf$data[13:18])

# multilabel. not available for resampling



###################### learner - ada ###############
lrn.ada = makeLearner("classif.ada",
                        predict.type = 'prob')
lrn.ada = makeMultilabelBinaryRelevanceWrapper(lrn.ada)
# lrn.ada.down = makeUndersampleWrapper(lrn.ada, usw.rate = 0.5)
train.size = sample(n, size = n/3)
ada = train(lrn.ada, task, subset = train.size)
pred_ada = predict(ada, newdata = test_log)

names(as.data.frame(pred_ada))
performance(pred_ada, measures = list(multilabel.subset01, multilabel.hamloss, multilabel.acc,
                                     multilabel.f1, timepredict))
getMultilabelBinaryPerformances(pred_ada, measures = list(acc, mmce, auc))
getMultilabelBinaryPerformances(pred_ada, measures = ppv)

pred_ada_toxic = ifelse(pred_ada$data$response.toxic,1,0)
pred_ada_hate = ifelse(pred_ada$data$response.identity_hate,1,0)
confusionMatrix(as.factor(pred_ada_toxic),
                as.factor(test_label$toxic))
confusionMatrix(as.factor(pred_ada_hate),
                as.factor(test_label$identity_hate))
MultilabelF1(pred_ada$data[1:6],pred_ada$data[13:18])
getFeatureImportance(ada)

# resampling
rdesc = makeResampleDesc(method = 'CV', stratify = F, iters = 2)
rs_ada = resample(learner = lrn.ada, task, resampling = rdesc, show.info = FALSE)


################# learner - gradient boosting ###############
lrn.bst = makeLearner("classif.bst")
lrn.bst = makeMultilabelBinaryRelevanceWrapper(lrn.bst)
# lrn.ada.down = makeUndersampleWrapper(lrn.ada, usw.rate = 0.5)
bst = train(lrn.bst, task, subset = train_sub_10)
pred_bst = predict(bst, newdata = test_log)

names(as.data.frame(pred_bst))
performance(pred_bst, measures = list(multilabel.subset01, multilabel.hamloss, multilabel.acc,
                                      multilabel.f1, timepredict))
getMultilabelBinaryPerformances(pred_bst, measures = list(acc, mmce))
getMultilabelBinaryPerformances(pred_bst, measures = ppv)

pred_bst_toxic = ifelse(pred_bst$data$response.toxic,1,0)
pred_ada_hate = ifelse(pred_bst$data$response.identity_hate,1,0)
confusionMatrix(as.factor(pred_bst_toxic),
                as.factor(test_label$toxic))
confusionMatrix(as.factor(pred_bst_hate),
                as.factor(test_label$identity_hate))
MultilabelF1(pred_bst$data[1:6],pred_bst$data[13:18])
getFeatureImportance(ada)

################# learner - adabag boosting ###############
lrn.abb = makeLearner("classif.boosting",
                      predict.type = 'prob')
lrn.abb = makeMultilabelBinaryRelevanceWrapper(lrn.abb)
# lrn.ada.down = makeUndersampleWrapper(lrn.ada, usw.rate = 0.5)
abb = train(lrn.abb, task, subset = train_sub_10)
pred_abb = predict(abb, newdata = test_log)

names(as.data.frame(pred_abb))
performance(pred_abb, measures = list(multilabel.subset01, multilabel.hamloss, multilabel.acc,
                                      multilabel.f1, timepredict))
getMultilabelBinaryPerformances(pred_abb, measures = list(acc, mmce,auc))
getMultilabelBinaryPerformances(pred_abb, measures = ppv)

pred_bst_toxic = ifelse(pred_bst$data$response.toxic,1,0)
pred_ada_hate = ifelse(pred_bst$data$response.identity_hate,1,0)
confusionMatrix(as.factor(pred_bst_toxic),
                as.factor(test_label$toxic))
confusionMatrix(as.factor(pred_bst_hate),
                as.factor(test_label$identity_hate))
MultilabelF1(pred_bst$data[1:6],pred_bst$data[13:18])
getFeatureImportance(ada)


########################################################
## 
down_train = downSample(x = train_new,
                        y = as.factor(train_label$toxic))
 
table(down_train$Class)
down_train$Class = as.numeric(down_train$Class) - 1

## 
grep("Class", colnames(down_train))
xgbtrain_down = xgb.DMatrix(as.matrix(down_train[,-505]), 
                       label =as.numeric(down_train$Class) - 1)

param <- list(max_depth = 5, eta = 0.3, subsample = 0.8,
              objective = "binary:logistic", eval_metric = "auc")
xgb_down = xgb.train(data = xgbtrain_down, nrounds = 2,param)

pred_down = predict(xgb_down, as.matrix(test_new1),type = 'prob')
xgbdown_pred = ifelse(pred_down > 0.5, 1, 0)
confusionMatrix(as.factor(xgbdown_pred), as.factor(test_label$toxic))


####
set.seed(1000)
xgbtrain = xgb.DMatrix(as.matrix(train_new), 
                                 label = train_label$toxic)
param <- list(objective = "binary:logistic", eval_metric = "auc", max_depth = 10,
              eta = 0.05,gamma = 0.01, colsample_bytree = .8,
              min_child_weight = 1,alpha = 0.2,
              subsample = 0.5)
xgb = xgb.train(data = xgbtrain, nrounds = 200,param)

xgb_pred = ifelse(predict(xgb, as.matrix(test_new))> 0.5, 1, 0)
confusionMatrix(as.factor(xgb_pred), as.factor(test_label$toxic),positive = '1')

auc(as.factor(test_label$toxic),xgb_pred)

svm = svm(Class ~.,data = down_train,
          kernel = 'radial',
          fitted = F, na.action = na.omit)


### sample model
train_new$toxic = as.factor(train_label$toxic) 
fitControl <- trainControl(method="none",classProbs=TRUE, summaryFunction=twoClassSummary)
levels(train_new$toxic) = make.names(unique(train_new$toxic))
xgbGrid <- expand.grid(nrounds = 500,
                       max_depth = 6,
                       eta = .05,#learning rage
                       gamma = 0.001, #regularization parameter.
                       colsample_bytree = .8,
                       min_child_weight = 1,
                       subsample = 1)
set.seed(13)
formula = toxic ~ .
model = train(formula, data = train_new,
              method = "xgbTree",trControl = fitControl,
              tuneGrid = xgbGrid,na.action = na.pass,metric="ROC", 
              maximize=FALSE)
xgbtree_pred = predict(model,test_new1)
xgbtree_pred = ifelse(xgbtree_pred == 'X0','0','1')
levels(test_label$toxic) = make.names(unique(test_label$toxic))
confusionMatrix(as.factor(xgbtree_pred), as.factor(test_label$toxic))
summary(test_label$toxic)
