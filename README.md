# Toxic Comment Classification from Wiki

Data Source - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

# Introduction

This is a multi-label classification with six different labels of toxicity types: __Toxic, Severe Toxic, Obscene, Threat, Insult and Identity Hate__.

The original dataset contains two parts: the training set with 159,571 comments (observations) and the test set with 153,164 comments (observations). Each of the datasets contains 8 different columns: unique id for comment, comment and 6 different labels for toxicity.

# EDA

+ Cleaning
  + Remove URL
  + Keep English Alphabet
  + Lowercase
  + Vectorize
  + Stemming
+ TFIDF 

# Pre-Modelling

+ PCA for dimension reduction - not suitable
+ Imbalance 
  + Cost-Sensitive
  + SMOTE
  + RUS
 
# Modeling
  + Lasso
  + Ridge
  + Logistic
  + XGBOOST
  + Multi-label Model
 
 # Final Result
 | Label        | Accuracy | AUC    | f1      |
 |--------------|----------|--------|---------|
 | Toxic        | 0.851    | 0.804  | 0.913   |
 | Severe_Toxic | 0.911    | 0.873  | 0.953   |
 | Obscence     | 0.882    | 0.818  | 0.934   |
 | Threat       | 0.898    | 0.836  | 0.946   |
 | Insult       | 0.869    | 0.809  | 0.927   |
 | Identity_Hate| 0.863    | 0.811  | 0.926   |
 
 
