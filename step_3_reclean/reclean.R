############## library #############
# install.packages('tidytext')
library(dplyr)
library(qdap)
library(devtools)
library(tidyverse)
library(SnowballC)
library(hunspell)
library(ggplot2)
library(tm)
library(tidytext)

############# import dataset ###########
test_path = 'https://raw.githubusercontent.com/rainieluo2016/DNSC6279_Data_Mining_final_project/master/test.csv'
train_path = 'https://raw.githubusercontent.com/rainieluo2016/DNSC6279_Data_Mining_final_project/master/train.csv'
test_labels_path = 'https://raw.githubusercontent.com/rainieluo2016/DNSC6279_Data_Mining_final_project/master/test_labels.csv'
test = read.csv(test_path)
train = read.csv(train_path)
test_labels = read.csv(test_labels_path)

## remove the noise test rows from test
test_new = test
test_new = test_new[test_labels$toxic != -1,]
test_labels_new = test_labels[test_labels$toxic != -1,]

train_new = train

sum(is.na(train_new$comment_text))
sum(is.na(test_new$comment_text))

############ corpus on train #############
train_new$comment_text = train_new$comment_text %>% 
  gsub("[^[:graph:]]", " ",.) %>% # remove graph
  gsub('http\\S+\\s*',' ',.) %>%  # remove url
  gsub("([^A-Za-z ])+", "",.) #keep only alphabet
sum(is.na(train_new$comment_text))


########## normed version
corpus = VCorpus(VectorSource(train_new$comment_text))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords('english'))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

dtm_train_noremove = DocumentTermMatrix(corpus, control = list(weighting = 
                                                        weightTfIdf))
dtm_train_999 = removeSparseTerms(dtm_train_noremove, 0.9999)
name_dtm = as.data.frame(dtm_train_999$dimnames[[2]])
dtm_train = removeSparseTerms(dtm_train, 0.99)
train.tfidf = as.data.frame(as.matrix(dtm_train))
train.tfidf_noremove = as.data.frame(as.matrix(dtm_train_999))

name_dtm_freq = findFreqTerms(dtm_train_noremove,3)

### print for slides

stem_slide = corpus$content[[1]]

# add columns
train.tfidf$id = train$id
train.tfidf$toxic = train$toxic
train.tfidf$severe_toxic = train$severe_toxic
train.tfidf$obscene = train$obscene
train.tfidf$threat = train$threat
train.tfidf$insult = train$insult
train.tfidf$identity_hate = train$identity_hate
 
########## not normed
dtm_train2 = DocumentTermMatrix(corpus, 
                               control = list(weighting = 
                               function(x) weightTfIdf(x, 
                                normalize = F)))
dtm_train2 = removeSparseTerms(dtm_train2, 0.99)
remove(dtm_train2)
train.tfidf2 = as.data.frame(as.matrix(dtm_train2))
remove(train.tfidf2)
train.tfidf2$id = train$id
train.tfidf2$toxic = train$toxic
train.tfidf2$severe_toxic = train$severe_toxic
train.tfidf2$obscene = train$obscene
train.tfidf2$threat = train$threat
train.tfidf2$insult = train$insult
train.tfidf2$identity_hate = train$identity_hate



############ corpus on test #############
test_new = test
test_new = test_new[test_labels$toxic != -1,]
test_new$comment_text = test_new$comment_text %>% 
  gsub("[^[:graph:]]", " ",.) %>%
  gsub('http\\S+\\s*',' ',.) %>% 
  gsub("([^A-Za-z ])+", "",.)
sum(is.na(test_new$comment_text))

corpus2 = VCorpus(VectorSource(test_new$comment_text))
corpus2 = tm_map(corpus2, content_transformer(tolower))
corpus2 = tm_map(corpus2, removeNumbers)
corpus2 = tm_map(corpus2, removePunctuation)
corpus2 = tm_map(corpus2, removeWords, stopwords('english'))
corpus2 = tm_map(corpus2, stemDocument)
corpus2 = tm_map(corpus2, stripWhitespace)

##### with normed

dtm_test = DocumentTermMatrix(corpus2, control = list(weighting = weightTfIdf))
dtm_test = removeSparseTerms(dtm_test, 0.99)
test.tfidf = as.data.frame(as.matrix(dtm_test))

test.tfidf$id = test_labels_new$id
test.tfidf$toxic = test_labels_new$toxic
test.tfidf$severe_toxic = test_labels_new$severe_toxic
test.tfidf$obscene = test_labels_new$obscene
test.tfidf$threat = test_labels_new$threat
test.tfidf$insult = test_labels_new$insult
test.tfidf$identity_hate = test_labels_new$identity_hate
                                                                                  
# without normed

dtm_test2 = DocumentTermMatrix(corpus2, control = list(weighting = function(x)weightTfIdf(x, normalize = F)))
dtm_test2 = removeSparseTerms(dtm_test2, 0.99)
test.tfidf2 = as.data.frame(as.matrix(dtm_test2))

test.tfidf2$id = test_labels_new$id
test.tfidf2$toxic = test_labels_new$toxic
test.tfidf2$severe_toxic = test_labels_new$severe_toxic
test.tfidf2$obscene = test_labels_new$obscene
test.tfidf2$threat = test_labels_new$threat
test.tfidf2$insult = test_labels_new$insult
test.tfidf2$identity_hate = test_labels_new$identity_hate

################# compare columns ###########
col_in_both = intersect(colnames(train.tfidf),colnames(test.tfidf))

## normalized
train.tfidf = train.tfidf[,col_in_both]
test.tfidf = test.tfidf[,col_in_both]

## un-normalized
train.tfidf2 = train.tfidf2[,col_in_both]
test.tfidf2 = test.tfidf2[,col_in_both]

################## remove all 0 rows ###########
train.n.0 = which(rowSums(train.tfidf[,-462]) == 0)
train.not.0 = which(rowSums(train.tfidf2[,-462]) == 0)
length(intersect(train.n.0,train.not.0))

test.n.0 = which(rowSums(test.tfidf[,-462]) == 0)
test.not.0 = which(rowSums(test.tfidf2[,-462]) == 0)
length(intersect(test.n.0,test.not.0))

train.tfidf.f = train.tfidf[-train.n.0,]
train.tfidf2.f = train.tfidf2[-train.n.0,]

test.tfidf.f = test.tfidf[-test.n.0,]
test.tfidf2.f = test.tfidf2[-test.n.0,]


################# write ##################
write.csv(train.tfidf.f, 'train_tfidf_norm.csv', row.names = F)
write.csv(train.tfidf2.f, 'train_tfidf.csv', row.names = F)

write.csv(test.tfidf.f, 'test_tfidf_norm.csv', row.names = F)   
write.csv(test.tfidf2.f, 'test_tfidf.csv', row.names = F) 
### set a small sample ###############
sample.num = sample(c(1:nrow(train)),100)
sample.train =train[sample.num,]

sample.train$comment_text = sample.train$comment_text %>% gsub('http\\S+\\s','',.)
sample.corpus = VCorpus(VectorSource(sample.train$comment_text))
sample.corpus = tm_map(sample.corpus, content_transformer(tolower))
sample.corpus = tm_map(sample.corpus, removeNumbers)
sample.corpus = tm_map(sample.corpus, removePunctuation)
sample.corpus = tm_map(sample.corpus, removeWords, stopwords("english"))
sample.corpus = tm_map(sample.corpus, stripWhitespace)

sample.dtm = DocumentTermMatrix(sample.corpus,
                                control = list(weightning = weightTfIdf))
sample.dtm
remove(sample.corpus)
remove(sample.dtm)
