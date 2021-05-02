############## library #############
# install.packages('tidytext')
# install.packages('quanteda')
# install.packages('quanteda.textplots')
library(quanteda)
library(dplyr)
library(qdap)
library(devtools)
library(tidyverse)
library(ggplot2)
library(tidytext)
library(tm)
library(quanteda.textplots)

############# import dataset ###########
test_path = 'https://raw.githubusercontent.com/rainieluo2016/DNSC6279_Data_Mining_final_project/master/test.csv'
train_path = 'https://raw.githubusercontent.com/rainieluo2016/DNSC6279_Data_Mining_final_project/master/train.csv'
test_labels_path = 'https://raw.githubusercontent.com/rainieluo2016/DNSC6279_Data_Mining_final_project/master/test_labels.csv'
test = read.csv(test_path, encoding = 'UTF-8')
train = read.csv(train_path, encoding = 'UTF-8')
test_labels = read.csv(test_labels_path, encoding = 'UTF-8')

## remove the noise test rows from test
test_new = test
test_new = test_new[test_labels$toxic != -1,]
test_labels_new = test_labels[test_labels$toxic != -1,]

train_new = train


############ corpus on train #############
train_new$comment_text = train_new$comment_text %>% 
  gsub("[^[:graph:]]", " ",.) %>% # remove graph
  gsub('http\\S+\\s*',' ',.) %>%  # remove url
  gsub("([^A-Za-z ])+", "",.) #keep only alphabet

train_new$comment_text = tolower(train_new$comment_text)

corpus_train = corpus(train_new[,c(1:2)], text_field = 'comment_text')
summary(corpus_train,2)
head(docvars(corpus_train))
toks_train = tokens(corpus_train,
                    remove_punct = T)
toks_train = tokens_select(toks_train, pattern = stopwords('en'),
                           selection = 'remove')
toks_train = tokens_wordstem(toks_train)
head(toks_train)
print(toks_train[1])

toks_ngrams_train = tokens_ngrams(toks_train, n = 1:3)
head(toks_ngrams_train)


################### perform dfm
dfm_train_nogram = dfm(toks_train)
dfm_train_nogram = dfm_tfidf(dfm_train_nogram, scheme_tf = 'prop')
nfeat(dfm_train_nogram)



dfm_train = dfm(toks_ngrams_train)

### occur more than 10% of the documents are removed
dfm_train_freq = dfm_trim(dfm_train, 
                          max_docfreq = 0.1, 
                          docfreq_type = "prop")

dfm_train_freq.tfidf = dfm_tfidf(dfm_train_freq, scheme_tf = 'prop')
dfm_train_freq.tfidf = dfm_trim(dfm_train_freq.tfidf,
                               sparsity = 0.99)
nfeat(dfm_train_freq.tfidf)
print(dfm_train)
topfeatures(dfm_train_freq.tfidf, 10)

################ do the same on test ############
test_new$comment_text = test_new$comment_text %>% 
  gsub("[^[:graph:]]", " ",.) %>% # remove graph
  gsub('http\\S+\\s*',' ',.) %>%  # remove url
  gsub("([^A-Za-z ])+", "",.) %>% #keep only alphabet
  tolower()

corpus_test = corpus(test_new[,c(1:2)], 
                      text_field = 'comment_text')
toks_test = tokens(corpus_test, remove_punct = T, remove_numbers = T) %>%
            tokens_remove(pattern = stopwords('en')) %>%
            tokens_wordstem()
toks_ngrams_test = tokens_ngrams(toks_test, n = 1:3)

dfm_test = dfm(toks_ngrams_test)
dfm_test_freq = dfm_trim(dfm_test, 
                          max_docfreq = 0.1, 
                          docfreq_type = "prop")### occur more than 10% of the documents are removed
dfm_test_freq.tfidf = dfm_tfidf(dfm_test_freq, scheme_tf = 'prop')
dfm_test_freq.tfidf = dfm_trim(dfm_test_freq.tfidf,
                               sparsity = 0.99)
nfeat(dfm_test_freq.tfidf)
topfeatures(dfm_test_freq.tfidf, 10)

##################### intersection ##############
intersection = intersect(colnames(dfm_test_freq.tfidf),
                         colnames(dfm_train_freq.tfidf))

################# to dataframe ################
train.tfidf = as.data.frame(dfm_train_freq.tfidf)
test.tfidf = as.data.frame(dfm_test_freq.tfidf)

train.tfidf = train.tfidf[,intersection]
test.tfidf = test.tfidf[,intersection]

train.tfidf$toxic = train$toxic
train.tfidf$severe_toxic = train$severe_toxic
train.tfidf$obscene = train$obscene
train.tfidf$threat = train$threat
train.tfidf$insult = train$insult
train.tfidf$identity_hate = train$identity_hate
train.tfidf$user_id = train$id

test.tfidf$toxic = test_labels_new$toxic
test.tfidf$severe_toxic = test_labels_new$severe_toxic
test.tfidf$obscene = test_labels_new$obscene
test.tfidf$threat = test_labels_new$threat
test.tfidf$insult = test_labels_new$insult
test.tfidf$identity_hate = test_labels_new$identity_hate
test.tfidf$user_id = test_labels_new$id

train.n.0 = which(rowSums(train.tfidf[,-471]) == 0)
train.tfidf.n0 = train.tfidf[-train.n.0,]

test.n.0 = which(rowSums(test.tfidf[,-471]) == 0)
test.tfidf.n0 = test.tfidf[-test.n.0,]

write.csv(train.tfidf.n0, 'train_tfidf_norm_v2.csv', row.names = F)
write.csv(test.tfidf.n0, 'test_tfidf_norm_v2.csv', row.names = F)

####################### feature cocurrance ################
fcm_train = fcm(dfm_train_freq.tfidf)
feat <- names(topfeatures(fcm_train, 50))

fcmat_news_select <- fcm_select(fcm_train, pattern = feat, selection = "keep")

size <- log(colSums(dfm_select(fcm_train, feat, selection = "keep")))

set.seed(144)
textplot_network(fcmat_news_select, min_freq = 0.8, vertex_size = size / max(size) * 3)


######################## 