__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import numpy as np
import pandas as pd
import math as mt
import itertools
import sklearn.preprocessing as pre
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from collections import defaultdict, Counter
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Convolution1D, MaxPooling1D, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



############################################# DATA PREPARATION PART #############################################
### PREPARATION: GENERAL (EXERCISES 01-03)
# read preprocessed data, store in lists (clean_data_unigrams|bigrams)
reader = open("data_unigrams.txt", 'r')
labels_unigrams = list()
clean_data_unigrams = list()
for line in reader.readlines():
    line_split = line.split("\t")
    if line_split[0] == "pos":
        labels_unigrams.append(1)
    elif line_split[0] == "neg":
        labels_unigrams.append(0)
    sentence = ""
    for sent in line_split[1:]:
        sent_split = sent.split(",")
        for word in sent_split:
            if word.startswith("['"):
                sentence += word[2:-1] + " "
            elif word.endswith("']"):
                sentence += word[2:-2] + " "
            else:
                sentence += word[2:-1] + " "
    clean_data_unigrams.append(sentence)

reader = open("data_bigrams.txt", 'r')
labels_bigrams = list()
clean_data_bigrams = list()
for line in reader.readlines():
    line_split = line.split("\t")
    if line_split[0] == "pos":
        labels_bigrams.append(1)
    elif line_split[0] == "neg":
        labels_bigrams.append(0)
    sentence = ""
    for sent in line_split[1:]:
        sent_split = sent.split(",")
        for word in sent_split:
            if word.startswith("['"):
                sentence += word[2:-1] + " "
            elif word.endswith("']"):
                sentence += word[2:-2] + " "
            else:
                sentence += word[2:-1] + " "
    clean_data_bigrams.append(sentence)



############################################# APPLICATIONAL PART #############################################
### EXERCISE 01
# create feature vector employing CountVectorizer, covering 5000 most frequently occurring uni/bigrams
clean_data_01 = list()
for sentA, sentB in zip(clean_data_unigrams, clean_data_bigrams):
    clean_data_01.append(sentA + sentB)

vectorizer = CountVectorizer(analyzer='word', max_features=5000)
data_features_01 = vectorizer.fit_transform(clean_data_01)

# input data
labels_01 = np.array(labels_unigrams)
data_features_01 = np.array(data_features_01.toarray())

print("################### EXERCISE 01: Logistic Regression (10-fold cross validation) ###################")
print("labels_01", len(labels_01))
print("data_features_01", data_features_01.shape)

# create and fit the model
lrm_cv = LogisticRegression(penalty='l2', C=1/50)
lrm_cv.fit(data_features_01, labels_01)

# report resulting accuracies
cv_scores_lrm = cross_val_score(lrm_cv, data_features_01, labels_01, cv=10)
print("cross-validation scores: ", cv_scores_lrm)
cv_mean_lrm = np.mean(cv_scores_lrm)
print("mean accuracy cv: ", cv_mean_lrm)
sterr_lrm = np.std(cv_scores_lrm)/(mt.sqrt(len(cv_scores_lrm)))
print("standard error: ", sterr_lrm)



### EXERCISE 02
# # create word vector employing Word2Vec
# # output: matrix of documents, where each document is represented as average of all word vectors in the document
# data_features_02 = [[word for word in sent.split()] for sent in clean_data_01]
# data_features_02 = np.array(data_features_02)
#
# w2v_model = Word2Vec(data_features_02, min_count=45, size=5000, window=5, workers=2)
# w2v_model.init_sims(replace=True)
# index2word_set = set(w2v_model.index2word)
# w2v_model_name = "ex02_ex03"
# w2v_model.save(w2v_model_name)
#
# # calculate average word vectors for each document
# counter = 0
# doc_feature_vecs = np.zeros((len(clean_data_01), 5000), dtype="float32")
# for doc in clean_data_01:
#     feature_vec = np.zeros((5000,), dtype="float32")
#     # iterate over each word in the document
#     # if it is in w2v_model's vocabulary, add its feature vector
#     num_words = 0
#     for sent in doc:
#         for word in sent:
#             if word in index2word_set:
#                 num_words += 1
#                 feature_vec = np.add(feature_vec, w2v_model[word])
#     # divide result by the number of words to calculate average
#     feature_vec = np.divide(feature_vec, num_words)
#     doc_feature_vecs[counter] = feature_vec
#     counter += 1
#
# # input data - only for approach employing word2vec
# labels_02 = np.array(labels_unigrams)
# data_features_02 = doc_feature_vecs

# to use word vector approach (uncommented part above) uncomment following rows, until model (MLP) creation
max_doc = max([doc for doc in clean_data_01])

# set variables
nb_words = 5000
max_doc_len = len(max_doc)

# keras tokenizer: only consider 5000 most frequent words in all documents
# output: list of word indexes for each document
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(clean_data_01)
sequences = tokenizer.texts_to_sequences(clean_data_01)
word_index = tokenizer.word_index

# calculate averages of word occurrences
for doc in range(len(clean_data_01)):
    count = 0
    for word in clean_data_01[doc]:
        if word in word_index.keys():
            count += 1
    sequences[doc] = np.divide(sequences[doc], count)

# keras padding: truncate sequences to the number of words of the longest document
clean_data_02 = pad_sequences(sequences, maxlen=max_doc_len)
#clean_data_02 = pad_sequences(sequences, maxlen=nb_words)

# store vectorized data in 'data_vectors.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(labels_unigrams), pd.DataFrame(clean_data_02)], axis=1)
data_matrix.to_csv('data_vectors_ex02.csv', index=False, delimiter=',')

# input data - only for approach employing own word vector
clean_data_02 = pd.read_csv('data_vectors_ex02.csv', sep=',')
clean_data_02 = clean_data_02.iloc[np.random.permutation(len(clean_data_02))]
data_features_02 = clean_data_02.iloc[:, 1:].as_matrix()
labels_02 = clean_data_02.iloc[:, 0].as_matrix()

print("################### EXERCISE 02: MLP (10-fold cross validation) ###################")
print("labels_02", len(labels_02))
print("data_features_02", data_features_02.shape)

# create and fit the model
hidden_dims = 100 # tested for [25, 50, 75, 100]
kfold_mlp = StratifiedKFold(y=labels_02, n_folds=10, shuffle=True, random_state=True)
cv_scores_mlp = []
for i, (train, test) in enumerate(kfold_mlp):
    mlp_cv = Sequential()
    mlp_cv.add(Dense(input_dim=data_features_02.shape[1], output_dim=hidden_dims, activation='relu', init='uniform'))
    mlp_cv.add(Dropout(0.5))
    mlp_cv.add(Dense(output_dim=1, activation='sigmoid', init='uniform'))
    # compile model
    mlp_cv.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    mlp_cv.fit(data_features_02[train], labels_02[train], nb_epoch=2, verbose=0)
    # evaluate model
    cv_scores = mlp_cv.evaluate(data_features_02[test], labels_02[test], verbose=0)
    cv_scores_mlp.append(cv_scores[1] * 100)

# report resulting accuracies
print("cross-validation scores: ", cv_scores_mlp)
cv_mean_mlp = np.mean(cv_scores_mlp)
print("mean accuracy cv: ", cv_mean_mlp)
sterr_mlp = np.std(cv_scores_mlp)/(mt.sqrt(len(cv_scores_mlp)))
print("standard error: ", sterr_mlp)



### EXERCISE 03
# create a word vector
# output: matrix of documents, where each document is represented as array of integers (words)
# to use word vector approach (uncommented part above) uncomment following rows, until model (MLP) creation
max_doc = max([doc for doc in clean_data_unigrams])

# set variables
nb_words = 5000
max_doc_len = len(max_doc)

# keras tokenizer: only consider 5000 most frequent words in all documents
# output: list of word indexes for each document
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(clean_data_unigrams)
sequences = tokenizer.texts_to_sequences(clean_data_unigrams)
word_index = tokenizer.word_index

# for doc in range(len(clean_data_unigrams)):
#     count = 0
#     for word in clean_data_unigrams[doc]:
#         if word in word_index.keys():
#             count += 1
#     sequences[doc] = np.divide(sequences[doc], count)

# keras padding: truncate sequences to the number of words of the longest document
clean_data_03 = pad_sequences(sequences, maxlen=max_doc_len)
#clean_data_03 = pad_sequences(sequences, maxlen=nb_words)

# store vectorized data in 'data_vectors.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(labels_unigrams), pd.DataFrame(clean_data_03)], axis=1)
data_matrix.to_csv('data_vectors_ex03.csv', index=False, delimiter=',')

clean_data_03 = pd.read_csv('data_vectors_ex03.csv', sep=',')
clean_data_03 = clean_data_03.iloc[np.random.permutation(len(clean_data_03))]
data_features_03 = clean_data_03.iloc[:, 1:].as_matrix()
labels_03 = clean_data_03.iloc[:, 0].as_matrix()

print("################### EXERCISE 03: CNN (10-fold cross validation) ###################")
print("labels_03", len(labels_03))
print("data_features_03", data_features_03.shape)

# set variables
max_features = len(word_index)+1 # vocabulary: number of features/unique tokens after limitation of data to most frequent words
max_len = max_doc_len # maximum document/sequence length - all documents are padded to this length
embedding_dims = 100 # vocabulary mapped onto x dimensions
feature_maps = 25 # number of feature maps for each filter size
filter_size = 5 # size of applied filter, covering at least bigrams = 2
hidden_dims = 50
batch_size = 16
pool_length = 2

### create and fit the model
kfold_cnn = StratifiedKFold(y=labels_03, n_folds=10, shuffle=True, random_state=True)
cv_scores_cnn = []
for i, (train, test) in enumerate(kfold_cnn):
    cnn_cv = Sequential()
    cnn_cv.add(Embedding(max_features, embedding_dims, input_length=max_len, dropout=0.5))
    cnn_cv.add(Convolution1D(nb_filter=feature_maps, filter_length=filter_size, activation='relu'))
    cnn_cv.add(MaxPooling1D(pool_length=pool_length))
    cnn_cv.add(Flatten())
    cnn_cv.add(Dense(hidden_dims, activation='relu'))
    cnn_cv.add(Dropout(0.2))
    cnn_cv.add(Dense(1, activation='sigmoid'))
    # compile model
    cnn_cv.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    cnn_cv.fit(data_features_03[train], labels_03[train], batch_size=batch_size, nb_epoch=2, verbose=0)#, validation_data=(data_features_03[test], labels_03[test]))
    # evaluate model
    cv_scores = cnn_cv.evaluate(data_features_03[test], labels_03[test], verbose=0)
    cv_scores_cnn.append(cv_scores[1] * 100)

# report resulting accuracies
print("cross-validation scores: ", cv_scores_cnn)
cv_mean_cnn = np.mean(cv_scores_cnn)
print("mean accuracy cv: ", cv_mean_cnn)
sterr_cnn = np.std(cv_scores_cnn)/(mt.sqrt(len(cv_scores_cnn)))
print("standard error: ", sterr_cnn)