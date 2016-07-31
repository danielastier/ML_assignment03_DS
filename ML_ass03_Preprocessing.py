__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import os as os
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag, bigrams, corpus


############### IMPLEMENTATION OF HELPER METHODS ###############
### read in input text
def read_file(infile):
    reader = open(infile, 'r', encoding='utf-8')
    intext = reader.read()
    reader.close()
    return intext

### return individual sentences of an input text
def extract_sentences(input_text):
    sentences = list()
    input_text = input_text.split("\n")
    for sent in input_text:
        sentences.append(sent)
    return sentences

### tokenize input text - nltk.TweetTokenizer does not split contractions
def tokenize_sentences(input_list):
    sent_tokens = list()
    for sent in input_list:
        sent_tokens.append(TweetTokenizer().tokenize(sent))
    return sent_tokens

### add pos-information to sentences of an input text - nltk as reliable
# def pos_tag_sentences(input_list):
#     sentences_tagged = pos_tag(input_list)
#     return sentences_tagged

### delete stop words from text using NLTK stopword list
# !for calculations concerning unigrams and bigrams!
def delete_stopwords(input_list):
    intext = list()
    stops = corpus.stopwords.words("english")
    for sent in input_list:
        temp = list()
        [temp.append(word) for word in sent if word.lower() not in stops]
        intext.append(temp)
    return intext

### delete sentence markers (per sentence) from pre-processed list
def delete_sent_markers(input_list):
    sent_markers = [".", "!", "?", ". . .", "(", ")", '"', ",", "-", "'", "~", "*", ":", "&", ";", "=", "/", "--"]
    intext = [[word for word in sent if word not in sent_markers] for sent in input_list]
    return intext

### create possible bigrams (per sentence) from pre-processed list
def create_bigrams(input_list):
    bgrams = [list(bigrams(sent)) for sent in input_list]
    ngrams_formatted = [[str(ngram[0]+"/"+ngram[1]) for ngram in sent] for sent in bgrams]
    return ngrams_formatted


############### PREPROCESSING ###############
path_neg = "review_polarity/txt_sentoken/neg/"
path_pos = "review_polarity/txt_sentoken/pos/"
#path_neg = "review_polarity/ds_test/neg/"
#path_pos = "review_polarity/ds_test/pos/"
neg = "neg"
pos = "pos"

class_labels = ((path_neg, neg), (path_pos, pos))

# read in data, pre-process input texts, create bigrams and unigram
# output: all sentences of one document to a single line, including class labels
with open("data_unigrams.txt", 'a', newline='') as w1:
    with open("data_bigrams.txt", 'a', newline='') as w2:
        for entry in class_labels:
            label = entry[1]
            # read all documents in directory
            for file in os.listdir(entry[0]):
                w1.write(label + "\t")
                w2.write(label + "\t")
                if not file == ".DS_Store":
                    # read file content
                    content = read_file(str(entry[0] + file))
                    # extract sentences
                    cont_sentences = extract_sentences(content)
                    # tokenize sentences
                    cont_tokens = tokenize_sentences(cont_sentences)
                    # delete sentence markers from sentences
                    cont_wo_marker = delete_sent_markers(cont_tokens)
                    # delete stopwords from sentences
                    cont_wo_stops = delete_stopwords(cont_wo_marker)
                    # create bigrams from sentences (excluding stopwords!)
                    cont_bigrams = create_bigrams(cont_wo_stops)

                    # store pre-processed data in 'data.txt' file
                    for sent in cont_wo_stops:
                        if not len(sent) == 0:
                            w1.write(str(sent) + "\t")
                    for sent in cont_bigrams:
                        if not len(sent) == 0:
                            w2.write(str(sent) + "\t")
                w1.write("\n")
                w2.write("\n")
        w2.close()
    w1.close()
