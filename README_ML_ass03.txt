README for HS Machine Learning, Assignment 03

lecturer: Cagri Cöltekin
summer 2016

task: Assignment 03
deadline: August 01, 2016

------------------------------------------------------------------------------------------
GENERAL INFORMATION
----------------------------------------------

- Two Python files have been created, containing all relevant information for the present assignment. They will be described briefly in the following.

- Packages providing required capabilities have been used and imported within the present assignment submission.

- Comments are provided for all exercised, explaining the procedure briefly. The following description, therefore, should be considered as a general overview without going into the very details.

- The submitted Python files only contain code and comments. Please find all results for the present assignment submission in the additionally submitted PDF 'ML_Assignment03_DS'. Note that the data used for exercises 1-3 are stored in TXT (see 'data_unigrams.txt' and 'data_bigrams.txt') and CSV files (see 'data_vectors_ex02.csv' and 'data_vectors_ex03.csv'). Additionally, the created Word2Vec model is stored in 'ex02_ex03'. All files can be found in the present submission, or be newly created by employing the respective parts of code.

- Please note: The file containing all movie reviews (folder: ‚review_polarity‘) has been zipped inside the folder ‚ml_assignment03_DS‘. In order to use the data to run the code in the Python file ‚ML_ass03_Preprocessing.py‘, please unzip the folder containing movie reviews and ensure, the path is equal to the path defined in ‚ML_ass03_Preprocessing.py‘.

----------------------------------------------
IMPLEMENTATION OVERVIEW
----------------------------------------------

- The Python file 'ML_ass03_Preprocessing.py' includes all methods required for
  pre-processing the provided movie reviews:
  		-> the content of all movie reviews in all directories (see folders following
  		   '/review_polarity/') is read
  		-> pre-processing includes tokenization (without splitting contractions), deletion
  		   of sentence markers, exclusion of stopwords and creation of bigrams
  		-> unigrams and bigrams are stored separately in TXT files 'data_unigrams.txt' and
  		   'data_bigrams.txt', including the respective class labels
  		-> notes concerning pre-processing, e.g. exclusion of stopwords, can be found in
  		   the submitted PDF 'ML_Assignment03_DS'
- The Python file 'ML_ass03_Models.py' includes the code for all created models.
  Additionally, Print-Statements indicate the results for the respective tasks (Logistic
  Regression, Multi-Layer Perceptron, Convolutional Neural Network).
  		-> pre-processed data (unigrams, bigrams and respective class labels) are read and
  		   stored in variables
  		-> for LG and MLP, uni- and bigrams are concatenated into one matrix representing
  		   the data for all documents
  		-> LG: sklearn's CountVectorizer is applied to the data (uni- & bigrams), covering
  		   only the 5000 most frequently occurring uni-/bigrams. The model uses L2 regula-
  		   rization with parameter lambda (= 50). Ten-fold cross validation is employed,
  		   mean accuracy and standard error are computed.
  		-> MLP: two approaches are presented (Word2Vec vs. own created document vector,
  		   containing relative frequencies for 5000 most frequently occurring words in all
  		   documents). The code for the Word2Vec approach has been uncommented, while the
  		   part covering the own created word vector remains active. Here, keras' pre-
  		   processing package (including Tokenizer and pad_sequences) is employed. The
  		   vocabulary is limited to 5000 most frequently occurring words in all documents,
  		   the matrix' dimension includes the length of the longest document. Shorter
  		   documents were padded to this size. The resulting matrix is stored in CSV file
  		   'data_vectors_ex02.csv' and contains both uni- and bigrams. Several hidden
  		   layers have been tested, best results have been obtained for 100. Ten-fold
  		   cross validation is employed, mean accuracy and standard error are computed.
  		-> CNN: input data is constructed similarly to the own created word vector for
  		   MLP, but in this context, only unigrams are covered and no average (or rela-
  		   tive frequencies) are calculated. The resulting matrix is stored in CSV file
  		   'data_vectors_ex03.csv'. Several settings have been tested, best results have
  		   been obtained for the settings as reported in 'ML_ass03_Models.py'. Ten-fold
  		   cross validation is employed, mean accuracy and standard error are computed.
  		   A description of the here created CNN can be found in the submitted PDF 'ML_
  		   Assignment03_DS'.


------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
author: Daniela Stier
date: 31/07/2016

