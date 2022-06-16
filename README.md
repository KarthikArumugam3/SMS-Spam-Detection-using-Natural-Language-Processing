# SMS-Spam-Detection-using-Natural-Language-Processing
## Project Description
### Working on SMS dataset - 

* This dataset contains set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
* It has 2 message classes & the model trained on these 2 classes to detect Spam messages.
 
The 2 classes of mesages are as following:- 
* Ham
* Spam

## Tech
The project uses the following technologies/packages :- 

- ![Python](https://img.shields.io/badge/-Python-black?style=flat-square&logo=Python)
- ![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
- ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
- ![Pandas](https://img.shields.io/badge/-Pandas-black?style=flat-square&logo=Pandas)
- ![Numpy](https://img.shields.io/badge/-Numpy-black?style=flat-square&logo=Numpy)
- ![Matplotlib](https://img.shields.io/badge/-Matplotlib-black?style=flat-square&logo=Matplotlib)
- ![Seaborn](https://img.shields.io/badge/-Seaborn-black?logo=seaborn&logoColor=white)
- [![Editor](https://img.shields.io/badge/Editor-VSCode-blue?style=flat-square&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/)

### Representing text as numerical data:-
The raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.

Use of CountVectorizer to convert text into matrix of token counts.
 
```
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
```
Sample Text:-
```
simple_text = ['This is Karthik','Karthik is calling you','Karthik is a good guy']
```
After fitting the simple text using vect object
```
vect.vocabulary_
o/p:- {'this': 5, 'is': 3, 'karthik': 4, 'calling': 0, 'you': 6, 'good': 1, 'guy': 2}
```

Getting the Document-Term matrix:-
```
simple_train_dtm = vect.transform(simple_text)
simple_train_dtm
```
Coverting the sparse matrix to a dense matrix , we get boolen value for the words present in the sentence at the respective position.
```
simple_train_dtm.toarray()
o/p:-
array([[0, 0, 0, 1, 1, 1, 0],
       [1, 0, 0, 1, 1, 0, 1],
       [0, 1, 1, 1, 1, 0, 0]], dtype=int64)
```

### Process of CountVectorizer:-
From Scikit-learn doc:-
- Each individual token occurrence frequency (normalized or not) is treated as a feature.
- The vector of all the token frequencies for a given document is considered a multivariate sample.

A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus.

We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or "Bag of n-grams" representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.
```
pd.DataFrame(data=simple_train_dtm.toarray(), columns=vect.get_feature_names())
```
O/P:-
|  | calling | good  | guy | is  | karthik |  this  | you|
|--- | --- | --- | --- |--- |--- |--- |--- |
|0|0|0|0|1|1|1|0|
|1|1|0|0|1|1|0|1|
|2|0|1|1|1|1|0|0|

