import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import classification_report

np.random.seed(500)

Corpus = pd.read_csv(r"articles.csv",encoding='latin-1')

# get category wise data present for each article type
# Commercial            2470
# Executives              65
# Financing                9
# Military              1677
# Others                  52
# Support & Services      26
# Training                 6

# SVM Accuracy Score ->  91.75377468060395

print(Corpus.groupby('Article_Type')['Full_Article'].nunique())

# Step - a : Remove blank rows if any.
Corpus['Full_Article'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['Full_Article'] = [entry.lower() for entry in Corpus['Full_Article']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['Full_Article']= [word_tokenize(entry) for entry in Corpus['Full_Article']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['Full_Article']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Article_Type'],test_size=0.2)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

np.save('labels.npy', Encoder.classes_)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
joblib.dump(Tfidf_vect, 'T_vect.pkl')
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)




# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=4, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

print(classification_report(Test_Y, predictions_SVM, target_names=['Commercial','Executives','Financing','Military','Others','Support & Services','Training']))

# save
joblib.dump(SVM, "articles.pkl")



# fit the training dataset on the NB classifier
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf,Train_Y)
#
# # predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_Tfidf)
# # Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)