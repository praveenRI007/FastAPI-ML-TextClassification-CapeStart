import joblib
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer, pos_tag
from collections import defaultdict
from nltk.corpus import wordnet as wn, stopwords
from sklearn import preprocessing

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def predict_a(text):
    le = preprocessing.LabelEncoder()

    # load
    clf2 = joblib.load("articles.pkl")

    df = pd.DataFrame([text], columns=['Full_Article'])

    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    df['Full_Article'] = [entry.lower() for entry in df['Full_Article']]

    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    df['Full_Article']= [word_tokenize(entry) for entry in df['Full_Article']]

    print(text)
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(df['Full_Article']):
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
        df.loc[index,'text_final'] = str(Final_words)

    loaded_tfidf = joblib.load('T_vect.pkl')

    test_new = loaded_tfidf.transform(df['text_final'])
    prediction_encoded = clf2.predict(test_new)


    le.classes_ = np.load('labels.npy',allow_pickle=True)
    prediction_decoded = le.inverse_transform(prediction_encoded)

    return prediction_decoded
