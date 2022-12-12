import streamlit as st
# import pandas as pd
# import numpy as np
import pickle
import re

import nltk
from nltk import pos_tag, word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# loading models
model = pickle.load(open('model.pkt', 'rb'))
tf_idf = pickle.load(open('tf_idf.pkt', 'rb'))

# state variables
wnl = WordNetLemmatizer()


# state def
def prepare_text(text):
	def get_wordnet_pos(treebank_tag):
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			return wordnet.NOUN

	text = re.sub(r'[^a-zA-Z]', ' ', text)
	text = text.split()
	text = ' '.join(text)
	text = word_tokenize(text)
	text = pos_tag(text)
	lem = []
	for i in text: lem.append(wnl.lemmatize(i[0], pos=get_wordnet_pos(i[1])))
	lem = ' '.join(lem)
	return lem


# Header
st.header('✍️Text toxicity detector')
st.markdown('This is a text toxicity detector. You can enter your text in the field below and find out the probability that your text is toxic.')

# getting user's input
text_input = st.text_input("Enter your text: ", value='I love you', key="text")
text_input = prepare_text(text_input)
sample_tfidf = tf_idf.transform([text_input])

# Result message
st.markdown(f'Chance that your text is toxic: {round(model.predict_proba(sample_tfidf)[0][1] * 100, 2)} %')

# Final message
st.success('Thank you for interacting with this model. '
			 'You can find the source code on [my GitHub 👾](https://github.com/VasilyBakshaev/Toxic_NLP)')