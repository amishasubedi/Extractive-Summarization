#!/usr/bin/env python
# coding: utf-8

# In[1]:


import chardet

with open('Chat.csv', 'rb') as f:
    result = chardet.detect(f.read())

print(result['encoding'])


# ## Data Preprocessing  

# In[2]:


import pandas as pd
import re
import nltk
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


# In[3]:


# import pandas as pd
df = pd.read_csv('Chat.csv', encoding='Windows-1252')


# In[4]:


df.head()


# In[5]:


# Select columns that have a name
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Remove columns that have only NaN values
df = df.dropna(axis=1, how='all')


# In[6]:


df.head()


# In[7]:


def preprocess_text(text):
    # Loading spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Check if text is a string
    if isinstance(text, str):
        # Converting text to lowercase
        text = text.lower()

        # Removing extra whitespaces
        text = ' '.join(text.split())

        # Removing punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Lemmatization
        doc = nlp(text)
        text = ' '.join([token.lemma_ for token in doc])

    return text


# In[8]:


# Preprocessing text data
df['preprocessed_text'] = df['Answer'].apply(preprocess_text)


# In[10]:


df['preprocessed_text'] = df['preprocessed_text'].astype(str)


# In[9]:


# Saving preprocessed data as a new CSV file
df.to_csv('preprocessed_file.csv', index=False)


# ## NLP Analysis

# In[10]:


df.head()


# In[11]:


from sumy.summarizers.lex_rank import LexRankSummarizer

summarizer = LexRankSummarizer()


# In[12]:


from sumy.parsers.plaintext import PlaintextParser

document = PlaintextParser.from_string(df['preprocessed_text'], Tokenizer("english"))


# In[13]:


def get_summary(text, length=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, length)
    summary_text = "\n".join([str(sentence) for sentence in summary])
    return summary_text


# In[25]:


df['SummaryOfficial'] = df['Answer'].apply(get_summary)

df['Summary(411)'] = df['Answers(411)'].apply(get_summary)


# In[26]:


df.head()


# In[27]:


df['SummaryOfficial'][0]


# In[28]:


df['preprocessed_text'][0]


# In[29]:


df['Answers(411)'][0]


# In[30]:


df['Summary(411)'][0]


# In[ ]:




