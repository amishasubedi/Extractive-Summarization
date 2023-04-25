#!/usr/bin/env python
# coding: utf-8

# In[29]:


import chardet

with open('Chat.csv', 'rb') as f:
    result = chardet.detect(f.read())

print(result['encoding'])


# ## Data Preprocessing  

# In[30]:


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


# In[31]:


# import pandas as pd
df = pd.read_csv('Chat.csv', encoding='Windows-1252')


# In[ ]:





# In[32]:


df.head()


# In[33]:


# Select columns that have a name
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Remove columns that have only NaN values
df = df.dropna(axis=1, how='all')


# In[34]:


df.head()


# In[35]:


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


# In[36]:


# Preprocessing text data
df['preprocessed_text'] = df['Answer'].apply(preprocess_text)


# In[37]:


df['preprocessed_text'] = df['preprocessed_text'].astype(str)


# In[38]:


# Saving preprocessed data as a new CSV file
df.to_csv('preprocessed_file.csv', index=False)


# ## NLP Analysis

# In[39]:


df.head()


# In[40]:


from sumy.summarizers.lex_rank import LexRankSummarizer

summarizer = LexRankSummarizer()


# In[41]:


from sumy.parsers.plaintext import PlaintextParser

document = PlaintextParser.from_string(df['preprocessed_text'], Tokenizer("english"))


# In[42]:


def get_summary(text, length=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = summarizer(parser.document, length)
    summary_text = "\n".join([str(sentence) for sentence in summary])
    return summary_text


# In[43]:


df['SummaryOfficial'] = df['Answer'].apply(get_summary)

df['Summary(411)'] = df['Answers(411)'].apply(get_summary)


# In[44]:


df.head()


# In[45]:


df['SummaryOfficial'][0]


# In[46]:


df['preprocessed_text'][0]


# In[47]:


df['Answers(411)'][0]


# In[48]:


df['Summary(411)'][0]


# ## SC Data

# In[49]:


df1 = pd.read_csv('sc_voter_faq.csv')


# In[50]:


df1.head()


# In[51]:


df1['Summary'] = df1['Answer'].apply(get_summary)


# In[52]:


df1.head()


# In[54]:


df1['Summary'][0]


# In[55]:


df1['Answer'][0]


# In[ ]:




