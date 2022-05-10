#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install flask_cors
# !pip install flask


# In[2]:


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from flask import Flask
from flask import jsonify, request
from flask_cors import CORS
from waitress import serve

# In[3]:


def load_model_file(filename):
    obj = pickle.load(open(filename, 'rb'))
    
    model = obj['model']
    vectorizer = obj['vectorizer']
    
    return model, vectorizer
    return obj

def predict(sentences, model, vectorizer):
    sentences = sentences if isinstance(sentences, list) else [sentences]
    vector = vectorizer.transform(sentences)
    return model.predict(vector)


# In[4]:


model, vectorizer = load_model_file('movie_model.pkl')


# In[5]:


print(predict('this is a great movie.', model, vectorizer))


# In[6]:


print(predict('this is a horrible movie.', model, vectorizer))


# In[7]:


app = Flask(__name__)
CORS(app)

@app.route('/')
def start():
    sentence = request.args.get('sentence')

    if sentence == None: 
        return jsonify({})

    # print(sentence)
    
    result = predict(sentence, model, vectorizer)
    
    # print(result)
    return jsonify({
        'sentence': sentence,
        'sentiment': result[0]
    })

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=80)


# In[ ]:




