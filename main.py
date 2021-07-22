from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from pydantic import BaseModel
from fastapi import FastAPI

import re
import html
import emoji
# import uvicorn
import numpy as np
import nltk
import tensorflow as tf
import pickle

nltk.download('stopwords')

app = FastAPI()

model = tf.keras.models.load_model('model')
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


class Speech(BaseModel):
    speech: str


def preprocess(tweet):
    clean_twt = tweet.strip()  # trim
    clean_twt = emoji.demojize(html.unescape(clean_twt)) # translate emoji into its meaning
    clean_twt = clean_twt.lower() # lowercase
    clean_twt = re.sub(r"http\S+", "", clean_twt) # remove URL
    clean_twt = re.sub('@[\w]+','',clean_twt) # remove username
    clean_twt = re.sub('#[\w]+','',clean_twt) # remove hashtags
    clean_twt = clean_twt.replace('rt', '') # remove retweet (RT)
    clean_twt = re.sub('[^a-zA-Z]', ' ', clean_twt) # remove punctuations and numbers
    clean_twt = re.sub('\s+', ' ', clean_twt) # remove extra space

    stops = set(stopwords.words('english'))
    clean_twt = " ".join([w for w in clean_twt.split() if w not in stops])

    return clean_twt


def detection_pipeline(txt):
    txt = preprocess(txt)
    txt_seq = tokenizer.texts_to_sequences([txt])
    txt_seq = pad_sequences(txt_seq, maxlen=150)
    return np.argmax(model.predict(txt_seq))


@app.get('/')
async def index():
    return {'message': 'This is an API for Hate, Offensive and Neutral Speech Classifier!'}


@app.post('/predict')
async def predict(data: Speech):
    """ FastAPI
    Args:
        data (Reviews): json file
    Returns:
        prediction: probability of review being positive
    """
    data = data.dict()
    text = data['speech']
    print(text)
    result = detection_pipeline(text)
    print(result)
    return {
        'prediction': int(result)
    }

#
# if __name__ == '__main__':
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
