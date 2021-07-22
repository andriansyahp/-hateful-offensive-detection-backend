# Hateful-Offensive-Detection-Backend
This is a back-end service of a Hate Speech vs Offensive vs Neutral Text Classification

## Overview 
This app consists of:
+ A TensorFlow deep learning ML model;
+ A FastAPI for API access utilizing the model.

### ML Model
The machine learning can be found in the `model/` directory. </br></br> 
The dataset can be found in the `data/` directory, and it is adapted from [https://paperswithcode.com/dataset/hate-speech-and-offensive-language](https://paperswithcode.com/dataset/hate-speech-and-offensive-language). </br></br> 

### API
The API is based on FastAPI, hosted on Heroku.

#### Hosted Endpoints:
1. __Root__: 'landing page' of the API. </br>
   </br>__URL__: `https://offensive-detection-be.herokuapp.com` </br>
   </br>__Method__: GET </br>
   </br>__Body__: None </br>
   </br>__Response__: 
   ```
   {"message":"This is an API for Hate, Offensive and Neutral Speech Classifier!"} 
   ```
2. __Predict__: serves the text classification function. It takes a body consisting the message to be classified and will returns the result (2: Neutral, 1: Offensive, 0: Hate Speech).</br></br>
   __URL__: `https://offensive-detection-be.herokuapp.com/predict` </br>
   </br>__Method__: POST </br>
   </br>__Body__:
   ```
   {"speech":<the-text-to-be-classified>} 
   ```
   __Example body__: 
   ```
   {"speech":"Hi!"} 
   ```
   </br>__Response__:
   ```
   {"prediction":<result-of-classification>} 
   ```
   __Example response__: 
   ```
   {"prediction":2} 
   ```
