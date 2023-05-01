# import pickle
# from flask import Flask, request, jsonify
# import pandas as pd

# app = Flask(__name__)

# with open('ridge1.pkl', 'rb') as file:
#     model = pickle.load(file)

# @app.route('/', methods=['GET'])
# def predict():
#     text=request.args.get('text')
#     s = pd.Series([text])
#     y_pred = model.predict(s)
#     print(y_pred)
#     if y_pred == "Hate_Speech":
#         output = {"result": "This is a hate speech."}
#     elif y_pred == "Offensive_Speech":
#         output = {"result": "This is an offensive speech."}
#     else:
#         output = {"result": "This is a safe speech."}

#     response = jsonify(output)
#     response.headers.add('Content-Type', 'application/json')
#     return response



# if __name__ == '__main__':
#     app.run()

import pickle
import pandas as pd
import streamlit as st

with open('grid.pkl', 'rb') as file:
    model = pickle.load(file)

def app():
    st.title("Hate Speech Prediction App")
    text = st.text_input("Enter your text here")
    if st.button("Predict"):
        s = pd.Series([text])
        y_pred = model.predict(s)
        st.write("Prediction:", y_pred[0])

if __name__ == '__main__':
    app()

