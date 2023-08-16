import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as maperror

st.title("Welcome to Energy Demand Forecast Web Application!")
st.header("Energy Demand Forecast")
st.subheader("This application predicts the future energy demand based on historical data you provided.")
st.write('Please upload file in excel format as in example below:')

# load data from Excel file into a DataFrame
sample = pd.read_excel('energy_sample.xlsx')
styler = sample.style.hide_index()
st.write(styler.to_html(), unsafe_allow_html=True)

# Create a file uploader widget
file = st.file_uploader("Upload your file here", type=["xlsx"])

if file:
    # create target values
    y_df = pd.read_excel(file)
    y = pd.concat([y_df[i] for i in y_df.columns]).to_numpy().reshape(-1,)

    # create input X
    X = np.array([x for x in range(len(y))]).reshape(-1,1)

    # create train set and test set
    X_train, X_test = X[:len(y)-12,0].reshape(-1,1), X[len(y)-12:,0].reshape(-1,1)
    y_train, y_test = y[:len(y)-12], y[len(y)-12:]

    model = LinearRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    pred_dic = {i+1: [y_pred[i]] for i in range(12)}
    index = ['Energy demand']
    index = pd.Index(index)
    pred_df = pd.DataFrame(pred_dic).set_index(index)

    st.write(f'The 12 month predictions are listed below:')
    st.write(pred_df.style.to_html(), unsafe_allow_html=True)
    
    # Create a line chart using Matplotlib
    fig, ax = plt.subplots()
    ax.bar(pred_df.columns, pred_df.values[0,:])
    ax.set_xlabel('Month')
    ax.set_ylabel('Energer Demand')
    # Display the plot in Streamlit
    st.pyplot(fig)

    mean_error = maperror(y_test, y_pred)
    st.write(f'Average erorr of the prediction is {mean_error:.2%}')
    st.write(f'The score of the model is {model.score(X_train, y_train):.2f}. Note that 1 is the best score.')



