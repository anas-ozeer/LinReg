import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn as sk


from streamlit_option_menu import option_menu


selected = option_menu(menu_title="Title", options=["Data", "Visualization", "Prediction"], orientation="horizontal")

df = pd.read_csv("housing.csv")

if selected == "Data":
    st.markdown("### Data Exploration")

    n = st.number_input("Number of Rows", 5, 20)
    st.dataframe(df.head(n))

if selected == "Visualization":
    import plotly.express as px
    st.markdown("### Data Visualization")

    # Columns in a list
    columns = df.select_dtypes(include = ["number"]).columns.to_list()

    # Create a list of variables that the user can select from to create a correlation matrix
    selected_vars = st.multiselect("Select the variables for correlation matrix", options=columns)

    # Assuming df is your DataFrame and selected_vars is a list of column names
    correlation_matrix = df[selected_vars].corr()

    # Plot the heatmap using Plotly Express
    fig = px.imshow(correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    labels=dict(color='Correlation'),
                    text_auto=True)
    
    fig.update_layout(title='Correlation Matrix')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

if selected == "Prediction":


    X = df.drop(columns=["Address","Price"])
    y = df["Price"]

    from sklearn.model_selection import  train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()

    lr.fit(X_train,y_train)

    y_pred = lr.predict(X_test)

    from sklearn.metrics import mean_absolute_error

    mse = mean_absolute_error(y_test, y_pred)

    st.write("Mean absolute error = ", mse)
