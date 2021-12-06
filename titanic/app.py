import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb # XGBoost stuff

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
    

        var_age = st.sidebar.slider('Age', 0.0,80.1)
        var_fare = st.sidebar.slider('Fare', 0,512,2)
        var_sp = st.sidebar.slider('SP', 0,10,1)


        var_pclass = st.sidebar.selectbox('Pclass',(1,2,3))
        var_sex = st.sidebar.selectbox('Sex',('male','female'))
        var_embarked = st.sidebar.selectbox('Embarked',('S','C','Q'))  

        data = {
                'Age': var_age,
                'Fare': var_fare,
                'SP': var_sp,
                'Pclass': var_pclass,
                'Sex': var_sex,
                'Embarked': var_embarked,       
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase

titanic_raw = pd.read_csv('titanic_cleaned.csv',index_col=0)
titanic = titanic_raw.drop(columns=['Survived'])
df = pd.concat([input_df,titanic],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Sex','Pclass','Embarked']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = xgb.Booster()
load_clf.load_model("model.json")

load_clf2 = xgb.XGBRegressor()
load_clf2.load_model("model.json")

# Apply model to make predictions
dtest = xgb.DMatrix(df)
prediction = load_clf.predict(dtest)

prediction_proba = load_clf2.predict(df)


st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
