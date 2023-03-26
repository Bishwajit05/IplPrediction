import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title='Breast Cancer Prediction', layout='wide')
st.title("Breast Cancer Prediction")
@st.cache_data
def cancerPredictionModel():
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
    data_frame['label'] = breast_cancer_dataset.target
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model


cancerModel = cancerPredictionModel()


col1,epty,col2,epty1,col3 = st.columns([1,0.1,1,0.1,1])

with col1:
    mean_radius = st.number_input("Mean Radius",step=1.,format="%.f")
   
    # step=1.,format="%.2f
with epty:
    st.write("")
with col2:
    texture_mean = st.number_input("Texture Mean",step=1.,format="%.f")
   
with epty1:
    st.write("")
with col3:
    perimeter_mean= st.number_input("Mean Perimeter",step=1.,format="%.f")
  

col4,ept2,col5,ept3,col6=st.columns([1,0.1,1,0.1,1])
with col4:
    area_mean = st.number_input("Mean Area",step=1.,format="%.f")
    
with ept2:
    st.write("")
with col5:
    smoothness_mean = st.number_input("Mean Smoothness",step=1.,format="%.f")
   
with ept3:
    st.write("")
with col6:
    compactness_mean= st.number_input("Mean Compactness",step=1.,format="%.f")
 
    
col7,ept4,col8,ept5,col9=st.columns([1,0.1,1,0.1,1])
with col7:
    concavity_mean = st.number_input("Mean Concativity",step=1.,format="%.f")
  
with ept4:
    st.write("")
with col8:
    concave_points_mean = st.number_input("Mean concave points",step=1.,format="%.f")
    
with ept5:
    st.write("")
with col9:
    symmetry_mean= st.number_input("Symmetry mean",step=1.,format="%.f")
   
col10,ept6,col11,ept7,col12=st.columns([1,0.1,1,0.1,1])
with col10:
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean",step=1.,format="%.f")

with ept6:
    st.write("")
with col11:
    radious_se = st.number_input("Radius Se",step=1.,format="%.f")
    
with ept7:
    st.write("")
with col12:
    texture_se= st.number_input("Texture Se",step=1.,format="%.f")
   

col13,ept8,col14,ept9,col15=st.columns([1,0.1,1,0.1,1])
with col13:
    perimeter_se = st.number_input("Perimeter Se",step=1.,format="%.f")

with ept8:
    st.write("")
with col14:
    area_se = st.number_input("Area Se",step=1.,format="%.f")
  
with ept9:
    st.write("")
with col15:
    somoothness_Se= st.number_input("Smoothness Se",step=1.,format="%.f")

col16,ept10,col17,ept11,col18=st.columns([1,0.1,1,0.1,1])
with col16:
    compactness_se = st.number_input("Compactness Se",step=1.,format="%.f")
with ept10:
    st.write("")
with col17:
    concavity_se = st.number_input("Concativity Se",step=1.,format="%.f")
with ept11:
    st.write("")
with col18:
    concave_points_se= st.number_input("Concave points Se",step=1.,format="%.f")


col19,ept12,col20,ept13,col21=st.columns([1,0.1,1,0.1,1])
with col19:
    symmetry_se = st.number_input("Symmetry Se",step=1.,format="%.f")
with ept12:
    st.write("")
with col20:
    fractal_dimension_se = st.number_input("Fractal Dimension Se",step=1.,format="%.f")
with ept13:
    st.write("")
with col21:
    radius_worst= st.number_input("Worst Radius",step=1.,format="%.f")

col22,ept14,col23,ept15,col24=st.columns([1,0.1,1,0.1,1])
with col22:
    texture_worst = st.number_input("Worst Texture",step=1.,format="%.f")
with ept14:
    st.write("")
with col23:
    perimeter_worst = st.number_input("Worst Perimeter",step=1.,format="%.f")
with ept15:
    st.write("")
with col24:
    area_worst= st.number_input("Worst Area",step=1.,format="%.f")

col25,ept16,col26,ept17,col27=st.columns([1,0.1,1,0.1,1])
with col25:
    texturesmoothness_worst_worst =st.number_input("Worst Texture Smoothness",step=1.,format="%.f")
with ept16:
    st.write("")
with col26:
    compactness_worst = st.number_input("Worst Compactness",step=1.,format="%.f")
with ept17:
    st.write("")
with col27:
    concavity_worst= st.number_input("Worst Concavity",step=1.,format="%.f")

col28,ept18,col29,ept19,col30=st.columns([1,0.1,1,0.1,1])
with col28:
    concave_worst =st.number_input("Worst Concave",step=1.,format="%.f")
with ept18:
    st.write("")
with col29:
    symmetry_worst = st.number_input("Worst Symmetry",step=1.,format="%.f")
with ept19:
    st.write("")
with col30:
    fractal_dimension_worst= st.number_input("Worst fractal dimension",step=1.,format="%.f")


if st.button('Click me!'):
    input_data = (mean_radius,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radious_se,texture_se,perimeter_se,area_se,somoothness_Se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,texturesmoothness_worst_worst,compactness_worst,concavity_worst,concave_worst,symmetry_worst,fractal_dimension_worst)
    input_data_as_numpy_array = np.asarray(input_data)


    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    pred=cancerModel.predict(input_data_reshaped)
    if (pred[0] == 0):
        st.write('The Breast cancer is Malignant')

    else:
        st.write('The Breast Cancer is Benign')
        # st.write(predictionIpl.predict_proba(fdf)*100)


# id, diagnosis, radious_mean, texture_mean ,perimeter_mean,area_mean, 
# smoothness_mean ,compactness_mean,concavity_mean concave_points_mean, 
# symmetry_mean, fractal_dimension_mean,radious_se,texture_se,perimeter_se,area_se,
# somoothness_Se,compactness_se,concavity_se,concave_points_se, symmetry_se, 
# fractal_dimension_se, radius_worst, texture_worst,perimeter_worst,area_worst,
# smoothness_worst,compactness_worst,concavity_wprst,concave_worst,symmetry_worst,
# fractal_dimesnsion_worst




