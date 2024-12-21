import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from PIL import Image
import sys
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_model():
    try:
        try:
            import sklearn_rvm
        except ImportError:
            st.error("Required package 'sklearn_rvm' not found. Please install it using: pip install sklearn-rvm")
            return None, None
            
        with open('model/rvm_rbf_87.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('model/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        st.info("Please ensure that model files are in the correct directory")
        return None, None

def preprocess_input(input_data, scaler):
    columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
               'fasting blood sugar', 'resting ecg', 'max heart rate',
               'exercise angina', 'oldpeak', 'ST slope']
    df = pd.DataFrame(input_data, columns=columns)
    
    columns_to_scale = ['age', 'resting bp s', 'cholesterol', 'max heart rate']
    df[columns_to_scale] = scaler.transform(df[columns_to_scale].values)
    
    return df.values

def main():
    st.set_page_config(
        page_title="Heart Disease Predictor",
        page_icon="❤️",
        layout="wide"
    )

    st.title("❤️ Heart Disease Prediction System")
    st.markdown("---")

    st.sidebar.title("Navigation Panel")
    pages = ["Prediction", "About the Dataset"]
    choice = st.sidebar.radio("Pages", pages)

    if choice == "Prediction":
        show_prediction_page()
    else:
        show_about_page()

def show_prediction_page():
    st.header("Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
                         options=["Typical Angina", 
                                 "Atypical Angina",
                                 "Non-anginal Pain",
                                 "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                  min_value=90, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", 
                              min_value=100, max_value=600, value=200)
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                          options=["Yes", "No"])
        restecg = st.selectbox("Resting ECG Results",
                              options=["Normal",
                                      "ST-T Wave Abnormality",
                                      "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate", 
                                 min_value=50, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", 
                            options=["Yes", "No"])
        oldpeak = st.number_input("ST Depression", 
                                 min_value=0.0, max_value=6.0, value=0.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment",
                            options=["Upsloping", "Flat", "Downsloping"])

    sex = 1 if sex == "Male" else 0
    cp = {"Typical Angina": 1, 
          "Atypical Angina": 2,
          "Non-anginal Pain": 3,
          "Asymptomatic": 4}[cp]
    fbs = 1 if fbs == "Yes" else 0
    restecg = {"Normal": 0,
               "ST-T Wave Abnormality": 1,
               "Left Ventricular Hypertrophy": 2}[restecg]
    exang = 1 if exang == "Yes" else 0
    slope = {"Upsloping": 1,
             "Flat": 2,
             "Downsloping": 3}[slope]

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope]])

    if st.button("Predict"):
        model, scaler = load_model()
        
        if model is not None and scaler is not None:
            try:
                scaled_input = preprocess_input(input_data, scaler)
                
                prediction = model.predict(scaled_input)
                probability = model.predict_proba(scaled_input)
                
                st.toast('Prediction Successful', icon='✅')
                st.success("Prediction Successful")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction[0] == 1:
                        st.error("⚠️ High Risk of Heart Disease")
                    else:
                        st.success("✅ Low Risk of Heart Disease")
                        
                with col2:
                    st.info(f"Probability of Heart Disease: {probability[0][1]:.2%}")
                
                # color legend, red is high risk, green is low risk
                st.markdown("---")
                st.subheader("Color Legend")
                st.markdown("""
                - 🔴 High Risk of Heart Disease
                - 🟢 Low Risk of Heart Disease
                """)
                    
                if prediction[0] == 1:
                    fig = px.pie(values=[probability[0][1], 1-probability[0][1]], 
                            names=['Risk', 'Safe'],
                            hole=0.7,
                            color_discrete_sequence=['#f8312f', '#00d26a'])
                    fig.update_layout(showlegend=False, 
                                    height=300,
                                    annotations=[dict(text=f"{probability[0][1]:.1%}", 
                                                    x=0.5, y=0.5, 
                                                    font_size=40, 
                                                    showarrow=False)])
                else:
                    fig = px.pie(values=[probability[0][1], 1-probability[0][1]], 
                            names=['Risk', 'Safe'],
                            hole=0.7,
                            color_discrete_sequence=['#00d26a', '#f8312f'])
                    fig.update_layout(showlegend=False, 
                                    height=300,
                                    annotations=[dict(text=f"{probability[0][1]:.1%}", 
                                                    x=0.5, y=0.5, 
                                                    font_size=40, 
                                                    showarrow=False)])
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check if the input data format matches the model's requirements")

def show_about_page():
    st.header("About the Dataset")
    st.write("""
    This heart disease prediction model is trained on a comprehensive dataset that combines 
    5 popular heart disease datasets:
    - Cleveland
    - Hungarian
    - Switzerland
    - Long Beach VA
    - Statlog (Heart) Data Set
    
    The combined dataset consists of 1190 instances with 11 features, making it one of the 
    largest heart disease datasets available for research purposes.
    """)
    
    st.subheader("Feature Descriptions")
    feature_desc = pd.DataFrame({
        'Feature': [
            'Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure',
            'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG',
            'Maximum Heart Rate', 'Exercise Induced Angina',
            'ST Depression', 'Slope of ST Segment'
        ],
        'Description': [
            'Age of the patient',
            'Male = 1, Female = 0',
            '1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic',
            'Resting blood pressure in mm Hg',
            'Serum cholesterol in mg/dl',
            'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
            'Resting electrocardiogram results',
            'Maximum heart rate achieved',
            'Exercise induced angina (1 = yes, 0 = no)',
            'ST depression induced by exercise relative to rest',
            'Slope of the peak exercise ST segment'
        ]
    })
    st.dataframe(feature_desc, hide_index=True)

if __name__ == "__main__":
    main()