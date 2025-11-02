import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
import base64
from tensorflow.keras.models import load_model
#Function to create a download link for a DataFrame as a CSV file
def get_binary_file_downloader_html(df):
    csv=df.to_csv(index=False)
    b64=base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href
st.title(" 🫀 Heart Disease Predictor 🫀")
tab1,tab2,tab3=st.tabs(['Predict','Bulk Predict','Model Information'])
with tab1:
    age=st.number_input("Age (years)",min_value=0,max_value=150)
    sex=st.selectbox("Sex",["Female","Male"])
    chest_pain=st.selectbox("Chest pain type",["Atypical angina","Non-Anginal Pain","Asymptomatic","Typical Angina"])
    resting_bp=st.number_input("Resting Blood Pressure (mm/dl)",min_value=0)
    cholesterol=st.number_input("Serum Cholesterol (mm/dl)",min_value=0)
    fasting_bs=st.selectbox("Fasting Blood Sugar",["<=120 mg/dl","> 120 mg/dl"])
    resting_ecg=st.selectbox("Resting ECG Results",["Normal","ST-T wave abnormality","left ventricular hypertrophy"])
    max_hr=st.number_input("Maximum Heart Rate Achived",min_value=60,max_value=202)
    exercise_angina=st.selectbox("Exercise-Induced Angina",["Yes","No"])
    oldpeak=st.number_input("Oldpeak (ST Depression)",min_value=0.0,max_value=10.0)
    st_slope=st.selectbox("Slope pf peak Exercise ST Segment",["Upsloping","Flat","Downsloping"])
    #convert categorcal inputs to numeric 
    sex=0 if sex== "Male" else 1
    chest_pain=["Atypical angina","Non-Anginal Pain","Asymptomatic","Typical Angina"].index(chest_pain)
    fasting_bs=1 if fasting_bs==">120 mg/dl" else 0
    resting_ecg=["Normal","ST-T wave abnormality","left ventricular hypertrophy"].index(resting_ecg)
    exercise_angina=1 if exercise_angina== "Yes" else 0
    st_slope=["Upsloping","Flat","Downsloping"].index(st_slope)

    #create a DataFrame with user inputs
    input_data=pd.DataFrame({
        'Age':[age],
        'Sex':[sex],
        'ChestPainType':[chest_pain],
        'RestingBP':[resting_bp],
        'Cholesterol':[cholesterol],
        'FastingBS':[fasting_bs],
        'RestingECG':[resting_ecg],
        'MaxHR':[max_hr],
        'ExerciseAngina':[exercise_angina],
        'Oldpeak':[oldpeak],
        'ST_Slope':[st_slope]
    })
    algonames=['Decision Trees','Logistic Regression','Random Forest','Support Vector Machine','neural network']
    modelnames=['dtree.pkl','LogisticR.pkl','rfc.pkl','svm.pkl']

    predictions=[]
    def predict_heart_disease(data):
        for modelname in modelnames:
            model=pickle.load(open(modelname,'rb'))
            prediction=model.predict(data)
            predictions.append(prediction)
        nn_model = load_model('Nn_model.h5')
        prediction = (nn_model.predict(input_data) > 0.5).astype(int)
        predictions.append(prediction)
        return predictions
    #create a submit button to make predictions
    if st.button("Submit"):
        st.subheader('Results...')
        st.markdown('------------------')

        result=predict_heart_disease(input_data)
        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0]==0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('-----------')
with tab2:
    st.title("Upload CSV File")
    st.subheader('Instructions to note before uploading the file : ')
    st.info("""
    1. No NaN values are allowed.
    2. Total 11 features in this order ('Age','Sex','chestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Aldpeak','ST_Slope').\n
    3. Check the spellings of the features names.
    4. Feature Values conventions: \n
        - Age:age of the patient [years] \n
        - Sex:sex of the patient [0:Male,1:Female]\n
        - chestPainType: chest pain type [3:Typical Angina,0:Atypical angina,1:Non_Anginal Pain,2:Asymptomatic]\n
        - RestingBp:resting blood pressure [mm Hg]\n
        - Cholesterol: serum cholesterol [mm/dl]\n
        - FastingBD: fasting blood sugar [1:if FastingBS>120 mg/dl,0:otherwise]\n
        - RestingECG: resting electrocardiogram results [0: Normal,1: having ST_T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]\n
        - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]  \n
        - ExerciseAngina: exercise-induced angina [1: Yes, 0: No]\n
        - Oldpeak: oldpeak = ST [Numeric value measured in depression]\n
        - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping]
        """)
    
    #create a file uploader in the sidebar
    uploaded_file=st.file_uploader("Upload a CSV file",type=["csv"])
    if uploaded_file is not None:
        #Read the uploaded CSV file into a Dataframe
        input_data=pd.read_csv(uploaded_file)
        model1=pickle.load(open('LogisticR.pkl','rb'))
        model2=pickle.load(open('svm.pkl','rb'))
        model3=pickle.load(open('dtree.pkl','rb'))
        model4=pickle.load(open('rfc.pkl','rb'))
        nn_model = load_model('nn_model.h5')
        #Ensure that the input DataFrame matches the expected columns and format
        expected_columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
        if set(expected_columns).issubset(input_data.columns) :
            input_data['prediction_nn']=''
            input_data['Prediction_LR']=''
            input_data['Prediction_svm']=''
            input_data['Prediction_rfc']=''
            input_data['Prediction_dtree']=''
            for i in range(len(input_data)):
                arr = input_data.loc[i, expected_columns].values.astype(float)
                arr = arr.reshape(1, -1)
                input_data['Prediction_LR'][i] = model1.predict(arr)[0]
                input_data['Prediction_svm'][i] = model2.predict(arr)[0]
                input_data['Prediction_dtree'][i] = model3.predict(arr)[0]
                input_data['Prediction_rfc'][i] = model4.predict(arr)[0]
                input_data['prediction_nn'][i] = int((nn_model.predict(arr)[0][0]) > 0.5)
            input_data.to_csv('PredictedHeart.csv')
            #Display the predictions
            st.subheader("Predictions:")
            st.write(input_data)
            #create a button to download the updated CSV file
            st.markdown(get_binary_file_downloader_html(input_data),unsafe_allow_html=True)
        else:
            st.warning("Please make sure the uploaded CSV file has the correct columns.")
    else:
        st.warning("Upload a CSV to get predictions.")
with tab3:
    import plotly.express as px

    data = {
    'Decision Trees': 80.97,
    'Logistic Regression': 85.86,
    'Random Forest': 84.23,
    'Support Vector Machine': 84.22,
    'Neural Network': 85.19
    }

    # Créer un DataFrame
    df = pd.DataFrame(list(data.items()), columns=['Models', 'Accuracies'])

    # Trier par Accuracy décroissante
    df = df.sort_values(by='Accuracies', ascending=False)

    # Créer le graphique
    fig = px.bar(df, x='Models', y='Accuracies', text='Accuracies')

    # Afficher dans Streamlit
    st.plotly_chart(fig)
    st.markdown("### Model Comparison")
    df_sorted = df.sort_values(by='Accuracies', ascending=False).reset_index(drop=True)
    best_model = df_sorted.iloc[0]
    second_model = df_sorted.iloc[1]
    st.info(
    f"The **{best_model['Models']}** performs the best with an accuracy of {best_model['Accuracies']}%, "
    f"followed closely by **{second_model['Models']}** at {second_model['Accuracies']}%. "
    "Other models have slightly lower performance, but still provide valuable predictions.\n\n"
    "**Reason:** Logistic Regression is particularly suitable here because the features are well-behaved, "
    "linearly correlated with the outcome, and the dataset is not too large, reducing the risk of overfitting."
)

