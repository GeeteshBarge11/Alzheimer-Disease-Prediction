import streamlit as st
import pandas as pd
import joblib

model = joblib.load("alzheimers_xgb.pkl")

st.title("Alzheimer’s Disease Detection")

# -------- INPUTS (ALL 32 FEATURES) --------

Age = st.number_input("Age", min_value=0)
BMI = st.number_input("BMI")
SystolicBP = st.number_input("Systolic BP")
DiastolicBP = st.number_input("Diastolic BP")
CholesterolTotal = st.number_input("Total Cholesterol")
CholesterolLDL = st.number_input("LDL Cholesterol")
CholesterolHDL = st.number_input("HDL Cholesterol")
CholesterolTriglycerides = st.number_input("Triglycerides")
MMSE = st.number_input("MMSE Score")

Gender = st.selectbox("Gender", [0, 1])  
Ethnicity = st.selectbox("Ethnicity", [0, 1, 2])
EducationLevel = st.selectbox("Education Level", [0, 1, 2])
Smoking = st.selectbox("Smoking", [0, 1])
AlcoholConsumption = st.selectbox("Alcohol Consumption", [0, 1])
PhysicalActivity = st.selectbox("Physical Activity", [0, 1])
DietQuality = st.selectbox("Diet Quality", [0, 1])
SleepQuality = st.selectbox("Sleep Quality", [0, 1])
FamilyHistoryAlzheimers = st.selectbox("Family History of Alzheimer’s", [0, 1])
CardiovascularDisease = st.selectbox("Cardiovascular Disease", [0, 1])
Diabetes = st.selectbox("Diabetes", [0, 1])
Depression = st.selectbox("Depression", [0, 1])
HeadInjury = st.selectbox("Head Injury", [0, 1])
Hypertension = st.selectbox("Hypertension", [0, 1])

FunctionalAssessment = st.selectbox("Functional Assessment", [0, 1])
MemoryComplaints = st.selectbox("Memory Complaints", [0, 1])
BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1])
ADL = st.selectbox("ADL Issues", [0, 1])
Confusion = st.selectbox("Confusion", [0, 1])
Disorientation = st.selectbox("Disorientation", [0, 1])
PersonalityChanges = st.selectbox("Personality Changes", [0, 1])
DifficultyCompletingTasks = st.selectbox("Difficulty Completing Tasks", [0, 1])
Forgetfulness = st.selectbox("Forgetfulness", [0, 1])

# -------- PREDICTION --------
if st.button("Predict"):
    input_df = pd.DataFrame({
        "Age": [Age],
        "Gender": [Gender],
        "Ethnicity": [Ethnicity],
        "EducationLevel": [EducationLevel],
        "BMI": [BMI],
        "Smoking": [Smoking],
        "AlcoholConsumption": [AlcoholConsumption],
        "PhysicalActivity": [PhysicalActivity],
        "DietQuality": [DietQuality],
        "SleepQuality": [SleepQuality],
        "FamilyHistoryAlzheimers": [FamilyHistoryAlzheimers],
        "CardiovascularDisease": [CardiovascularDisease],
        "Diabetes": [Diabetes],
        "Depression": [Depression],
        "HeadInjury": [HeadInjury],
        "Hypertension": [Hypertension],
        "SystolicBP": [SystolicBP],
        "DiastolicBP": [DiastolicBP],
        "CholesterolTotal": [CholesterolTotal],
        "CholesterolLDL": [CholesterolLDL],
        "CholesterolHDL": [CholesterolHDL],
        "CholesterolTriglycerides": [CholesterolTriglycerides],
        "MMSE": [MMSE],
        "FunctionalAssessment": [FunctionalAssessment],
        "MemoryComplaints": [MemoryComplaints],
        "BehavioralProblems": [BehavioralProblems],
        "ADL": [ADL],
        "Confusion": [Confusion],
        "Disorientation": [Disorientation],
        "PersonalityChanges": [PersonalityChanges],
        "DifficultyCompletingTasks": [DifficultyCompletingTasks],
        "Forgetfulness": [Forgetfulness]
    })

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Alzheimer’s Disease Detected")
    else:
        st.success("No Alzheimer’s Disease Detected")
