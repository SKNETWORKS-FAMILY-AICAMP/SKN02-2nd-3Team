import streamlit as st
import pandas as pd
import pickle

# Streamlit 앱 제목 설정
st.title('회원 이탈 확률 예측')

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 선택해주세요.", type="csv")


if uploaded_file is not None:
    # CSV 파일 로드
    df = pd.read_csv(uploaded_file)
    
    with open("./model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # 데이터 전처리 (필요한 경우)
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace('Credit Card','CC')
    df['PreferredPaymentMode'].replace('Cash on Delivery','COD', inplace=True)
    df['PreferredPaymentMode'].replace('Mobile Phone','Phone', inplace=True)
    df['PreferredPaymentMode'].replace('Credit Card','CC')

    
    # 이탈 확률 예측
    predictions = model.predict_proba(df)[:, 1]  # 클래스 1에 대한 확률을 선택
    
    # 예측 결과를 DataFrame에 추가
    df['Churn Probability (%)'] = predictions * 100
    df['Churn Risk'] = ['RED' if p > 0.5 else 'YELLOW' if p > 0.3 else 'GREEN' for p in predictions]
    
    st.write("Red 고객 비율: ", len(df[df['Churn Risk'] == 'RED']) / len(df)* 100, "%") 
    st.write("Yellow 고객 비율: ", len(df[df['Churn Risk'] == 'YELLOW']) / len(df)* 100 , "%")

    # 결과 표시
    st.write(df[['CustomerID', 'Churn Probability (%)','Churn Risk']])
else:
    st.write("CSV 파일을 업로드해주세요.")