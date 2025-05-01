
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"{ticker} 抓不到資料，請確認代碼或日期")
    df.dropna(inplace=True)
    return df

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ['Return', 'MA5', 'MA10']
    X = df[features]
    y = df['Label']
    if len(X) == 0:
        raise ValueError("資料筆數不足，無法建立模型")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict_next_day(df, model):
    latest = df.iloc[-1][['Return', 'MA5', 'MA10']].values.reshape(1, -1)
    prob = model.predict_proba(latest)[0][1]
    return prob

st.set_page_config(page_title="台股預測", page_icon="💰")
st.title("台股漲跌預測工具")

ticker = st.text_input("輸入台股代碼（加上 .TW，例如：2330.TW）", value="2330.TW")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("結束日期", pd.to_datetime("2024-12-31"))

if st.button("開始預測"):
    try:
        df = get_stock_data(ticker, str(start_date), str(end_date))
        st.success(f"成功取得資料，共 {len(df)} 筆")

        df = add_features(df)
        model = train_model(df)
        prob = predict_next_day(df, model)

        st.subheader("預測結果")
        st.write(f"**明天上漲的機率為：{prob:.2%}**")

        fig, ax = plt.subplots()
        df['Close'].plot(ax=ax, title=f"{ticker} 收盤價")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"錯誤：{str(e)}")
