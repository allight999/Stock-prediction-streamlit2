import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
#วันที่ที่ใช้เป็นตัวอ้างอิง
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('แอพคาดการณ์หุ้น')
#เลือกหุ้นที่ต้องการ
stocks = ('GOOG', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'TSLA', 'NFLX' , 'AMD' , 'META' , 'TTB.BK' , 'PTT.BK' , 'ADVANC.BK' , 'CPALL.BK' , 'GULF.BK' , 'TRUE.BK')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
#จำนวนปีที่ต้องการพยากรณ์
n_years = st.slider('Years of prediction:', 1, 8)
period = n_years * 365

#โหลดข้อมูล
@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

#สร้างกราฟจากข้อมูลดิบ
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

#คาดการณ์ด้วย propeht 
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# พลอตผลออกมาเป็นกราฟ
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
