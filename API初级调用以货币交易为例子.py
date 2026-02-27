import streamlit as st
import pandas as pd
import requests
import random
from datetime import datetime
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh


#将渐变定义为函数，若换成其他的子标题对应改变h1为h6，颜色对应改变为对应的16进制
import streamlit as st

def gradient_title(text, color1="#ff8a00", color2="#da1b60", angle=90, font_size="3rem", tag="h1", align="center"):
    """
    生成渐变标题（支持居中）
    :param text: 标题文字内容
    :param color1: 起始颜色
    :param color2: 结束颜色
    :param angle: 渐变角度
    :param font_size: 字体大小
    :param tag: HTML标签
    :param align: 文本对齐方式，可选 'left', 'center', 'right'
    """
    html = f"""
    <{tag} style="
        background: linear-gradient({angle}deg, {color1}, {color2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: {font_size};
        font-weight: bold;
        text-align: {align};
        margin: 0;
    ">
        {text}
    </{tag}>
    """
    st.markdown(html, unsafe_allow_html=True)




# ---------- 页面配置 ----------
#即网页在浏览器中的显示名字以及图标
st.set_page_config(
    page_title="实时加密货币看板",
    page_icon="📈",
   layout="wide", 

)

#自动刷新机制间隔单位为毫秒
st_autorefresh(interval=60 * 1000, key="auto_refresh")

#Streamlit的缓存装饰器。
@st.cache_data(ttl=5)

#数据获取函数（带 fallback 模拟数据）

def fetch_crypto_prices():
    """
    从 CoinGecko 获取比特币和以太坊实时价格（免费，无密钥）
    若失败则返回模拟数据
    """
    #捕获异常情况函数，若执行异常将返回exception中的内容
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum",
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'bitcoin': {
                    'price': data['bitcoin']['usd'],
                    'change24h': data['bitcoin']['usd_24h_change'],
                    'volume': data['bitcoin']['usd_24h_vol'],
                    'name': 'Bitcoin'
                },
                'ethereum': {
                    'price': data['ethereum']['usd'],
                    'change24h': data['ethereum']['usd_24h_change'],
                    'volume': data['ethereum']['usd_24h_vol'],
                    'name': 'Ethereum'
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'source': 'CoinGecko'
            }
        else:
            # 状态码不是200，也返回模拟数据
            st.warning(f"API返回状态码: {response.status_code}，使用模拟数据")
            return get_mock_data()
    except Exception as e:
        st.warning(f"网络连接异常，使用模拟数据: {e}")
        return get_mock_data()   # 关键：必须返回模拟数据

#随机数据生成略
def get_mock_data():
    """生成模拟的实时数据，使页面看起来仍在更新"""
    # 用随机数模拟价格波动
    btc_price = 50000 + random.uniform(-2000, 2000)
    eth_price = 3000 + random.uniform(-150, 150)
    return {
        'bitcoin': {
            'price': round(btc_price, 2),
            'change24h': round(random.uniform(-5, 5), 2),
            'volume': round(25e9 + random.uniform(-2e9, 2e9), 0),
            'name': 'Bitcoin (模拟)'
        },
        'ethereum': {
            'price': round(eth_price, 2),
            'change24h': round(random.uniform(-5, 5), 2),
            'volume': round(15e9 + random.uniform(-1e9, 1e9), 0),
            'name': 'Ethereum (模拟)'
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'source': '模拟数据'
    }

@st.cache_data(ttl=60)
def fetch_historical_data(coin_id="bitcoin", days=1):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            # 先去掉 interval 参数测试是否必需
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['time', 'price'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            return df
        else:
            st.warning(f"API 返回状态码 {response.status_code}: {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"请求异常: {e}")
        return pd.DataFrame()

# ---------- 获取最新数据 ----------
data = fetch_crypto_prices()

# ---------- 页面标题 ----------
gradient_title("加密货币看板", color1="#294cad", color2="#2980e4", angle=135, font_size="3.5rem", tag="h1", align="center")
st.caption(f"最后更新时间: {data['timestamp']} | 数据来源: {data['source']}")


gradient_title("📋当前货币价格", color1="#294cad", color2="#2980e4", angle=135, font_size="1.5rem", tag="h2", align="left")
st.divider()

# ---------- 核心指标卡片 ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label=f"💰 {data['bitcoin']['name']} (BTC)",
        value=f"${data['bitcoin']['price']:,.2f}",
        delta=f"{data['bitcoin']['change24h']:.2f}%"
    )

with col2:
    st.metric(
        label=f"💰 {data['ethereum']['name']} (ETH)",
        value=f"${data['ethereum']['price']:,.2f}",
        delta=f"{data['ethereum']['change24h']:.2f}%"
    )

with col3:
    btc_vol_billions = data['bitcoin']['volume'] / 1e9
    st.metric(
        label="📊 BTC 24h交易量",
        value=f"${btc_vol_billions:.2f}B"
    )

with col4:
    eth_vol_billions = data['ethereum']['volume'] / 1e9
    st.metric(
        label="📊 ETH 24h交易量",
        value=f"${eth_vol_billions:.2f}B"
    )



# ---------- 走势图 ----------
gradient_title("📋价格走势", color1="#294cad", color2="#2980e4", angle=135, font_size="1.5rem", tag="h2", align="left")

tab1, tab2 = st.tabs(["比特币 BTC", "以太坊 ETH"])

with tab1:
    btc_history = fetch_historical_data("bitcoin", days=1)
    if not btc_history.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=btc_history['time'],
            y=btc_history['price'],
            mode='lines+markers',
            name='BTC价格',
            line=dict(color="#226ec5", width=2)
        ))
        fig.update_layout(
            title="比特币价格走势（每小时）",
            xaxis_title="时间",
            yaxis_title="价格 (USD)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("历史数据暂不可用，请稍后重试")

with tab2:
    eth_history = fetch_historical_data("ethereum", days=1)
    if not eth_history.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eth_history['time'],
            y=eth_history['price'],
            mode='lines+markers',
            name='ETH价格',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="以太坊价格走势（每小时）",
            xaxis_title="时间",
            yaxis_title="价格 (USD)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("历史数据暂不可用，请稍后重试")



# ---------- 数据明细表格 ----------

gradient_title("📋实时数据明细",color1="#294cad", color2="#2980e4", angle=135, font_size="1.5rem", tag="h2", align="left")
st.divider()
table_data = {
    "指标": ["价格 (USD)", "24h涨跌幅", "24h交易量 (USD)", "数据源"],
    "比特币 (BTC)": [
        f"${data['bitcoin']['price']:,.2f}",
        f"{data['bitcoin']['change24h']:.2f}%",
        f"${data['bitcoin']['volume']:,.0f}",
        data['source']
    ],
    "以太坊 (ETH)": [
        f"${data['ethereum']['price']:,.2f}",
        f"{data['ethereum']['change24h']:.2f}%",
        f"${data['ethereum']['volume']:,.0f}",
        data['source']
    ]
}

df_display = pd.DataFrame(table_data)
st.dataframe(df_display, use_container_width=True, hide_index=True)

st.divider()
st.caption("🚀 数据每10秒自动刷新 | 使用 CoinGecko 免费API，网络异常时自动切换为模拟数据")