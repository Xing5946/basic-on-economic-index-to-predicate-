"""
宏观经济预测系统 - Streamlit前端界面
提供交互式数据可视化和预测结果展示
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import base64
# 页面配置
st.set_page_config(
    page_title="宏观经济核心指标预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
    return html

# API配置
API_BASE_URL = "http://localhost:8000"

# ============ 样式配置 ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
 </style>
""", unsafe_allow_html=True)         

st.markdown("""
<style>
    .prediction-highlight {
     background: linear-gradient(135deg, var(--grad-start, #667eea) 0%, var(--grad-end, #764ba2) 100%);
     color: var(--text-color, white);  /* 默认白色 */
     padding:1.8rem;
     border-radius: 1.2rem;
     text-align: center;
    }
    .prediction-highlight:hover {
     box-shadow: var(--hover-shadow, 0 8px 16px rgba(0, 0, 0, 0.1));
     }

    .prediction-highlight h1 {
      color: var(--title1-color, inherit);
     }
    .prediction-highlight h2 {
      color: var(--title2-color, inherit);
      }
    .prediction-highlight h3 {
      color: var(--title3-color, inherit);
     }
    .prediction-highlight strong {
     color: var(--rate-color, inherit);
     }
 </style>
""", unsafe_allow_html=True)  
 
st.markdown("""
<style>   
    .transparent-card {
     background: transparent;
     border: var(--card-border, 1px solid #e0e0e0);
     border-radius: 1rem;
     padding: 1.5rem;
     text-align: center;
     transition: box-shadow 0.2s ease;
  }
    .transparent-card:hover {
     box-shadow: var(--card-hover-shadow, 0 8px 16px rgba(0, 0, 0, 0.1));
     }
</style>
""", unsafe_allow_html=True)





# ============ 辅助函数 ============




def fetch_api(endpoint: str, params: dict = None):
    """调用API"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if params:
            response = requests.get(url, params=params, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return None


def create_gdp_chart(df: pd.DataFrame):
    """创建GDP趋势图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['gdp_value'],
        mode='lines+markers',
        name='GDP绝对值',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='GDP历史趋势',
        xaxis_title='日期',
        yaxis_title='GDP（亿元）',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_indicator_chart(df: pd.DataFrame, indicator: str):
    """创建指标趋势图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[indicator],
        mode='lines',
        name=indicator,
        line=dict(width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title=f'{indicator} 趋势',
        xaxis_title='日期',
        yaxis_title='数值',
        template='plotly_white'
    )
    
    return fig


def create_factor_chart(factors_df: pd.DataFrame):
    """创建因子分析图"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Factor 1', 'Factor 2', 'Factor 3', 'All Factors'),
        specs=[[{}, {}], [{}, {}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, col in enumerate(['Factor_1', 'Factor_2', 'Factor_3']):
        if col in factors_df.columns:
            row = i // 2 + 1
            col_idx = i % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=factors_df.index,
                    y=factors_df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i], width=2)
                ),
                row=row, col=col_idx
            )
    
    # 所有因子对比
    for i, col in enumerate(['Factor_1', 'Factor_2', 'Factor_3']):
        if col in factors_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=factors_df.index,
                    y=factors_df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i], width=2)
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_midas_weights_chart(weights: list):
    """创建MIDAS权重图"""
    lags = list(range(1, len(weights) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=lags,
        y=weights,
        marker_color='#1f77b4',
        name='滞后权重'
    ))
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=weights,
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2),
        name='权重趋势'
    ))
    
    fig.update_layout(
        title='MIDAS模型滞后权重分布',
        xaxis_title='滞后月份',
        yaxis_title='权重',
        template='plotly_white'
    )
    
    return fig


def create_prediction_gauge(value: float, title: str, min_val: float, max_val: float):
    """创建预测仪表盘"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        delta={'reference': (min_val + max_val) / 2},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [min_val, min_val + (max_val - min_val) * 0.3], 'color': "lightgray"},
                {'range': [min_val + (max_val - min_val) * 0.3, min_val + (max_val - min_val) * 0.7], 'color': "gray"},
                {'range': [min_val + (max_val - min_val) * 0.7, max_val], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': (min_val + max_val) / 2
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


# ============ 页面组件 ============

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("导航菜单")
        
        page = st.radio(
            "选择页面",
            ["🏠 概览驾驶舱", "📈 现时预测", "📊 历史数据", "🔍 因子分析", "⚙️ 模型管理"]
        )
        
        st.divider()
        
        # 系统状态
        st.subheader("系统状态")
        
        # 模拟状态显示
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MIDAS", "✅")
        with col2:
            st.metric("DFM", "✅")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("TSLM", "✅")
        with col4:
            st.metric("API", "✅")
        
        st.divider()
        
        # 快速操作
        st.subheader("快速操作")
        if st.button("🔄 刷新数据", use_container_width=True):
            st.rerun()
        if st.button("🔮 立即预测", use_container_width=True):
            st.session_state['trigger_prediction'] = True
            st.rerun()
            
        return page

def get_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("test/刘_初步集合/frontend/overlay_bg.png")

def render_overview():
    """渲染概览驾驶舱"""
    title_text = gradient_title(" 宏观经济核心指标预测系统",color1="#D19C8B",color2="#8d346b", angle=135, font_size="3rem", tag="h1", align="center")
    
    subtitle_text = gradient_title("基于 MIDAS + DFM + TSLM 的混合预测模型",color1="#FCFCFC",color2="#d3dae2", angle=135, font_size="1.3rem", tag="h2", align="right")
    st.markdown(
    f"""
    <style>
      .dynamic-title {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-position: center;
        background-size: cover;
        padding: 60px 20px;
        border-radius: 10px;
        width: 100%;              /* 容器宽度占满父元素 */
        height: 300px;   
      }}
    </style>
    <div class="dynamic-title">
        {title_text}
        {subtitle_text}
    </div>
    """,
    unsafe_allow_html=True,
 )
  
   
    # 关键指标卡片
    st.markdown(gradient_title("📌 关键指标概览",color1="#D19C8B",color2="#8d346b", angle=135, font_size="2rem", tag="h2", align="left"), unsafe_allow_html=True)
               

    st.markdown(gradient_title(f"数据更新时间:{datetime.now().strftime("%m-%d %H:%M")}",color1="#F0F3FA",color2="#e5e9ec", angle=135, font_size="1.3rem", tag="h2", align="right"), unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    with cols[0]:
     st.markdown("""
      <div class="prediction-highlight" 
            style="--grad-start:#D19C8B ; 
                   --grad-end: #8d346b ; 
                   --text-color: #333; --title3-color: #fff;
                   --rate-color: #00FF00; 
                   --hover-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);">
       <h3 style="text-align:center; font-size:1.2rem; margin:0; color:#FFFFFF;"> 最新GDP(当季)
       </h3>
       <h1 style="text-align:center; font-size:1.8rem; margin:0; line-height:10px; color:#FFFFFF;">
        ¥12,580亿
       </h1>
       <p style="text-align:center; color:#5FBB5F; line-height:15px;"><strong>⬆+5.8%</strong>
      </p>
      </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
      <div class="prediction-highlight" 
            style="--grad-start:#D19C8B ; 
                   --grad-end: #8d346b ; 
                   --text-color: #333; --title3-color: #fff;
                   --rate-color: #00FF00; 
                   --hover-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);">
      <h3 style= "text-align:center; font-size: 1.2rem;margin:0; color:#FFFFFF;">现实预测(下季)</h3>
      <h1 style= "text-align:center; font-size: 1.8rem; margin:0;line-height:10px; color:#FFFFFF;">¥13,020亿</h1>
      <p style = "text-align:center;color:#5FBB5F; line-height:15px;"> <strong>⬆+3.5%</strong></p>
            
      </div>
      """, unsafe_allow_html=True) 
      
    
    with cols[2]:
        st.markdown("""
      <div class="prediction-highlight" 
            style="--grad-start:#D19C8B ; 
                   --grad-end: #8d346b ; 
                   --text-color: #333; --title3-color: #fff;
                   --rate-color: #00FF00; 
                   --hover-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);">
      <h3 style= "text-align:center; font-size: 1.2rem;margin:0; color:#FFFFFF;">预测准确率</h3>
      <h1 style= "text-align:center; font-size: 1.8rem; margin: 0;line-height:10px; color:#FFFFFF;">92.5%</h1>
      <p style = "text-align:center;color:#5FBB5F; line-height:15px;"> <strong>⬆+1.2%</strong></p>
            
      </div>
      """, unsafe_allow_html=True) 
      
    
    st.divider()
    
    # 主要图表区域
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 GDP历史趋势与预测")
        
        # 模拟GDP数据
        dates = pd.date_range(start='2015-01-01', periods=40, freq='QE')
        gdp_values = 8000 * (1.035 ** np.arange(40)) + np.random.normal(0, 100, 40)
        
        gdp_df = pd.DataFrame({
            'date': dates,
            'gdp_value': gdp_values,
            'type': ['历史'] * 38 + ['预测'] * 2
        })
        
        fig = go.Figure()
        
        # 历史数据
        fig.add_trace(go.Scatter(
         x=gdp_df[gdp_df['type'] == '历史']['date'],
         y=gdp_df[gdp_df['type'] == '历史']['gdp_value'],
         mode='lines+markers',
         name='历史数据',
         line=dict(color='#1f77b4', width=2),
         fill='tozeroy',                     # 填充到 y=0
         fillcolor='rgba(31,119,180,0.2)'    # 半透明蓝色（与线同色）
         ))
        # 预测数据
        
        fig.add_trace(go.Scatter(
         x=gdp_df[gdp_df['type'] == '预测']['date'],
         y=gdp_df[gdp_df['type'] == '预测']['gdp_value'],
         mode='lines+markers',
         name='预测值',
         line=dict(color='#ff7f0e', width=2, dash='dash'),
         fill='tozeroy',                     # 填充到 y=0
         fillcolor='rgba(255,127,14,0.2)'    # 半透明橙色
             ))
        
        fig.update_layout(
            xaxis_title='日期',
            yaxis_title='GDP（亿元）',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height = 800
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🔮 最新现时预测")
        
        

        st.markdown(
      """
      <div style="text-align: center;">
        <div style="font-size: 0.9rem;">2024Q4 GDP预测</div>
        <div style="font-size: 2rem; font-weight: bold;">¥13,020亿</div>
        <div style="color: green;">+5.8%</div>
       </div>
      """,
      unsafe_allow_html=True,
         ) 
        
        st.divider()
        
        st.subheader("📊 模型贡献度")
        
        contribution_data = pd.DataFrame({
            '模型': ['MIDAS线性', 'TSLM非线性修正'],
            '贡献度': [60, 40]
        })
        
        fig_pie = px.pie(
            contribution_data,
            values='贡献度',
            names='模型',
            color_discrete_sequence=['#1f77b4', '#ff7f0e'],
            height= 400
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()
    
    # 高频指标监控
    st.subheader("📡 高频指标实时监控")
    
    indicator_cols = st.columns(4)
    
    indicators = [
        ("全社会用电量", "285.6亿千瓦时", "+4.2%", "electricity"),
        ("工业增加值", "+6.8%", "+0.5%", "industrial"),
        ("社消零总额", "¥3,250亿", "+3.1%", "retail"),
        ("制造业PMI", "51.2", "+0.3", "pmi")
    ]
    
    for col, (name, value, delta, key) in zip(indicator_cols, indicators):
        with col:
            st.metric(label=name, value=value, delta=delta)


def render_nowcast():
    """渲染现时预测页面"""
    st.markdown('<p class="main-header">🔮 现时预测（Nowcasting）</p>', unsafe_allow_html=True)
    
    st.info("""
    **现时预测**利用最新发布的高频数据（月度、周度、日度），
    在季度GDP正式发布前进行实时预测和修正。
    """)
    
    # 当前季度预测
    st.subheader("📅 当前季度预测")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="prediction-highlight">
            <h4>线性模型预测 (MIDAS)</h4>
            <h2>¥12,850亿</h2>
            <p>权重: 60%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="prediction-highlight" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h4>非线性修正 (TSLM)</h4>
            <h2>+170亿</h2>
            <p>权重: 40%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="prediction-highlight" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h4>最终预测值</h4>
            <h2>¥13,020亿</h2>
            <p>置信区间: [12,780, 13,260]</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 数据可用性
    st.subheader("📊 数据可用性")
    
    availability_data = {
        '指标': ['全社会用电量', '工业增加值', '社消零', '固投', 'PMI', 'CPI'],
        '可用数据比例': [85, 80, 75, 70, 90, 100],
        '最新数据日期': ['2024-11', '2024-11', '2024-10', '2024-10', '2024-11', '2024-11']
    }
    
    avail_df = pd.DataFrame(availability_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=avail_df['指标'],
        y=avail_df['可用数据比例'],
        marker_color='#1f77b4',
        text=avail_df['可用数据比例'].apply(lambda x: f'{x}%'),
        textposition='outside'
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="green", 
                  annotation_text="100%")
    
    fig.update_layout(
        title='当前季度数据可用性',
        xaxis_title='指标',
        yaxis_title='可用数据比例 (%)',
        yaxis_range=[0, 110],
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(avail_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 预测历史
    st.subheader("📈 预测修正历史")
    
    # 模拟预测修正数据
    revision_dates = pd.date_range(end=datetime.now(), periods=10, freq='W')
    predictions = [12800, 12850, 12900, 12920, 12950, 12980, 13000, 13010, 13015, 13020]
    
    revision_df = pd.DataFrame({
        '日期': revision_dates,
        '预测值': predictions
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=revision_df['日期'],
        y=revision_df['预测值'],
        mode='lines+markers',
        name='预测值',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='本季度预测修正轨迹',
        xaxis_title='日期',
        yaxis_title='GDP预测值（亿元）',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_historical_data():
    """渲染历史数据页面"""
    st.markdown('<p class="main-header">📊 历史数据查询</p>', unsafe_allow_html=True)
    
    # 数据选择
    st.subheader("🔍 数据筛选")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_type = st.selectbox(
            "数据类型",
            ["GDP数据", "月度指标", "日度金融", "周度指标"]
        )
    
    with col2:
        start_date = st.date_input(
            "开始日期",
            value=datetime(2020, 1, 1)
        )
    
    with col3:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now()
        )
    
    # 根据数据类型显示不同内容
    if data_type == "GDP数据":
        st.subheader("📈 GDP历史数据")
        
        # 模拟数据
        dates = pd.date_range(start=start_date, end=end_date, freq='QE')
        gdp_values = 8000 * (1.035 ** np.arange(len(dates))) + np.random.normal(0, 100, len(dates))
        yoy = np.random.normal(5.5, 1.5, len(dates))
        
        gdp_df = pd.DataFrame({
            '日期': dates,
            'GDP绝对值（亿元）': gdp_values.round(2),
            '同比增速（%）': yoy.round(2)
        })
        
        st.dataframe(gdp_df, use_container_width=True, hide_index=True)
        
        fig = create_gdp_chart(pd.DataFrame({
            'date': dates,
            'gdp_value': gdp_values
        }))
        st.plotly_chart(fig, use_container_width=True)
        
    elif data_type == "月度指标":
        st.subheader("📊 月度经济指标")
        
        indicator = st.selectbox(
            "选择指标",
            ["industrial_output", "retail_sales", "fixed_investment", 
             "electricity", "pmi", "cpi", "ppi"]
        )
        
        # 模拟数据
        dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        values = 100 + np.cumsum(np.random.normal(0, 2, len(dates)))
        
        indicator_df = pd.DataFrame({
            '日期': dates,
            '数值': values.round(2)
        })
        
        st.dataframe(indicator_df, use_container_width=True, hide_index=True)
        
        fig = create_indicator_chart(pd.DataFrame({
            'date': dates,
            indicator: values
        }), indicator)
        st.plotly_chart(fig, use_container_width=True)


def render_factor_analysis():
    """渲染因子分析页面"""
    st.markdown('<p class="main-header">🔍 因子分析</p>', unsafe_allow_html=True)
    
    st.info("""
    **动态因子模型（DFM）**从众多宏观经济指标中提取共同因子，
    这些因子代表了驱动经济波动的核心力量。
    """)
    
    # 因子解释方差
    st.subheader("📊 因子解释方差")
    
    variance_data = pd.DataFrame({
        '因子': ['Factor 1', 'Factor 2', 'Factor 3'],
        '解释方差比例': [45.2, 28.6, 15.3],
        '累计解释方差': [45.2, 73.8, 89.1]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(variance_data, use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=variance_data['因子'],
            y=variance_data['解释方差比例'],
            name='单独解释',
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Scatter(
            x=variance_data['因子'],
            y=variance_data['累计解释方差'],
            name='累计解释',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.update_layout(
            title='因子解释方差',
            yaxis_title='方差比例 (%)',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 因子载荷
    st.subheader("📈 因子载荷矩阵")
    
    loadings_data = pd.DataFrame({
        '指标': ['electricity', 'industrial_output', 'retail_sales', 
                'fixed_investment', 'pmi', 'cpi', 'ppi', 'exports'],
        'Factor 1': [0.85, 0.82, 0.75, 0.78, 0.65, 0.45, 0.52, 0.70],
        'Factor 2': [0.25, 0.30, 0.45, 0.35, 0.55, 0.75, 0.68, 0.40],
        'Factor 3': [0.15, 0.20, 0.15, 0.25, 0.30, 0.25, 0.35, 0.45]
    })
    
    st.dataframe(loadings_data, use_container_width=True, hide_index=True)
    
    # 热力图
    fig = px.imshow(
        loadings_data.set_index('指标').values,
        x=['Factor 1', 'Factor 2', 'Factor 3'],
        y=loadings_data['指标'],
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    fig.update_layout(
        title='因子载荷热力图',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # MIDAS权重
    st.subheader("⚖️ MIDAS滞后权重")
    
    weights = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.12, 0.08]
    
    fig = create_midas_weights_chart(weights)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **权重解读**：MIDAS模型给予近期数据更高的权重，
    最近1个月的权重占比约 **40%**，体现了高频数据的时效性价值。
    """)


def render_model_management():
    """渲染模型管理页面"""
    st.markdown('<p class="main-header">⚙️ 模型管理</p>', unsafe_allow_html=True)
    
    # 模型状态
    st.subheader("📊 模型状态")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MIDAS模型", "✅ 运行中", "RMSE: 125.3")
    
    with col2:
        st.metric("DFM模型", "✅ 运行中", "解释方差: 89.1%")
    
    with col3:
        st.metric("TSLM模型", "✅ 运行中", "GPU: 使用中")
    
    with col4:
        st.metric("混合模型", "✅ 运行中", "准确率: 92.5%")
    
    st.divider()
    
    # 模型控制
    st.subheader("🎮 模型控制")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 重新训练模型", use_container_width=True):
            with st.spinner("正在重新训练模型..."):
                # 模拟训练
                import time
                time.sleep(2)
            st.success("模型重训练完成！")
    
    with col2:
        if st.button("💾 保存模型", use_container_width=True):
            st.success("模型已保存！")
    
    with col3:
        if st.button("📥 加载模型", use_container_width=True):
            st.success("模型已加载！")
    
    st.divider()
    
    # 模型参数
    st.subheader("🔧 模型参数配置")
    
    with st.expander("MIDAS参数"):
        midas_lags = st.slider("滞后阶数", 3, 24, 12)
        midas_poly = st.selectbox("多项式类型", ["exp_almon", "beta", "almon"])
    
    with st.expander("DFM参数"):
        dfm_factors = st.slider("因子数量", 1, 10, 3)
        dfm_order = st.slider("自回归阶数", 1, 5, 1)
    
    with st.expander("TSLM参数"):
        tslm_model = st.selectbox("大模型", ["timesfm", "moirai", "chronos"])
        tslm_context = st.slider("上下文长度", 64, 1024, 512)
    
    with st.expander("混合模型参数"):
        linear_weight = st.slider("线性模型权重", 0.0, 1.0, 0.6)
        st.write(f"非线性模型权重: {1 - linear_weight:.2f}")
    
    st.divider()
    
    # 性能监控
    st.subheader("📈 性能监控")
    
    # 模拟性能历史
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    rmse_history = 130 - np.cumsum(np.random.normal(0.2, 0.5, 30))
    
    perf_df = pd.DataFrame({
        '日期': dates,
        'RMSE': rmse_history
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=perf_df['日期'],
        y=perf_df['RMSE'],
        mode='lines',
        name='RMSE',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title='模型RMSE历史',
        xaxis_title='日期',
        yaxis_title='RMSE',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============ 主程序 ============

def main():
    """主函数"""
    # 渲染侧边栏并获取当前页面
    page = render_sidebar()
    
    # 根据页面选择渲染内容
    if "概览驾驶舱" in page:
        render_overview()
    elif "现时预测" in page:
        render_nowcast()
    elif "历史数据" in page:
        render_historical_data()
    elif "因子分析" in page:
        render_factor_analysis()
    elif "模型管理" in page:
        render_model_management()


if __name__ == "__main__":
    main()
