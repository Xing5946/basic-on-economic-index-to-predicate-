import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="四川经济指标 · 深色版",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- 深色主题CSS --------------------
st.markdown("""
<style>
    /* 全局深色背景 + 白色字体 */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* 侧边栏深色 */
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 1px solid #333333;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* 卡片容器深色边框 */
    .card-border {
        border: 1px solid #333333;
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        background-color: #1e1e1e;
        box-shadow: 0 6px 12px rgba(0,0,0,0.5);
        margin-bottom: 1.5rem;
    }
    /* 指标数值卡片深色 */
    .metric-card {
        background: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 20px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .metric-card label, .metric-card div {
        color: #ffffff !important;
    }
    /* 标题装饰 */
    .section-title {
        font-size: 1.3rem;
        font-weight: 500;
        color: #ffffff;
        margin-bottom: 1.2rem;
        padding-left: 0.5rem;
        border-left: 5px solid #4a9eff;
    }
    /* 覆盖streamlit原生元素颜色 */
    .stMarkdown, .stCaption, .stText, h1, h2, h3, h4, h5, h6, p, li {
        color: #ffffff !important;
    }
    /* 选择框、radio等文字颜色 */
    .stRadio > div {
        color: white;
    }
    /* 下拉框背景 */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #2d2d2d;
        border-color: #444;
    }
    /* 数据表格背景 */
    .dataframe {
        background-color: #1e1e1e !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- 数据准备 (同前) --------------------
df_econ = pd.DataFrame({
    '经济类型': ['国有控股企业', '股份制企业', '外商及港澳台商投资企业', '私营企业'],
    '11月同比增长 (%)': [9.7, 6.5, 5.3, -1.9],
    '1-11月累计增长 (%)': [8.2, 6.3, 14.1, 1.5]
})

df_industry = pd.DataFrame({
    '行业': ['汽车制造业', '电气机械和器材制造业', '计算机、通信和其他电子设备制造业',
             '化学原料和化学制品制造业', '石油和天然气开采业'],
    '累计增长 (%)': [18.2, 13.7, 12.7, 12.0, 11.1]
})

products_data = {
    '产品': ['智能电视', '锂离子电池', '工业机器人', '汽车', '液晶显示屏', '汽油',
             '集成电路', '智能手表', '天然气', '啤酒', '钢材', '发电量', '农用化肥',
             '粗钢', '生铁', '水泥', '微型计算机设备'],
    '累计增长 (%)': [65.6, 45.9, 42.1, 33.9, 22.2, 16.6,
                    14.8, 12.3, 11.5, 5.6, 5.0, 0.7, -1.4,
                    -1.4, -3.3, -4.7, -19.8]
}
df_products = pd.DataFrame(products_data).sort_values('累计增长 (%)', ascending=False).reset_index(drop=True)

df_invest_sector = pd.DataFrame({
    '产业/构成': ['第一产业', '第二产业', '#工业', '第三产业', '建安工程', '设备工器具购置', '其他费用'],
    '增速 (%)': [10.8, 7.7, 8.0, -4.6, -5.2, 11.8, 20.6]
})

real_estate = {
    '指标': ['房地产开发投资', '商品房施工面积', '新建商品房销售面积'],
    '增速 (%)': [-7.1, -11.9, -10.4]
}
df_real_estate = pd.DataFrame(real_estate)

df_consume_urban = pd.DataFrame({
    '所在地': ['城镇', '乡村'],
    '累计增长 (%)': [5.4, 6.0],
    '零售额 (亿元)': [21378.2, 5017.5]
})

df_consume_type = pd.DataFrame({
    '形态': ['餐饮收入', '商品零售'],
    '累计增长 (%)': [3.7, 5.8],
    '零售额 (亿元)': [3614.0, 22781.7]
})

df_hot_goods = pd.DataFrame({
    '商品类别': ['通讯器材类', '金银珠宝类', '粮油食品类', '汽车类'],
    '增速 (%)': [52.5, 29.0, 12.1, 10.7]
})

internet_retail_growth = 21.2

df_tax = pd.DataFrame({
    '指数': ['税电指数', '生产指数', '销售指数'],
    '值': [103.8, 103.7, 104.1]
})

total_industrial = 6.8
total_invest = -0.6
total_retail = 5.5
total_retail_value = 26395.7
产销率 = 95.2

# -------------------- 辅助绘图函数 (适配深色主题) --------------------
def plot_bar(df, x, y, title, color='#4a9eff', orientation='v', text_auto='.1f'):
    if orientation == 'v':
        fig = px.bar(df, x=x, y=y, title=title, text_auto=text_auto, color_discrete_sequence=[color])
    else:
        fig = px.bar(df, y=x, x=y, title=title, text_auto=text_auto, color_discrete_sequence=[color], orientation='h')
    fig.update_traces(textposition='outside', textfont_color='white')
    fig.update_layout(
        title_font_size=16, title_x=0.02,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='white'),
        xaxis=dict(gridcolor='#444', tickfont_color='white'),
        yaxis=dict(gridcolor='#444', tickfont_color='white')
    )
    return fig

def plot_dual_bar(df, x, y1, y2, title, color1='#4a9eff', color2='#ffaa66'):
    fig = go.Figure()
    fig.add_trace(go.Bar(name=y1, x=df[x], y=df[y1], text=df[y1], textposition='outside', marker_color=color1))
    fig.add_trace(go.Bar(name=y2, x=df[x], y=df[y2], text=df[y2], textposition='outside', marker_color=color2))
    fig.update_layout(
        title=title, barmode='group',
        title_font_size=16, title_x=0.02,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='white'),
        xaxis=dict(gridcolor='#444', tickfont_color='white'),
        yaxis=dict(gridcolor='#444', tickfont_color='white')
    )
    return fig

def plot_gauge(value, title, max_val=15):
    """仪表盘图"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [None, max_val], 'tickcolor': 'white'},
            'bar': {'color': "#4a9eff"},
            'bgcolor': '#333',
            'borderwidth': 2,
            'bordercolor': '#555',
            'steps': [
                {'range': [0, max_val/2], 'color': '#444'},
                {'range': [max_val/2, max_val], 'color': '#555'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.8}
        },
        number={'font': {'color': 'white'}}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': 'white'}, height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_pie(df, names, values, title):
    """饼图"""
    fig = px.pie(df, names=names, values=values, title=title, color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#333', width=2)))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'), legend_font_color='white',
                      title_font_color='white')
    return fig

def plot_radar(df, categories, values, title):
    """雷达图"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker_color='#4a9eff',
        line_color='white'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-5, 110],  # 根据数据调整
                gridcolor='#444',
                tickfont_color='white'
            ),
            bgcolor='rgba(30,30,30,0.8)'
        ),
        title=title,
        title_font_color='white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=40, b=20)
    )
    return fig

# -------------------- 侧边栏导航 --------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/region-code.png", width=60)
    st.markdown("## 四川经济指标")
    st.markdown("---")
    page = st.radio(
        "导航菜单",
        ["首页概览", "工业生产", "固定资产投资", "消费品市场", "主要产品产量", "景气指数"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("数据来源：四川省统计局")
    st.caption("时间范围：2025年1-11月")

# -------------------- 首页概览 --------------------
if page == "首页概览":
    st.markdown("<h2 style='font-weight:400; color:white;'>📌 四川经济核心指标</h2>", unsafe_allow_html=True)
    st.markdown("##### 2025年1-11月 主要经济数据速览")

    # 第一行关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("规上工业增加值", "6.8%", delta=None)
        st.caption("累计同比增长")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("固定资产投资", "-0.6%", delta=None)
        st.caption("累计同比增长")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("社消总额", f"{total_retail_value:.1f} 亿", delta=f"{total_retail}%")
        st.caption("累计增长5.5%")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("税电景气指数", "103.8", delta=None)
        st.caption("处于景气区间")
        st.markdown("</div>", unsafe_allow_html=True)

    # 第二行：雷达图 + 仪表盘 + 产销率
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>主要指标对比雷达图</div>", unsafe_allow_html=True)
        # 归一化处理：工业6.8/15，投资-0.6/15但负值处理为0，消费5.5/15，税电(103.8-80)/? 简单处理为(103.8-80)/40 => 约0.595
        radar_vals = [
            total_industrial / 15 * 100,  # 6.8/15≈45.3
            max(0, (total_invest + 5) / 20 * 100),  # 投资-0.6，映射到0-100：加5再除以20，-0.6+5=4.4/20=22
            total_retail / 15 * 100,  # 36.7
            (df_tax[df_tax['指数']=='税电指数']['值'].values[0] - 80) / 40 * 100  # (103.8-80)/40=59.5
        ]
        categories = ['规上工业', '固定资产投资', '社会消费', '税电景气']
        fig_radar = plot_radar(None, categories, radar_vals, "")
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>规上工业增速仪表</div>", unsafe_allow_html=True)
        fig_gauge = plot_gauge(total_industrial, "增速%", max_val=15)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 第三行：房地产 + 乡村消费 + 互联网零售
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("##### 🏭 工业产销率")
        st.markdown(f"<h1 style='color:#4a9eff;'>{产销率}%</h1>", unsafe_allow_html=True)
        st.caption("规模以上工业企业产品产销率")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("##### 🏠 房地产开发")
        fig_re = px.bar(df_real_estate, x='指标', y='增速 (%)', text_auto='.1f',
                        color_discrete_sequence=['#ffaa66'])
        fig_re.update_layout(showlegend=False, height=200,
                             plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                             font_color='white', xaxis_tickfont_color='white', yaxis_tickfont_color='white')
        st.plotly_chart(fig_re, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("##### 🌾 乡村消费")
        st.metric("乡村消费品零售额", "5017.5 亿", delta="6.0%")
        st.caption("增速高于城镇 (5.4%)")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 工业生产 --------------------
elif page == "工业生产":
    st.markdown("<h2 style='font-weight:400; color:white;'>🏭 规模以上工业生产</h2>", unsafe_allow_html=True)

    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>按经济类型增加值增速</div>", unsafe_allow_html=True)
    fig1 = plot_dual_bar(df_econ, '经济类型', '11月同比增长 (%)', '1-11月累计增长 (%)', '')
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("注：私营企业11月同比下滑1.9%，但累计增长1.5%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>重点行业增加值增速（1-11月累计）</div>", unsafe_allow_html=True)
    fig2 = plot_bar(df_industry, '行业', '累计增长 (%)', '', orientation='h', color='#ffaa66')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>主要工业产品产量增速 TOP8</div>", unsafe_allow_html=True)
    top_products = df_products.head(8)
    fig3 = plot_bar(top_products, '产品', '累计增长 (%)', '', orientation='h', color='#6daffe')
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 固定资产投资 --------------------
elif page == "固定资产投资":
    st.markdown("<h2 style='font-weight:400; color:white;'>🏗️ 固定资产投资</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>分产业投资增速</div>", unsafe_allow_html=True)
        fig_inv1 = plot_bar(df_invest_sector[df_invest_sector['产业/构成'].isin(['第一产业','第二产业','第三产业'])],
                            '产业/构成', '增速 (%)', '', color='#4a9eff')
        st.plotly_chart(fig_inv1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>按构成分增速</div>", unsafe_allow_html=True)
        df_invest_structure = df_invest_sector[df_invest_sector['产业/构成'].isin(['建安工程','设备工器具购置','其他费用'])]
        fig_inv2 = plot_bar(df_invest_structure, '产业/构成', '增速 (%)', '', color='#ffaa66')
        st.plotly_chart(fig_inv2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>房地产开发主要指标</div>", unsafe_allow_html=True)
    fig_re = px.bar(df_real_estate, x='指标', y='增速 (%)', text_auto='.1f', color_discrete_sequence=['#6daffe'])
    fig_re.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                         font_color='white', xaxis_tickfont_color='white', yaxis_tickfont_color='white')
    st.plotly_chart(fig_re, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 消费品市场 --------------------
elif page == "消费品市场":
    st.markdown("<h2 style='font-weight:400; color:white;'>🛍️ 消费品市场</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>城乡消费增速</div>", unsafe_allow_html=True)
        fig_con1 = plot_bar(df_consume_urban, '所在地', '累计增长 (%)', '', color='#4a9eff')
        st.plotly_chart(fig_con1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>消费形态增速</div>", unsafe_allow_html=True)
        fig_con2 = plot_bar(df_consume_type, '形态', '累计增长 (%)', '', color='#ffaa66')
        st.plotly_chart(fig_con2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 新增：饼图展示城乡零售额占比
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>城乡零售额占比</div>", unsafe_allow_html=True)
        fig_pie1 = plot_pie(df_consume_urban, '所在地', '零售额 (亿元)', '')
        st.plotly_chart(fig_pie1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>消费形态占比</div>", unsafe_allow_html=True)
        fig_pie2 = plot_pie(df_consume_type, '形态', '零售额 (亿元)', '')
        st.plotly_chart(fig_pie2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 热点商品
    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>热点商品零售增速</div>", unsafe_allow_html=True)
    fig_hot = plot_bar(df_hot_goods, '商品类别', '增速 (%)', '', orientation='v', color='#6daffe')
    st.plotly_chart(fig_hot, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📱 互联网零售</div>", unsafe_allow_html=True)
    st.metric("限额以上单位互联网商品零售额", "2110.2 亿元", delta=f"{internet_retail_growth}%")
    st.caption("1-11月累计同比增长21.2%，远高于商品零售整体增速")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 主要产品产量 --------------------
elif page == "主要产品产量":
    st.markdown("<h2 style='font-weight:400; color:white;'>📦 主要工业产品产量增速</h2>", unsafe_allow_html=True)
    st.markdown("##### 全产品列表 (1-11月累计同比)")

    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    fig_all = px.bar(df_products, y='产品', x='累计增长 (%)', text_auto='.1f',
                     orientation='h', color='累计增长 (%)',
                     color_continuous_scale=['#d62828', '#fcbf49', '#4a9eff'], range_color=[-25,70])
    fig_all.update_layout(height=700, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font_color='white', xaxis_tickfont_color='white', yaxis_tickfont_color='white',
                          coloraxis_colorbar=dict(title='增速%', tickfont_color='white', title_font_color='white'))
    fig_all.update_traces(textfont_color='white')
    st.plotly_chart(fig_all, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 新增：散点图展示产品增速分布
    st.markdown("<div class='card-border'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>产品增速分布散点图</div>", unsafe_allow_html=True)
    fig_scatter = px.scatter(df_products, x='产品', y='累计增长 (%)', size='累计增长 (%)', color='累计增长 (%)',
                              color_continuous_scale=['#d62828', '#fcbf49', '#4a9eff'], size_max=40)
    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font_color='white', xaxis_tickfont_color='white', yaxis_tickfont_color='white',
                              xaxis_tickangle=-45)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 查看产品数据表格"):
        st.dataframe(df_products, use_container_width=True, hide_index=True)

# -------------------- 景气指数 --------------------
elif page == "景气指数":
    st.markdown("<h2 style='font-weight:400; color:white;'>📈 经济景气度税电指数</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("##### 总指数")
        st.markdown(f"<h1 style='color:#4a9eff;'>{df_tax[df_tax['指数']=='税电指数']['值'].values[0]}</h1>", unsafe_allow_html=True)
        st.caption("景气临界值 = 100")
        st.markdown("</div>", unsafe_allow_html=True)

        # 新增仪表盘
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("##### 景气仪表")
        fig_gauge_tax = plot_gauge(df_tax[df_tax['指数']=='税电指数']['值'].values[0], "", max_val=120)
        st.plotly_chart(fig_gauge_tax, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-border'>", unsafe_allow_html=True)
        st.markdown("##### 分项指数")
        fig_tax = px.bar(df_tax[df_tax['指数']!='税电指数'], x='指数', y='值', text_auto='.1f',
                         color_discrete_sequence=['#ffaa66', '#6daffe'])
        fig_tax.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font_color='white', xaxis_tickfont_color='white', yaxis_tickfont_color='white')
        st.plotly_chart(fig_tax, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("税电指数根据企业税收和用电量合成，高于100表示经济处于扩张区间。")

# -------------------- 页脚 --------------------
st.markdown("---")
st.caption("可视化基于四川省统计局2025年1-11月数据制作 | 单位：%  | 部分数据因四舍五入存在分项差异")