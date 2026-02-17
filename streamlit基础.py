import streamlit as st
import numpy as np
import pandas as pd

#操作逻辑：对网页中的任何操作都将重新渲染一次网页
#3更多组件
st.title("四川省2025第四季度宏观经济指标")
st.header("第四季度")#子标题
st.subheader("宏观经济指标")

#st.markdown()#markdown的形式
#st.code()#代码格式
#st.divider()#分割线
#st.image()#插入图片

#4表格
df = pd.DataFrame({
    "month": ["Sept.", "Oct.", "Nov.", "Dec."],
    "cpi_index": [0.012, 0.015, 0.009, 0.010],
    "ppi": [0.021, 0.022, 0.018, 0.019],
})


st.dataframe(df.style.format({
   "cpi_index": "{:.1%}",  # 格式化为1位小数的百分比
    "ppi": "{:.1%}",
}))
#以字典的格式进行输入



st.metric(label="CPI growth index:",value="ABC") #突出型字符

#5画图各种图标到时候查
st.line_chart(data=df,x="month",y="cpi_index")

#6表单：所有内容集成到字典当中,表单中具体组件到时候查
with st.form(key="form 1"):
    respond_info = {
        "name":None,
        "month":None,
        "index":None,
        "age_info":None

    }
    respond_info["name"]=st.text_input(label="name")
    respond_info["month"]=st.selectbox(label="which month",options=["Sept.", "Oct.", "Nov.", "Dec."])
    respond_info["age_info"]=st.slider(label="age",min_value=18,max_value=80)
    respond_info["index"]=st.selectbox(label="index",options=["cpi","ppi"])
    form_summitted = st.form_submit_button(label="summit!")

def check_form_valid():
    # 只检查实际有输入控件的字段
    required_fields = ["name", "month", "age_info"]
    for field in required_fields:
        val = respond_info[field]
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return False
    return True
    
    
if form_summitted:
        if not check_form_valid():
            st.warning("Please fill all values")
        else:
            st.balloons()
            st.write("form summitted!")

print(respond_info)
    