import os
import random
import joblib
import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_pipeline(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


MODEL_FILES = {
    "KNN": "knn_pipeline.joblib",
    "LogisticRegression": "logistic_regression_pipeline.joblib",
    # "隨機森林": "randomforest_pipeline.joblib",
    "XgBoost": "xgboost_pipeline.joblib",
}


def main():
    st.set_page_config(layout="wide", page_title="信用卡違約預測")

    cols = st.columns([1, 3])
    left = cols[0]
    right = cols[1]

    with left:
        st.header("選擇模型")
        model_name = st.selectbox("模型", list(MODEL_FILES.keys()))
        predict_btn = st.button("隨機抽樣並預測")

    with right:
        st.header("資料預覽與目標分佈")
        data_path = "UCI_Credit_Card.csv"
        if not os.path.exists(data_path):
            st.error(f"找不到資料檔：{data_path}")
            return

        df = load_data(data_path)
        st.subheader("資料前10筆")
        st.dataframe(df.head(10))

        target = "default.payment.next.month"
        drop_cols = [c for c in ["ID", target] if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df[target] if target in df.columns else None

        if y is None:
            st.error(f"資料中找不到目標欄位：{target}")
        else:
            st.subheader("目標類別數量")
            st.bar_chart(y.value_counts())

    if predict_btn:
        if 'df' not in locals():
            st.error("資料尚未載入，無法進行預測")
            return

        sample = df.sample(n=1, random_state=random.randint(0, 2 ** 31 - 1))
        st.subheader("抽中的樣本")
        st.table(sample.T)

        if target in sample.columns:
            X_row = sample.drop(columns=[c for c in ["ID", target] if c in sample.columns])
        else:
            X_row = sample.drop(columns=[c for c in ["ID"] if c in sample.columns])

        model_file = MODEL_FILES.get(model_name)
        model_path = model_file if os.path.exists(model_file) else os.path.join(os.getcwd(), model_file)

        clf = load_pipeline(model_path)
        if clf is None:
            st.error(f"找不到模型檔：{model_file}")
            return

        try:
            pred = clf.predict(X_row)
            st.success(f"模型 {model_name} 預測結果： {pred[0]}")
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_row)
                st.write("預測機率：", proba[0])
        except Exception as e:
            st.error(f"預測失敗：{e}")


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. 頁面配置
st.set_page_config(page_title="金融信用預測儀表板", layout="wide")

# 2. 定義快取函式 (提升效能)
@st.cache_resource
def load_model(model_name):
    # 這裡的檔名需與你下載的 joblib 檔案名稱一致
    model_files = {
        "KNN": "pipeline_knn.joblib",
        "LogisticRegression": "pipeline_lr.joblib",
        "RandomForest": "pipeline_rf.joblib",
        "XGBoost": "pipeline_xgb.joblib"
    }
    return joblib.load(model_files[model_name])

@st.cache_data
def load_data():
    import os
    local_csv = "UCI_Credit_Card.csv"
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv)
    else:
        url = "https://raw.githubusercontent.com/ywang166/Credit-Card-Default-Prediction/master/data/default%20of%20credit%20card%20clients.csv"
        df = pd.read_csv(url, skiprows=1)

    # 分離特徵與標籤 (為了之後預測用)
    cols = df.columns.tolist()
    possible_labels = [
        'default payment next month',
        'default.payment.next.month',
        'default_payment_next_month',
        'default.payment_next_month'
    ]
    label_col = next((c for c in cols if c in possible_labels), None)
    if label_col is None:
        for c in cols:
            if 'default' in c.lower() and 'next' in c.lower():
                label_col = c
                break
    if label_col is None:
        raise ValueError("找不到標籤欄位 (default ...)，請檢查 CSV 欄位名稱")

    id_col = next((c for c in cols if c.lower() == 'id'), None)
    drop_cols = [label_col]
    if id_col:
        drop_cols.insert(0, id_col)

    X = df.drop(drop_cols, axis=1)
    y = df[label_col]
    return df, X, y

# 3. 載入資料
df_full, X, y = load_data()

# --- 左側選單 (Sidebar) ---
st.sidebar.title("🤖 模型控制中心")
selected_name = st.sidebar.selectbox(
    "請選擇分類模型：",
    ["KNN", "LogisticRegression", "RandomForest", "XGBoost"]
)
model = load_model(selected_name)

st.sidebar.divider()
st.sidebar.info(f"當前模型：{selected_name}\n\n這是一個包含 Scaler, PCA 與 Classifier 的完整 Pipeline。")

# --- 右側主畫面 ---
st.title("💳 信用卡違約風險預測展示")

# A. 數據概覽
st.subheader("📋 數據集概覽 (前 10 筆樣本)")
st.dataframe(df_full.head(10), use_container_width=True)

st.divider()

# B. 隨機預測區塊
st.subheader("🎯 即時預測測試")

# 初始化 session_state 用於儲存抽樣結果
if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = None

if st.button("🎲 隨機抽取一個樣本進行預測"):
    st.session_state.sample_idx = np.random.randint(0, len(X))

# 如果已經抽樣，則進行顯示與預測
if st.session_state.sample_idx is not None:
    idx = st.session_state.sample_idx
    
    # 取出單筆資料 (DataFrame 格式，Pipeline 才能吃)
    sample_data = X.iloc[[idx]]
    actual_label = y.iloc[idx]
    
    st.write(f"**抽取的樣本索引：** `{idx}`")
    st.dataframe(sample_data)
    
    # 執行 Pipeline 預測 (自動內含 Scaling 與 PCA)
    prediction = model.predict(sample_data)[0]
    # 預測機率 (XGB, RF, LR 支援，KNN 也支援)
    prob = model.predict_proba(sample_data)[0][1]
    
    # --- 下方顯示結果 ---
    st.subheader("🚀 預測結果")
    
    # 使用欄位排版顯示指標
    col1, col2, col3 = st.columns(3)
    
    with col1:
        res_text = "⚠️ 違約" if prediction == 1 else "✅ 正常"
        st.metric("模型預測", res_text)
        
    with col2:
        actual_text = "⚠️ 違約" if actual_label == 1 else "✅ 正常"
        st.metric("真實情況", actual_text)
        
    with col3:
        st.metric("違約機率", f"{prob:.2%}")

    # 比對結果
    if prediction == actual_label:
        st.success("🎉 預測正確！該模型成功捕捉到樣本特徵。")
    else:

        st.error("❌ 預測失誤。這反映了模型在邊際樣本上的侷限性。")
