import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
import io

# --- 1. CONFIGURATION & STYLES ---
st.set_page_config(
    page_title="Adiyat Energy OS v4",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS для "премиального" вида
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #161b22; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE CORE ENGINE ---
class ZenithEnterpriseCore:
    def __init__(self, country_target="Kazakhstan"):
        self.country = country_target
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()
        self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        self.model_pool = {
            'GBM': GradientBoostingRegressor(n_estimators=350, learning_rate=0.04, max_depth=4, loss='huber', random_state=42),
            'RF': RandomForestRegressor(n_estimators=250, max_depth=7, random_state=42),
            'Bayesian': BayesianRidge(max_iter=1500),
            'Elastic': ElasticNet(alpha=0.01, l1_ratio=0.7)
        }
        self.metrics = {}
        self.historical_data = None

    def _extract_data(self, df):
        sub = df[df['country'].str.lower() == self.country.lower()].copy()
        sub = sub[sub['year'] >= 1990][['year', 'energy_per_capita', 'gdp', 'population']].dropna()
        return sub if len(sub) > 10 else None

    def _build_features(self, years):
        years = np.array(years).reshape(-1, 1)
        X_poly = self.poly_transformer.fit_transform(years)
        X_log = np.log1p(years - years.min())
        X_sin = np.sin(2 * np.pi * years / 10.0)
        return np.column_stack([X_poly, X_log, X_sin])

    def train(self, df):
        data = self._extract_data(df)
        if data is None: return False
        
        X = data['year'].values.reshape(-1, 1)
        y = (data['energy_per_capita'].values * 0.438).reshape(-1, 1)
        
        X_feat = self._build_features(X)
        X_scaled = self.scaler_x.fit_transform(X_feat)
        y_scaled = self.scaler_y.fit_transform(y).flatten()
        
        preds = []
        for m in self.model_pool.values():
            m.fit(X_scaled, y_scaled)
            preds.append(m.predict(X_scaled))
        
        y_blend = np.average(preds, axis=0, weights=[0.2, 0.2, 0.4, 0.2])
        y_final = self.scaler_y.inverse_transform(y_blend.reshape(-1, 1)).flatten()
        
        self.metrics['r2'] = r2_score(y, y_final)
        self.metrics['mae'] = mean_absolute_error(y, y_final)
        self.historical_data = (X.flatten(), y.flatten())
        return True

    def predict(self, end_year, growth_rate=1.0142):
        start_year = int(self.historical_data[0][-1]) + 1
        future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
        
        X_feat = self._build_features(future_years)
        X_scaled = self.scaler_x.transform(X_feat)
        
        preds = [m.predict(X_scaled) for m in self.model_pool.values()]
        y_blend = np.average(preds, axis=0, weights=[0.2, 0.2, 0.4, 0.2])
        y_raw = self.scaler_y.inverse_transform(y_blend.reshape(-1, 1)).flatten()
        
        # Интеграция вектора роста
        base = self.historical_data[1][-1]
        final = []
        for i, val in enumerate(y_raw):
            target = base * (growth_rate ** (i + 1))
            final.append((val * 0.6) + (target * 0.4))
        return future_years.flatten(), np.array(final)

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def get_dataset():
    return pd.read_csv('owid-energy-data (1).csv')

# --- 4. SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2991/2991610.png", width=100)
st.sidebar.title("Adiyat Control Panel")

raw_df = get_dataset()
countries = sorted(raw_df['country'].unique())

primary_country = st.sidebar.selectbox("Основная страна", countries, index=countries.index("Kazakhstan"))
compare_mode = st.sidebar.checkbox("Режим сравнения")
compare_country = None
if compare_mode:
    compare_country = st.sidebar.selectbox("Страна для сравнения", countries, index=countries.index("Germany"))

predict_horizon = st.sidebar.slider("Горизонт (Год)", 2026, 2055, 2040)
custom_growth = st.sidebar.number_input("Коэф. роста (1.0142 = 1.42%)", 1.0, 1.42, 1.0142, step=0.001)

# --- 5. MAIN LOGIC ---
st.title("⚡ Adiyat Energy Intelligence v4")
st.caption("Industrial-grade predictive system for global energy consumption analysis")

if st.sidebar.button("RUN SYSTEM ANALYSIS", use_container_width=True):
    # Тренируем основную модель
    core1 = ZenithEnterpriseCore(primary_country)
    if core1.train(raw_df):
        st.session_state['core1'] = core1
        
    # Если режим сравнения
    if compare_mode and compare_country:
        core2 = ZenithEnterpriseCore(compare_country)
        if core2.train(raw_df):
            st.session_state['core2'] = core2

# --- 6. DASHBOARD ---
if 'core1' in st.session_state:
    c1 = st.session_state['core1']
    
    # Метрики
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Current Accuracy", f"{c1.metrics['r2']:.4%}")
    m_col2.metric("MAE", f"{c1.metrics['mae']:.2f} kWh")
    
    t1, t2, t3 = st.tabs(["📈 Визуализация прогноза", "🔍 Сравнение и Анализ", "📄 Отчеты"])
    
    with t1:
        f_yrs, f_vals = c1.predict(predict_horizon, custom_growth)
        
        fig = go.Figure()
        # История
        fig.add_trace(go.Scatter(x=c1.historical_data[0], y=c1.historical_data[1], 
                                 name=f"History ({c1.country})", mode='lines+markers', line=dict(color='#30363d')))
        # Прогноз
        fig.add_trace(go.Scatter(x=f_yrs, y=f_vals, name=f"AI Forecast ({c1.country})", 
                                 line=dict(color='#00d1ff', width=4, dash='dot')))
        
        fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        if compare_mode and 'core2' in st.session_state:
            c2 = st.session_state['core2']
            f_yrs2, f_vals2 = c2.predict(predict_horizon, custom_growth)
            
            fig_comp = px.line(title="Сравнительный анализ трендов")
            fig_comp.add_scatter(x=f_yrs, y=f_vals, name=c1.country)
            fig_comp.add_scatter(x=f_yrs2, y=f_vals2, name=c2.country)
            fig_comp.update_layout(template="plotly_dark")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Включите 'Режим сравнения' в боковой панели, чтобы увидеть графики двух стран одновременно.")

    with t3:
        st.subheader("Data Export Center")
        export_df = pd.DataFrame({"Year": f_yrs, "Prediction": f_vals})
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions (CSV)", csv, "zenith_forecast.csv", "text/csv")
        
        st.write("### Технический аудит модели")
        st.json(c1.model_pool['GBM'].get_params())
else:
    st.warning("Ожидание инициализации... Нажмите 'RUN SYSTEM ANALYSIS' в боковом меню.")
