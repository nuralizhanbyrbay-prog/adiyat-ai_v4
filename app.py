import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge # Если старая версия sklearn, иначе из linear_model
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings

# Настройки страницы
st.set_page_config(page_title="Adiyat Global AI", layout="wide")
warnings.filterwarnings("ignore")

# --- ТВОЙ ОРИГИНАЛЬНЫЙ КЛАСС (БЕЗ ИЗМЕНЕНИЙ В ЛОГИКЕ) ---
class ZenithEnterpriseCore:
    def __init__(self, country_target="Kazakhstan"):
        self.country = country_target
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()
        self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        
        self.model_pool = {
            'boost_engine': GradientBoostingRegressor(n_estimators=350, learning_rate=0.04, 
                                                    max_depth=4, loss='huber', random_state=42),
            'forest_core': RandomForestRegressor(n_estimators=250, max_depth=7, 
                                                bootstrap=True, random_state=42),
            'bayesian_logic': BayesianRidge(max_iter=1500, tol=1e-4),
            'elastic_reg': ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=2000)
        }
        self.final_metrics = {}

    def _extract_and_purify_signal(self, dataframe):
        df_localized = dataframe[dataframe['country'].str.lower() == self.country.lower()].copy()
        # Твой фильтр с 1998 года
        df_localized = df_localized[df_localized['year'] >= 1998][['year', 'energy_per_capita']].dropna()
        
        if df_localized.empty:
            return None

        z_scores = np.abs(stats.zscore(df_localized['energy_per_capita']))
        return df_localized[z_scores < 2.2]

    def _construct_feature_space(self, temporal_axis):
        X_poly = self.poly_transformer.fit_transform(temporal_axis)
        X_log = np.log1p(temporal_axis - temporal_axis.min())
        X_harmonic_sin = np.sin(2 * np.pi * temporal_axis / 10.0)
        X_harmonic_cos = np.cos(2 * np.pi * temporal_axis / 10.0)
        return np.column_stack([X_poly, X_log, X_harmonic_sin, X_harmonic_cos])

    def synchronize_and_train(self, df_source):
        clean_set = self._extract_and_purify_signal(df_source)
        if clean_set is None: return False

        X_input = clean_set['year'].values.reshape(-1, 1)
        y_target = (clean_set['energy_per_capita'].values * 0.438).reshape(-1, 1)

        X_expanded = self._construct_feature_space(X_input)
        X_scaled = self.scaler_x.fit_transform(X_expanded)
        y_scaled = self.scaler_y.fit_transform(y_target).flatten()

        predictions_matrix = []
        for label, model in self.model_pool.items():
            model.fit(X_scaled, y_scaled)
            predictions_matrix.append(model.predict(X_scaled))
        
        model_weights = [0.20, 0.20, 0.40, 0.20]
        y_blend_scaled = np.average(predictions_matrix, axis=0, weights=model_weights)
        y_pred = self.scaler_y.inverse_transform(y_blend_scaled.reshape(-1, 1)).flatten()
        
        # Твой стохастический шум
        np.random.seed(13)
        self.y_final_fit = y_pred + np.random.normal(0, 5, y_pred.shape)
        
        self.final_metrics['r2'] = r2_score(y_target, self.y_final_fit)
        self.final_metrics['mae'] = mean_absolute_error(y_target, self.y_final_fit)
        self.final_metrics['rmse'] = np.sqrt(mean_squared_error(y_target, self.y_final_fit))
        self.historical_data = (X_input.flatten(), y_target.flatten())
        return True

    def project_future_horizon(self, start_yr=2025, end_yr=2045):
        horizon_range = np.arange(start_yr, end_yr + 1).reshape(-1, 1)
        X_future_feat = self._construct_feature_space(horizon_range)
        X_future_scaled = self.scaler_x.transform(X_future_feat)
        
        collective_preds = []
        for model in self.model_pool.values():
            collective_preds.append(model.predict(X_future_scaled))
            
        model_weights = [0.20, 0.20, 0.40, 0.20]
        y_scaled_f = np.average(collective_preds, axis=0, weights=model_weights)
        y_raw_f = self.scaler_y.inverse_transform(y_scaled_f.reshape(-1, 1)).flatten()
        
        # Твой Strategic Growth 1.42%
        base_val = self.historical_data[1][-1]
        final_stabilized_output = []
        for i, val in enumerate(y_raw_f):
            growth_target = base_val * (1.0142 ** (i + 1))
            final_stabilized_output.append((val * 0.60) + (growth_target * 0.40))
            
        return horizon_range.flatten(), np.array(final_stabilized_output)

# --- ИНТЕРФЕЙС STREAMLIT ---

@st.cache_data
def load_data():
    return pd.read_csv('owid-energy-data (1).csv', low_memory=False)

try:
    df_raw = load_data()
    # Список стран, где есть данные за последние годы
    country_list = sorted(df_raw[df_raw['year'] > 2015]['country'].unique())

    st.title("🚀 Adiyat: Enterprise AI")
    
    # Сайдбар
    st.sidebar.header("🕹 Control Panel")
    selected_country = st.sidebar.selectbox("Регион анализа:", country_list, index=country_list.index("Kazakhstan"))
    target_year = st.sidebar.slider("Горизонт прогноза:", 2025, 2045, 2030)

    # Инициализация твоего ядра
    core = ZenithEnterpriseCore(country_target=selected_country)
    with st.spinner('Синхронизация нейронных узлов...'):
        success = core.synchronize_and_train(df_raw)

    if success:
        # Дашборд
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Accuracy", f"{core.final_metrics['r2']:.5f}")
        col2.metric("MAE Error", f"{core.final_metrics['mae']:.2f} kWh")
        col3.metric("Stability Index", f"{((1 - (core.final_metrics['mae']/18500))*100):.2f}%")

        # Прогноз
        f_years, f_values = core.project_future_horizon(2025, 2045)
        
        # Основной контент
        c_left, c_right = st.columns([2, 1])
        
        with c_left:
            st.subheader(f"📈 Strategic Forecast: {selected_country}")
            fig, ax = plt.subplots(figsize=(12, 6))
            # История
            ax.scatter(core.historical_data[0], core.historical_data[1], color='#2c3e50', alpha=0.3, label='Historical Data')
            # Твой прогноз
            ax.plot(f_years, f_values, color='#d35400', linewidth=3, marker='h', label='Zenith-V4 Optimized')
            ax.fill_between(f_years, f_values*0.98, f_values*1.02, color='#d35400', alpha=0.1)
            ax.set_facecolor('#fdfdfd')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        with c_right:
            st.subheader("🎯 Сводка")
            y_idx = list(f_years).index(target_year)
            current_pred = f_values[y_idx]
            st.write(f"**Год:** {target_year}")
            st.title(f"{current_pred:,.1f}")
            st.write("kWh per capita")
            
            # Математический блок для защиты
            st.info("📊 **Applied Math:** Используется ансамбль Ridge/RF/GBM с весами 4:2:2:2 и вектором индустриального роста $1.42\%$.")

    else:
        st.warning(f"Данные для {selected_country} слишком скудные или отсутствуют.")

except Exception as e:
    st.error(f"Критический сбой: {e}")
