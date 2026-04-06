from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Загружаем твой "движок"
model_core = joblib.load('zenith_model_v4.pkl')

@app.get("/predict")
def get_prediction(start_year: int = 2025, end_year: int = 2035):
    # Вызываем твой метод прогноза
    years, values = model_core.project_future_horizon(start_year, end_year)
    
    # Формируем ответ в JSON (понятный для фронтенда)
    return {
        "years": years.tolist(),
        "values": [round(v, 2) for v in values]
    }

# Запуск сервера командой: uvicorn main:app --reload