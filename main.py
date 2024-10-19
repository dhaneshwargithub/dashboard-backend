from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pickle
import pandas as pd
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np


app = FastAPI()

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = os.getenv('MODEL_PATH', './forecast_model (3).pkl')
try:
    with open(model_path, 'rb') as f:
        model_fit = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load the model from {model_path}") from e

# Hardcoded dropdown options
COUNTRIES = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Netherlands',
             'Sweden', 'Belgium', 'Austria', 'Ireland', 'Portugal', 'Finland', 'Denmark',
             'Norway', 'Switzerland']
CATEGORIES = ['Office Supplies', 'Technology', 'Furniture']
SUB_CATEGORIES = ['Storage', 'Accessories', 'Labels', 'Phones', 'Copiers', 'Appliances',
                  'Fasteners', 'Art', 'Machines', 'Binders', 'Bookcases', 'Paper', 'Supplies',
                  'Tables', 'Chairs', 'Furnishings', 'Envelopes']

class ForecastRequest(BaseModel):
    country: str
    category: str
    sub_category: str
    start_date: str  # Format: YYYY-MM
    forecast_months: int

    @validator('start_date')
    def validate_start_date(cls, v):
        try:
            pd.to_datetime(v, format='%Y-%m')
            return v
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM format.")

    @validator('forecast_months')
    def validate_forecast_months(cls, v):
        if v <= 0 or v > 24:
            raise ValueError("Forecast months must be a positive integer and up to 24.")
        return v

def forecast_sales(country: str, category: str, sub_category: str, start_date: str, forecast_months: int) -> Dict[str, Any]:
    try:

        # Convert start_date to datetime
        start_forecast_date = pd.to_datetime(start_date, format='%Y-%m')

        # Forecast the next forecast_months months
        forecast = model_fit.get_forecast(steps=forecast_months)
        forecast_df = forecast.summary_frame()
        forecast_df.index = pd.date_range(start=start_forecast_date, periods=forecast_months, freq='M')

        # Get actual sales for the forecasted period from the model data
        actual_sales_during_forecast = model_fit.data.endog

        # Get the index corresponding to the start date
        start_index = model_fit.data.row_labels.get_loc(start_forecast_date)

        # Adjust the actual sales to match the length of the forecast period
        actual_sales_during_forecast = actual_sales_during_forecast[start_index:start_index + forecast_months]

        # If actual sales data is shorter than the forecast period, pad with None/NaN
        if len(actual_sales_during_forecast) < forecast_months:
            padding_length = forecast_months - len(actual_sales_during_forecast)
            actual_sales_during_forecast = list(actual_sales_during_forecast) + [np.nan] * padding_length

        # Combine actual and forecasted sales for display
        combined_df = pd.DataFrame({
            'Forecasted Sales': forecast_df['mean'],
            'Actual Sales': actual_sales_during_forecast
        }, index=forecast_df.index)

        # Replace NaN and infinite values with None to make it JSON serializable
        combined_df = combined_df.replace([np.nan, np.inf, -np.inf], None)

        return combined_df.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")



@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        forecast_df = forecast_sales(
            request.country,
            request.category,
            request.sub_category,
            request.start_date,
            request.forecast_months
        )
        return forecast_df
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/dropdown-options")
def get_dropdown_options():
    try:
        return {
            "countries": COUNTRIES,
            "categories": CATEGORIES,
            "subCategories": SUB_CATEGORIES
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dropdown options: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
