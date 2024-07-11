import uvicorn
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile, Request
from joblib import dump, load
import json

#uvicorn app:app --host 0.0.0.0 --port $PORT
#render.com

### 
# Here some configurations 
###
description="""
API to estimate the rental price of a car.

This API provides an endpoint to predict the rental price per day of a car based on various features.

Endpoints:
- GET /: Display random examples of the data.
- POST /predict: Predict the rental price per day of a car.

"""

app = FastAPI(
    title="Getaround Pricing Predictor API",
    description=description
)

###
# Here you define enpoints 
###

@app.get("/")
async def index():
    message = """Welcome to the GetAround API !"""
    return message

@app.get("/Sample cars")
async def load_sample_cars():
    """
    Display 3 random examples
    """
    cars = pd.read_csv("data/get_around_pricing_project.csv", index_col=0)
    car = cars.sample(3)
    return car.to_dict("index")

class PredictionFeatures(BaseModel):
    model_key: str
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool
    
@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Get the predicted price of a car.

    Parameters:
    predictionFeatures (PredictionFeatures): A dictionary containing the following keys:
        - "model_key" (str): The model key of the car.
        - "mileage" (int): The mileage of the car.
        - "engine_power" (int): The engine power of the car.
        - "fuel" (str): The fuel type of the car. Must be one of the following: 'petrol', 'hybrid_petrol', 'electro'.
        - "paint_color" (str): The paint color of the car. Must be one of the following: 'black', 'grey', 'white', 'red', 'silver', 'blue', 'orange', 'beige', 'brown', 'green'.
        - "car_type" (str): The type of the car. Must be one of the following: 'convertible', 'coupe', 'estate', 'hatchback', 'sedan', 'subcompact', 'suv', 'van'.
        - "private_parking_available" (bool): Whether the car has private parking available.
        - "has_gps" (bool): Whether the car has GPS.
        - "has_air_conditioning" (bool): Whether the car has air conditioning.
        - "automatic_car" (bool): Whether the car is automatic.
        - "has_getaround_connect" (bool): Whether the car has Getaround Connect.
        - "has_speed_regulator" (bool): Whether the car has a speed regulator.
        - "winter_tires" (bool): True.

    Returns:
    dict: A dictionary containing the predicted rental price per day.
        - "predictions" (float): The predicted rental price per day.
    """
    data = pd.DataFrame(dict(predictionFeatures), index=[0])
    loaded_model = load('api/model_xg_getaround.joblib')
    prediction = loaded_model.predict(data)
    response ={"predictions": prediction.tolist()[0]}
    return response