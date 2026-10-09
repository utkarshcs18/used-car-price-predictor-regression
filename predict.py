import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_car_price_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "max_power",
    "torque_nm",
    "engine",
    "year",
    "brand_BMW",
    "brand_Mercedes-Benz",
    "fuel_Diesel",
    "brand_Audi",
    "brand_Volvo",
    "owner_Test Drive Car",
    "fuel_Petrol",
    "seller_type_Individual",
    "transmission",
]


def predict_car_price():
    print("\n===== Used Car Price Predictor =====\n")

    max_power = float(input("Maximum power (bhp): "))
    torque_nm = float(input("Torque (Nm): "))
    engine = float(input("Engine capacity (CC): "))
    year = float(input("Year of manufacture: "))

    brand = input(
        "Brand (BMW/Mercedes-Benz/Audi/Volvo/Other): "
    ).strip()

    fuel = input(
        "Fuel type (Diesel/Petrol/Other): "
    ).strip().lower()

    owner = input(
        "Is this a Test Drive Car? (yes/no): "
    ).strip().lower()

    seller_type = input(
        "Seller type (Individual/Dealer): "
    ).strip().lower()

    transmission_input = input(
        "Transmission (Manual/Automatic): "
    ).strip().lower()

    car = {feature: 0 for feature in FEATURES}

    car["max_power"] = max_power
    car["torque_nm"] = torque_nm
    car["engine"] = engine
    car["year"] = year

    if brand == "BMW":
        car["brand_BMW"] = 1
    elif brand == "Mercedes-Benz":
        car["brand_Mercedes-Benz"] = 1
    elif brand == "Audi":
        car["brand_Audi"] = 1
    elif brand == "Volvo":
        car["brand_Volvo"] = 1

    if fuel == "diesel":
        car["fuel_Diesel"] = 1
    elif fuel == "petrol":
        car["fuel_Petrol"] = 1

    if owner == "yes":
        car["owner_Test Drive Car"] = 1

    if seller_type == "individual":
        car["seller_type_Individual"] = 1

    if transmission_input == "automatic":
        car["transmission"] = 1

    input_df = pd.DataFrame([car], columns=FEATURES)

    prediction = model.predict(input_df)[0]

    print("\n-------------------------------")
    print(f"Predicted selling price: {prediction:,.4f}")
    print("-------------------------------")


if __name__ == "__main__":
    predict_car_price()
