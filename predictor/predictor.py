import os
from datetime import datetime

import joblib
import pandas as pd

# --------------------------------------------------
# Model Path
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "champion_model.pkl"
)

# --------------------------------------------------
# Load Model Once
# --------------------------------------------------


champion = None



delay_model = None
cost_model = None
encoders = None

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_for_project(project_data):
    """
    Predict cumulative delay and cumulative cost overrun.
    """

    df = pd.DataFrame([project_data])

    # --------------------------------------------------
    # State Encoding
    # --------------------------------------------------

    if (
        "Project_Location" in df.columns
        and "State_Encoded" not in df.columns
    ):

        df["State"] = df["Project_Location"].apply(
            lambda x: str(x).split(",")[0]
            if pd.notna(x)
            else "Unknown"
        )

        if "State" in encoders["label_encoders"]:

            try:

                df["State_Encoded"] = (
                    encoders["label_encoders"]["State"]
                    .transform(df["State"])
                )

            except ValueError:

                df["State_Encoded"] = 0

        else:

            df["State_Encoded"] = 0

        df.drop(
            columns=["State"],
            inplace=True,
            errors="ignore"
        )

    # --------------------------------------------------
    # Days Since Start
    # --------------------------------------------------

    if (
        "Start_Date" in df.columns
        and "Days_Since_Start" not in df.columns
    ):

        df["Start_Date"] = pd.to_datetime(
            df["Start_Date"],
            format="%d-%m-%Y",
            errors="coerce"
        )

        df["Days_Since_Start"] = (
            datetime.now() - df["Start_Date"]
        ).dt.days

        df.drop(
            columns=["Start_Date"],
            inplace=True,
            errors="ignore"
        )

    # --------------------------------------------------
    # Remove Unused Columns
    # --------------------------------------------------

    df.drop(
        columns=[
            "Project_Location",
            "DOCO_Date",
            "Unnamed: 62"
        ],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------
    # Encode Categorical Features
    # --------------------------------------------------

    label_encoders = encoders["label_encoders"]
    categorical_features = encoders["categorical_features"]

    for col in categorical_features:

        if col in df.columns and col in label_encoders:

            mask = df[col].notna()

            if mask.sum() > 0:

                try:

                    df.loc[mask, col] = (
                        label_encoders[col]
                        .transform(
                            df.loc[mask, col].astype(str)
                        )
                    )

                except ValueError:

                    df.loc[mask, col] = 0

            df[col] = df[col].astype(float)

    # --------------------------------------------------
    # Numeric Conversion
    # --------------------------------------------------

    for col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # Feature Selection
    # --------------------------------------------------
    print("Expected features:")
    print(encoders["all_features"])

    print("\nAvailable columns:")
    print(df.columns.tolist())
    features = [
            feature
            for feature in encoders["all_features"]
            if feature != "Unnamed: 62"
        ]
    X = df[features]
    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    delay = float(delay_model.predict(X)[0])
    cost = float(cost_model.predict(X)[0])

    return {
        "predicted_delay_days": round(delay, 2),
        "predicted_cost_overrun_cr": round(cost, 2)
    }
