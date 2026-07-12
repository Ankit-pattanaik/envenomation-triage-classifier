"""
data_prep.py
------------
Single source of truth for preprocessing. Both train_models.py and the
Streamlit app import from here, so a patient scored in the UI is transformed
EXACTLY the way the training rows were. That consistency is what makes the
saved models trustworthy at inference time.

Key domain decisions baked in:
  * Local_Swelling is missing IFF the culprit is a Black Widow Spider
    (latrodectism -> systemic rigidity, minimal local reaction). We therefore
    KEEP those rows and turn the missingness into two explicit signals:
        - Swelling_Missing  (binary flag: 1 when no swelling was recorded)
        - Swelling_Level    (ordinal: None=0 < Mild=1 < Medium=2 < Severe=3)
  * Age and Time_Since_Bite_Min are statistically flat across all classes
    (pure noise) but we keep them so the models can learn to ignore them.
  * Patient_ID is dropped (an identifier, never a feature).
"""

import numpy as np
import pandas as pd

TARGET = "Bite_Source_Target"

# Fixed feature order — the app rebuilds a single patient row in THIS order.
FEATURE_ORDER = [
    "Age",
    "Time_Since_Bite_Min",
    "Heart_Rate_BPM",
    "Blood_Pressure_Systolic",
    "Muscle_Paralysis_Present",
    "Blood_Coagulation_Failure",
    "Swelling_Missing",
    "Swelling_Level",
    "Gender_Male",
    "Gender_Female",
]

# Ordinal map for swelling severity. Missing -> "None" -> 0.
SWELLING_ORDER = {"None": 0, "Mild": 1, "Medium": 2, "Severe": 3}

# Stable class ordering so label integers mean the same thing everywhere.
CLASS_ORDER = ["Harmless_Insect", "Scorpion", "Viper_Snake", "Black_Widow_Spider"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into the numeric feature matrix (no target)."""
    out = pd.DataFrame(index=df.index)

    out["Age"] = df["Age"].astype(float)
    out["Time_Since_Bite_Min"] = df["Time_Since_Bite_Min"].astype(float)
    out["Heart_Rate_BPM"] = df["Heart_Rate_BPM"].astype(float)
    out["Blood_Pressure_Systolic"] = df["Blood_Pressure_Systolic"].astype(float)

    out["Muscle_Paralysis_Present"] = df["Muscle_Paralysis_Present"].astype(int)
    out["Blood_Coagulation_Failure"] = df["Blood_Coagulation_Failure"].astype(int)

    # --- the signal that matters most ---
    swelling = df["Local_Swelling"]
    out["Swelling_Missing"] = swelling.isna().astype(int)
    out["Swelling_Level"] = (
        swelling.fillna("None").map(SWELLING_ORDER).fillna(0).astype(int)
    )

    # Gender one-hot (Other == both zero, so no dummy trap)
    out["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    out["Gender_Female"] = (df["Gender"] == "Female").astype(int)

    return out[FEATURE_ORDER]


def load_and_prepare(csv_path: str):
    """Read the CSV and return (X, y, raw_df) ready for modelling."""
    df = pd.read_csv(csv_path)
    X = build_features(df)
    y = df[TARGET].map(CLASS_TO_INT).astype(int)
    return X, y, df


def row_from_inputs(
    age, time_since_bite, heart_rate, bp_systolic,
    muscle_paralysis, coag_failure, gender, swelling_choice,
):
    """
    Build a single-row feature frame from raw UI inputs.
    `swelling_choice` is one of: 'None (not recorded)', 'Mild', 'Medium', 'Severe'.
    Selecting 'None (not recorded)' sets the Swelling_Missing flag — the
    Black-Widow signature.
    """
    is_missing = swelling_choice.startswith("None")
    level_key = "None" if is_missing else swelling_choice
    row = {
        "Age": float(age),
        "Time_Since_Bite_Min": float(time_since_bite),
        "Heart_Rate_BPM": float(heart_rate),
        "Blood_Pressure_Systolic": float(bp_systolic),
        "Muscle_Paralysis_Present": int(muscle_paralysis),
        "Blood_Coagulation_Failure": int(coag_failure),
        "Swelling_Missing": int(is_missing),
        "Swelling_Level": int(SWELLING_ORDER[level_key]),
        "Gender_Male": int(gender == "Male"),
        "Gender_Female": int(gender == "Female"),
    }
    return pd.DataFrame([row])[FEATURE_ORDER]
