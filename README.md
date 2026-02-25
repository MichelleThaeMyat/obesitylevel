# Obesity Level Prediction

A machine learning web application that predicts obesity levels based on eating habits, physical activity, and lifestyle factors.

## Overview

This project uses a Random Forest classifier to predict one of seven obesity categories:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

## Features

The model considers multiple input features:

**Physical Attributes:**

- Age, Height, Weight, BMI (auto-calculated)

**Eating Habits:**

- FAVC - Frequent consumption of high caloric food
- FCVC - Frequency of vegetable consumption
- NCP - Number of main meals
- CAEC - Consumption of food between meals
- CH2O - Daily water consumption
- CALC - Alcohol consumption

**Lifestyle Factors:**

- SCC - Calorie consumption monitoring
- FAF - Physical activity frequency
- TUE - Time using technology devices
- SMOKE - Smoking status
- MTRANS - Transportation used
- Family history with overweight

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/obesitylevel.git
cd obesitylevel
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
obesitylevel/
├── app.py                              # Streamlit web application
├── obesity_best_model.joblib           # Trained Random Forest model
├── schema.json                         # Feature schema and configuration
├── Obesity_Level_Prediction_Project.ipynb  # Jupyter notebook with EDA and model training
├── requirements.txt                    # Python dependencies
├── LICENSE                             # License file
└── README.md                           # This file
```

## Requirements

- Python 3.8+
- streamlit >= 1.30.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn == 1.6.1
- joblib >= 1.3.0
- matplotlib >= 3.7.0

## Disclaimer

This application is for **educational purposes only** and should not be used as medical advice. Consult a healthcare professional for health-related decisions.

## License

See [LICENSE](LICENSE) for details.
