# Clinic No-Show Prediction

Predicting patient no-shows for clinic appointments using exploratory data analysis, statistical testing, and machine learning.

## Project Overview

Missed appointments cost clinics time and resources. This project explores synthetic clinic appointment data to understand which factors influence patient attendance and builds a simple predictive model.

**Key questions explored:**
- Are certain time slots preferred over others?
- Does the booking method (Phone, Web, Mobile App) affect attendance rates?
- Can we predict whether a patient will show up?

## Features

- Synthetic data generation for clinic appointments
- Exploratory Data Analysis (EDA) with visualizations
- Statistical tests (Chi-square Goodness of Fit & Independence)
- Logistic Regression model to predict attendance

## Project Structure
```
Clinic_NoShow_Prediction/
├── data/
│   └── clinic_appointments.csv
├── clinic_analysis.ipynb      # Main analysis notebook
├── clinic_analysis.py         # Script version of the analysis
├── requirements.txt
├── .gitignore
└── LICENSE
```
## Installation

1. Clone the repository:
```bash
git clone https://github.com/ewangila/Clinic_NoShow_Prediction.git
cd Clinic_NoShow_Prediction
```
2. Install dependencies:

```Bash
pip install -r requirements.txt
```
## Usage
Option 1: Jupyter Notebook (recommended)
```Bashj
upyter notebook clinic_analysis.ipynb
```
Option 2: Python Script
```Bash
python clinic_analysis.py
```

## Technologies Used

- **Python**
- **pandas** & **numpy** – data manipulation
- **matplotlib** & **seaborn** – visualization
- **scipy** – statistical testing
- **scikit-learn** – logistic regression model

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
