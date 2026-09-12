import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# 1. DATA GENERATION

def generate_synthetic_data(filepath='clinic_appointments.csv', n_records=1000):
    """Generates synthetic clinic data and saves it to a CSV."""
    np.random.seed(42)
    data = {
        'TimeSlot': np.random.choice(['Morning', 'Afternoon', 'Evening'], n_records, p=[0.45, 0.35, 0.20]),
        'BookingMethod': np.random.choice(['Phone', 'Web', 'Mobile App'], n_records),
        'Attended': np.random.choice(['Yes', 'No'], n_records, p=[0.75, 0.25])
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"[*] Synthetic data saved to {filepath}")
    return df


# 2. EXPLORATORY DATA ANALYSIS (EDA)

def perform_eda(df):
    """Generates visualizations for the clinic data."""
    print("\n[*] Running Exploratory Data Analysis...")
    df['Attended_Num'] = df['Attended'].map({'Yes': 1, 'No': 0})
    
    plt.figure(figsize=(15, 5))

    # Time Slot Distribution
    plt.subplot(1, 3, 1)
    sns.countplot(data=df, x='TimeSlot', order=['Morning', 'Afternoon', 'Evening'], palette='viridis')
    plt.title('Time Slot Distribution')

    # Booking Method Distribution
    plt.subplot(1, 3, 2)
    sns.countplot(data=df, x='BookingMethod', palette='viridis')
    plt.title('Booking Method Distribution')

    # Attendance Rate by Booking Method
    plt.subplot(1, 3, 3)
    sns.barplot(data=df, x='BookingMethod', y='Attended_Num', palette='viridis')
    plt.title('Attendance Rate by Booking Method')
    plt.ylabel('Attendance Rate (%)')

    plt.tight_layout()
    plt.show()


# 3. STATISTICAL TESTING

def run_statistical_tests(df, alpha=0.05):
    """Runs Chi-Square Goodness of Fit and Independence tests."""
    print("\n[*] Running Statistical Tests...")
    
    # 1. Goodness of Fit (Time Slot Preferences)
    n_slots = df['TimeSlot'].nunique()
    expected_freq = np.array([len(df) / n_slots] * n_slots)
    observed_freq = df['TimeSlot'].value_counts()[['Morning', 'Afternoon', 'Evening']].values
    
    chi2_gof, p_gof = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
    print(f"\n--- Goodness of Fit Test (Time Slots) ---")
    print(f"P-value: {p_gof:.4f} -> {'Significant Preference' if p_gof < alpha else 'No Preference'}")

    # 2. Test of Independence (Booking Method vs Attendance)
    contingency_table = pd.crosstab(df['BookingMethod'], df['Attended'])
    chi2_ind, p_ind, _, _ = stats.chi2_contingency(contingency_table)
    print(f"\n--- Independence Test (Booking vs Attendance) ---")
    print(f"P-value: {p_ind:.4f} -> {'Significantly Related' if p_ind < alpha else 'No Relationship'}")


# 4. PREDICTIVE MODELING

def build_predictive_model(df):
    """Trains a Logistic Regression model to predict patient attendance."""
    print("\n[*] Training Predictive Model (Logistic Regression)...")
    
    # Feature Engineering & Encoding
    le = LabelEncoder()
    df['TimeSlot_Encoded'] = le.fit_transform(df['TimeSlot'])
    df['BookingMethod_Encoded'] = le.fit_transform(df['BookingMethod'])
    
    X = df[['TimeSlot_Encoded', 'BookingMethod_Encoded']]
    y = df['Attended'].map({'Yes': 1, 'No': 0})
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("\nModel Evaluation:")
    print("-" * 30)
    print(classification_report(y_test, predictions, target_names=['No Show (0)', 'Attended (1)']))


# MAIN EXECUTION

if __name__ == "__main__":
    # 1. Load or Generate Data
    try:
        df = pd.read_csv('clinic_appointments.csv')
        print("[*] Loaded existing dataset.")
    except FileNotFoundError:
        df = generate_synthetic_data()

    # 2. Execute Pipeline
    perform_eda(df)
    run_statistical_tests(df)
    build_predictive_model(df)