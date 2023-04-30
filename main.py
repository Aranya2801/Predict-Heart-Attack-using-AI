# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the heart attack dataset into a pandas dataframe
data = pd.read_csv('heart_attack_dataset.csv')

# Perform EDA and Feature Engineering
# Calculate age squared
data['age_squared'] = data['age'] ** 2

# Categorize resting blood pressure
data['rest_bp_cat'] = pd.cut(data['rest_bp'], bins=[0, 120, 140, np.inf], labels=['normal', 'prehypertension', 'hypertension'])

# Categorize maximum heart rate achieved
data['max_hr_cat'] = pd.cut(data['max_hr'], bins=[0, 100, 150, np.inf], labels=['low', 'moderate', 'high'])

# Categorize cholesterol
data['cholesterol_cat'] = pd.cut(data['cholesterol'], bins=[0, 200, 240, np.inf], labels=['normal', 'high', 'very high'])

# Categorize alcohol consumption
data['alcohol_consumption_cat'] = pd.cut(data['alcohol_consumption'], bins=[0, 2, 4, np.inf], labels=['none', 'moderate', 'excessive'])

# Define the features and target variable
features = ['age', 'gender', 'chest_pain_type', 'rest_bp_cat', 'cholesterol_cat', 'fasting_bs', 'rest_ecg', 'max_hr_cat', 'exercise_induced_angina', 'st_depression', 'num_major_vessels', 'thalassemia_type', 'smoking', 'bmi', 'diabetes', 'exercise_angina', 'family_history', 'alcohol_consumption_cat', 'crp', 'homocysteine', 'sbp_variability', 'dbp_variability', 'depression', 'age_squared', 'fasting_bs_variability', 'waist_circumference_cat', 'physical_activity_hours', 'poor_sleep_quality', 'medication_use', 'vitamin_d_cat', 'magnesium_cat']
target = 'heart_attack'

# Add additional features
data['fasting_bs_variability'] = data.groupby('patient_id')['fasting_bs'].apply(lambda x: x.diff().std()).fillna(0)
data['waist_circumference_cat'] = pd.cut(data['waist_circumference'], bins=[0, 80, 94, np.inf], labels=['normal', 'high', 'very high'])
data['physical_activity_hours'] = data['physical_activity_moderate'] + data['physical_activity_vigorous']
data['poor_sleep_quality'] = data['sleep_quality'] > 5 # 5 is the cutoff for poor sleep quality on the Pittsburgh Sleep Quality Index
data['medication_use'] = data['statins'] | data['beta_blockers']
data['vitamin_d_cat'] = pd.cut(data['vitamin_d'], bins=[-np.inf, 20, 30, np.inf], labels=['deficient', 'insufficient', 'sufficient'])
data['magnesium_cat'] = pd.cut(data['magnesium'], bins=[-np.inf, 1.8, 2.3, np.inf], labels=['deficient', 'normal', 'excess'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.3, random_state=42)

#Train the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

#Make predictions on the test set
y_pred = lr.predict(X_test)

#Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

