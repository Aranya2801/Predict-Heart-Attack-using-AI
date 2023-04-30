# Predict-Heart-Attack-using-AI

Heart Attack Prediction
This project uses logistic regression to predict the likelihood of a heart attack based on various risk factors.

Data
The data used in this project was obtained from the Kaggle. The dataset contains information on various risk factors such as age, gender, blood pressure, cholesterol level, etc. as well as whether or not the individual had a heart attack.

Setup
To run this project, you will need Python 3 installed on your system, as well as the following packages:

pandas
numpy
sklearn
You can install these packages using pip:

Copy code
pip install pandas numpy sklearn
Usage
To use this project, simply run the heart_attack_prediction.py script:

Copy code
python heart_attack_prediction.py
The script will perform the following steps:

Load the data from the CSV file.
Preprocess the data by converting categorical variables into numerical categories and normalizing the numerical variables.
Split the data into training and testing sets.
Train a logistic regression model on the training data.
Evaluate the model's performance on the testing data.
The accuracy of the model will be printed to the console.

Future Work
There are several ways in which this project could be expanded:

Experiment with different machine learning algorithms to see if they can achieve better accuracy.
Collect more data to increase the size of the dataset.
Investigate the importance of each feature by performing feature selection.
Deploy the model in a web application or mobile app.
Credits
This project was created by ARANYA GHOSH. The data used in this project was obtained from the Kaggle.
