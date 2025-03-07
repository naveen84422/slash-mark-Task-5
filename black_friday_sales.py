import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load Dataset
data = pd.read_csv('black_friday_sales.csv')  # Replace with actual dataset path

# Display first few rows
data.head()

# Check for missing values
data.fillna(method='ffill', inplace=True)

# Convert categorical features to numerical
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Text Mining for Product Categories (if applicable)
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', str(text).lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

if 'Product_Category' in data.columns:
    data['Product_Category'] = data['Product_Category'].astype(str).apply(clean_text)

# Feature Selection
features = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'Product_Category']  # Adjust based on dataset
target = 'Purchase'
X = data[features]
y = data[target]

# Handle Text Data (if present)
if 'Product_Category' in X.columns:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    text_features = vectorizer.fit_transform(X['Product_Category']).toarray()
    X = X.drop(columns=['Product_Category'])
    X = np.hstack((X, text_features))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Machine Learning Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Final Model with Best Params
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Final Evaluation
print(f'Final R2 Score: {r2_score(y_test, y_pred_best)}')

# Feature Importance
feature_importances = best_model.feature_importances_
plt.bar(features[:-1], feature_importances[:len(features)-1])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Black Friday Sales Prediction')
plt.show()
