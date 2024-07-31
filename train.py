import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load data
houses = pd.read_csv('train.csv')

# Drop 'id' column if it exists
if 'id' in houses.columns:
    houses.drop(columns=['id'], inplace=True)

# Separate features and target
X = houses.drop(columns=['SalePrice'])
y = houses['SalePrice']

# Define feature types
# Ensure numeric_features only includes numeric columns
numeric_features = ['BedroomAbvGr', 'FullBath', 'YearBuilt']  # Removed 'Street'

# Define categorical features (including 'Street')
categorical_features = ['Street', 'Neighborhood', 'Condition1']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess data
X_preprocessed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f'Train score: {train_score}')
print(f'Test score: {test_score}')

# Save preprocessor and model
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
