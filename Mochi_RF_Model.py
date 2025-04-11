import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, make_scorer, f1_score

# Load data
df = pd.read_csv('mochi_activities.csv')

# Feature engineering
df['hour_sin'] = np.sin(2 * np.pi * df['Time']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['Time']/24)
df['is_weekend'] = df['Day'].isin(['Saturday', 'Sunday']).astype(int)

# Define features/target
X = df[['hour_sin', 'hour_cos', 'Duration_minutes', 'Location', 
        'Weather', 'People_home', 'Mood', 'Trigger', 'Reward_given', 'is_weekend']]
y = df['Activity']

# Preprocessing
categorical_features = ['Location', 'Weather', 'Mood', 'Trigger', 'Reward_given']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create base pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__max_depth': [5, 8, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__max_features': ['sqrt', 'log2']
}

# Custom scoring (weighted F1-score)
scorer = make_scorer(f1_score, average='weighted')

# Initialize Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Run Grid Search
grid_search.fit(X_train, y_train)

# Results
print("Best parameters:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save best model
import joblib
joblib.dump(best_model, 'optimized_mochi_rf_model.pkl')