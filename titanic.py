#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
def load_data():
    file_path = "Titanic-Dataset.csv"  # Direct file name without path
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Handle missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])  # Male=1, Female=0
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

    return data

# Split the data into training and testing sets
def split_data(data):
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and train the model
def build_and_train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Analyze feature importance
def plot_feature_importance(model, data):
    feature_importance = model.feature_importances_
    features = data.drop('Survived', axis=1).columns

    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.show()

# Main function
def main():
    print("Loading data...")
    data = load_data()
    
    print("Preprocessing data...")
    data = preprocess_data(data)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data)
    
    print("Building and training the model...")
    model = build_and_train_model(X_train, y_train)
    
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    
    print("Analyzing feature importance...")
    plot_feature_importance(model, data)

if __name__ == "__main__":
    main()


# In[ ]:




