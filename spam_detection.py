"""
Spam Email Detection Application using Logistic Regression
This application trains a logistic regression model to classify emails as spam or legitimate.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Common spam words/phrases
SPAM_WORDS = [
    'free', 'click', 'now', 'urgent', 'winner', 'congratulations', 'prize', 'limited',
    'offer', 'deal', 'discount', 'save', 'act now', 'buy now', 'order now', 'guaranteed',
    'risk free', 'no obligation', 'cash', 'money', 'million', 'dollar', 'profit',
    'investment', 'loan', 'credit', 'debt', 'refinance', 'viagra', 'cialis', 'pills',
    'weight loss', 'lose weight', 'diet', 'miracle', 'cure', 'guarantee', 'winner',
    'selected', 'winner', 'claim', 'trial', 'subscription', 'cancel', 'unsubscribe',
    'click here', 'visit now', 'limited time', 'expires', 'today only', 'act fast'
]


def load_data(file_path):
    """
    Load the spam email dataset from CSV file.
    
    Parameters:
    file_path: str, path to the CSV file
    
    Returns:
    df: pandas DataFrame containing the data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData info:")
    print(df.info())
    print(f"\nClass distribution:")
    print(df['is_spam'].value_counts())
    print(f"\nClass distribution (percentage):")
    print(df['is_spam'].value_counts(normalize=True) * 100)
    return df


def extract_features_from_email(email_text):
    """
    Extract features from email text that match the training data features:
    - words: total word count
    - links: number of links/URLs
    - capital_words: number of words in all capital letters
    - spam_word_count: count of spam-related words
    
    Parameters:
    email_text: str, the email text to analyze
    
    Returns:
    dict: dictionary containing the extracted features
    """
    if not email_text or not isinstance(email_text, str):
        return {
            'words': 0,
            'links': 0,
            'capital_words': 0,
            'spam_word_count': 0
        }
    
    # Count total words (split by whitespace and filter out empty strings)
    words = re.findall(r'\b\w+\b', email_text.lower())
    word_count = len(words)
    
    # Count links/URLs (http, https, www, or common URL patterns)
    link_patterns = [
        r'https?://\S+',
        r'www\.\S+',
        r'\b\w+\.(com|org|net|edu|gov|io|co|uk)\b',
    ]
    links = []
    for pattern in link_patterns:
        links.extend(re.findall(pattern, email_text, re.IGNORECASE))
    # Remove duplicates
    links = list(set(links))
    link_count = len(links)
    
    # Count words in all capital letters (at least 2 characters)
    capital_words = re.findall(r'\b[A-Z]{2,}\b', email_text)
    capital_word_count = len(capital_words)
    
    # Count spam words (case-insensitive)
    email_lower = email_text.lower()
    spam_word_count = sum(1 for spam_word in SPAM_WORDS if spam_word in email_lower)
    
    return {
        'words': word_count,
        'links': link_count,
        'capital_words': capital_word_count,
        'spam_word_count': spam_word_count
    }


def train_model(X_train, y_train):
    """
    Train a logistic regression model on the training data.
    
    Parameters:
    X_train: pandas DataFrame, training features
    y_train: pandas Series, training labels
    
    Returns:
    model: trained LogisticRegression model
    """
    print("\n" + "="*60)
    print("Training Logistic Regression Model")
    print("="*60)
    
    # Create and train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")
    print(f"\nModel coefficients:")
    feature_names = X_train.columns.tolist()
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {model.coef_[0][i]:.6f}")
    print(f"\nIntercept: {model.intercept_[0]:.6f}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and return metrics.
    
    Parameters:
    model: trained LogisticRegression model
    X_test: pandas DataFrame, test features
    y_test: pandas Series, test labels
    
    Returns:
    dict: dictionary containing accuracy, confusion matrix, and classification report
    """
    print("\n" + "="*60)
    print("Model Evaluation on Test Data")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam'])
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred
    }


def create_visualizations(df, model, evaluation_results, output_dir='.'):
    """
    Create visualizations for the dataset and model performance.
    
    Parameters:
    df: pandas DataFrame, the dataset
    model: trained LogisticRegression model
    evaluation_results: dict, results from evaluate_model
    output_dir: str, directory to save visualizations
    """
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    # Visualization 1: Class Distribution
    plt.figure(figsize=(10, 6))
    
    # Bar chart
    plt.subplot(1, 2, 1)
    class_counts = df['is_spam'].value_counts()
    colors = ['#3498db', '#e74c3c']
    bars = plt.bar(['Legitimate', 'Spam'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Email Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Class Distribution: Spam vs Legitimate Emails', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(class_counts.values, labels=['Legitimate', 'Spam'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: class_distribution.png")
    plt.close()
    
    # Visualization 2: Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    cm = evaluation_results['confusion_matrix']
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['Legitimate', 'Spam'],
                yticklabels=['Legitimate', 'Spam'],
                linewidths=2, linecolor='black', square=True)
    
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix Heatmap', fontsize=16, fontweight='bold')
    
    # Add accuracy text
    accuracy = evaluation_results['accuracy']
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
             transform=plt.gca().transAxes, ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_heatmap.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix_heatmap.png")
    plt.close()
    
    # Visualization 3: Feature Importance (Coefficients)
    plt.figure(figsize=(10, 6))
    feature_names = ['words', 'links', 'capital_words', 'spam_word_count']
    coefficients = model.coef_[0]
    
    colors_bar = ['red' if x < 0 else 'green' for x in coefficients]
    bars = plt.barh(feature_names, coefficients, color=colors_bar, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('Logistic Regression Feature Coefficients', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, coef) in enumerate(zip(bars, coefficients)):
        plt.text(coef, i, f' {coef:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_coefficients.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_coefficients.png")
    plt.close()
    
    print("\nAll visualizations created successfully!")


def classify_email(model, email_text):
    """
    Classify a single email as spam or legitimate.
    
    Parameters:
    model: trained LogisticRegression model
    email_text: str, the email text to classify
    
    Returns:
    dict: dictionary containing prediction, probability, and extracted features
    """
    # Extract features
    features = extract_features_from_email(email_text)
    
    # Create DataFrame with same column order as training data
    feature_df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    probability = model.predict_proba(feature_df)[0]
    
    result = {
        'prediction': 'Spam' if prediction == 1 else 'Legitimate',
        'probability_spam': probability[1],
        'probability_legitimate': probability[0],
        'features': features
    }
    
    return result


def main():
    """
    Main function to run the spam detection application.
    """
    # File path
    data_file = 'data/i_matiashvili25_54376.csv'
    
    # Load data
    df = load_data(data_file)
    
    # Prepare features and target
    feature_columns = ['words', 'links', 'capital_words', 'spam_word_count']
    X = df[feature_columns]
    y = df['is_spam']
    
    # Split data: 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Create visualizations
    create_visualizations(df, model, evaluation_results, output_dir='.')
    
    # Example: Classify a sample email
    print("\n" + "="*60)
    print("Example Email Classification")
    print("="*60)
    
    example_email = """
    Subject: Congratulations! You've Won $1,000,000!
    
    Dear Winner,
    
    CLICK HERE NOW to claim your prize! This is a LIMITED TIME OFFER!
    Visit www.freeprizes.com NOW to get your FREE money!
    
    ACT FAST! This offer expires TODAY!
    """
    
    result = classify_email(model, example_email)
    print(f"\nExample Email Classification:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability (Spam): {result['probability_spam']:.4f}")
    print(f"Probability (Legitimate): {result['probability_legitimate']:.4f}")
    print(f"\nExtracted Features:")
    for key, value in result['features'].items():
        print(f"  {key}: {value}")
    
    return model, evaluation_results


if __name__ == "__main__":
    model, results = main()
