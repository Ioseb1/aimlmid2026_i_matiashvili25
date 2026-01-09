"""
Script to test individual email classification
Usage: python test_email.py
"""

import pandas as pd
import pickle
import sys
from spam_detection import train_model, classify_email, load_data
from sklearn.model_selection import train_test_split

def load_or_train_model():
    """
    Load a saved model or train a new one.
    For this assignment, we'll train a fresh model each time.
    """
    # Load data
    data_file = 'data/i_matiashvili25_54376.csv'
    df = load_data(data_file)
    
    # Prepare features and target
    feature_columns = ['words', 'links', 'capital_words', 'spam_word_count']
    X = df[feature_columns]
    y = df['is_spam']
    
    # Split data: 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    return model


def test_email_interactive():
    """
    Interactive function to test email classification.
    """
    print("="*60)
    print("Spam Email Detection - Email Classifier")
    print("="*60)
    
    # Load or train model
    model = load_or_train_model()
    
    print("\n" + "="*60)
    print("Enter email text to classify (type 'quit' to exit)")
    print("="*60)
    
    while True:
        print("\n" + "-"*60)
        email_text = input("Enter email text (or 'quit' to exit):\n")
        
        if email_text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not email_text.strip():
            print("Please enter some text.")
            continue
        
        # Classify email
        result = classify_email(model, email_text)
        
        print("\n" + "="*60)
        print("Classification Result:")
        print("="*60)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Spam): {result['probability_spam']:.4f} ({result['probability_spam']*100:.2f}%)")
        print(f"Confidence (Legitimate): {result['probability_legitimate']:.4f} ({result['probability_legitimate']*100:.2f}%)")
        print(f"\nExtracted Features:")
        print(f"  Word Count: {result['features']['words']}")
        print(f"  Link Count: {result['features']['links']}")
        print(f"  Capital Words: {result['features']['capital_words']}")
        print(f"  Spam Word Count: {result['features']['spam_word_count']}")
        print("="*60)


def test_example_emails():
    """
    Test the example spam and legitimate emails.
    """
    print("="*60)
    print("Testing Example Emails")
    print("="*60)
    
    # Load or train model
    model = load_or_train_model()
    
    # Example Spam Email
    spam_email = """
    Subject: URGENT! CLAIM YOUR $1,000,000 PRIZE NOW!
    
    CONGRATULATIONS WINNER!
    
    You have been SELECTED to receive $1,000,000 CASH PRIZE!
    This is a LIMITED TIME OFFER that expires TODAY!
    
    CLICK HERE NOW: www.freemoney.com/claim
    Visit https://prizewinner.com/urgent to claim your FREE money!
    
    ACT FAST! This offer is GUARANTEED but LIMITED TIME!
    No obligation, RISK FREE! Click www.claimprize.net NOW!
    
    Don't miss this AMAZING DEAL! Order now and SAVE BIG!
    """
    
    # Example Legitimate Email
    legitimate_email = """
    Subject: Meeting Reminder - Project Discussion
    
    Hi Team,
    
    This is a reminder about our scheduled meeting tomorrow at 2:00 PM 
    to discuss the project progress and next steps.
    
    Please prepare the following:
    - Status update on your assigned tasks
    - Any questions or concerns you'd like to address
    - Suggestions for improvement
    
    The meeting will be held in Conference Room B. If you have any 
    questions, please let me know.
    
    Best regards,
    John Smith
    Project Manager
    """
    
    print("\n" + "="*60)
    print("Example 1: Spam Email")
    print("="*60)
    print("Email Text:")
    print(spam_email)
    print("\nClassification Result:")
    result = classify_email(model, spam_email)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence (Spam): {result['probability_spam']:.4f} ({result['probability_spam']*100:.2f}%)")
    print(f"Features: {result['features']}")
    
    print("\n" + "="*60)
    print("Example 2: Legitimate Email")
    print("="*60)
    print("Email Text:")
    print(legitimate_email)
    print("\nClassification Result:")
    result = classify_email(model, legitimate_email)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence (Legitimate): {result['probability_legitimate']:.4f} ({result['probability_legitimate']*100:.2f}%)")
    print(f"Features: {result['features']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        test_example_emails()
    else:
        test_email_interactive()
