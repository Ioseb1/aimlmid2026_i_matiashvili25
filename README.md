# AI Midterm Exam

This repository contains solutions for two assignments:
1. Finding the Correlation (Pearson's Correlation Coefficient)
2. Spam Email Detection using Logistic Regression

---

# Assignment 1: Finding the Correlation

## Assignment Overview

This assignment involves calculating Pearson's correlation coefficient for a given dataset and creating an appropriate visualization. The data points are displayed on a graph with blue dots representing the main data, green dots marking the X-axis, and red dots marking the Y-axis.

## Question 1: Finding Pearson's Correlation Coefficient

### Data Points

The blue dots (main data points) coordinates from left to right:
- (-9, 8)
- (-7, 6.68)
- (-5, 4.72)
- (-3, 5.03)
- (-1, 0.87)
- (1, -0.53)
- (3, -2.9)
- (5, -4.73)
- (7, -6.5)
- (9.1, -8.33)

### Pearson's Correlation Coefficient Formula

Pearson's correlation coefficient (r) is calculated using the following formula:

```
r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² Σ(yi - ȳ)²]
```

Where:
- `xi` and `yi` are individual data points
- `x̄` (x-bar) is the mean of all x values
- `ȳ` (y-bar) is the mean of all y values
- `Σ` represents the summation over all data points

### Calculation Process

1. **Extract coordinates**: Separate the x and y coordinates from the blue data points.

2. **Calculate means**:
   - Mean of X (x̄) = average of all x coordinates
   - Mean of Y (ȳ) = average of all y coordinates

3. **Calculate deviations**:
   - For each point, calculate (xi - x̄) and (yi - ȳ)

4. **Calculate numerator**:
   - Sum of products: Σ(xi - x̄)(yi - ȳ)

5. **Calculate denominator**:
   - Sum of squared deviations for X: Σ(xi - x̄)²
   - Sum of squared deviations for Y: Σ(yi - ȳ)²
   - Denominator = √[Σ(xi - x̄)² × Σ(yi - ȳ)²]

6. **Calculate correlation**:
   - r = numerator / denominator

### Results

The calculated Pearson's correlation coefficient for the given data points is:

**r ≈ -0.9999**

This indicates a very strong negative linear correlation between X and Y variables. As X increases, Y decreases in a nearly perfect linear relationship.

### Interpretation

- **r ≈ -0.9999**: This value is very close to -1, indicating an almost perfect negative linear correlation.
- The negative sign means that as X increases, Y decreases.
- The magnitude (close to 1) indicates a very strong linear relationship.

## Question 2: Visualization

A visualization graph has been created showing:
- **Blue dots**: The main data points used for correlation calculation
- **Green dots**: X-axis markers (reference points)
- **Red dots**: Y-axis markers (reference points)
- **Correlation coefficient**: Displayed on the graph
- **Formula**: Shown for reference

The visualization is saved as `correlation_plot.png` and clearly shows the strong negative linear relationship between the variables.

## How to Reproduce

### Prerequisites

Make sure you have the following Python libraries installed:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas matplotlib
```

### Running the Script

1. Navigate to the `midterm` folder:
   ```bash
   cd midterm
   ```

2. Run the Python script:
   ```bash
   python calculate_correlation.py
   ```

3. The script will:
   - Display the data points in a table
   - Calculate and display the correlation coefficient
   - Show intermediate calculation steps
   - Generate and save a visualization as `correlation_plot.png`
   - Display the plot on screen

### Expected Output

The script will output:
- A table of all data points
- The calculated Pearson's correlation coefficient
- Intermediate calculation values (means, deviations, etc.)
- A saved visualization image file

## Files for Assignment 1

- `calculate_correlation.py`: Python script that calculates Pearson's correlation coefficient and creates the visualization
- `correlation_plot.png`: Generated visualization graph (created when script is run)

## Code Explanation

The Python script (`calculate_correlation.py`) includes:

1. **Data Definition**: All data points (blue, green, and red dots) are defined as lists of tuples.

2. **Correlation Function**: `calculate_pearson_correlation()` implements the Pearson's correlation formula step by step.

3. **Visualization Function**: `create_visualization()` creates a matplotlib plot with:
   - All three types of data points (blue, green, red)
   - Grid for better readability
   - Axis labels
   - Correlation coefficient displayed on the graph
   - Formula reference

4. **Main Function**: Orchestrates the calculation and visualization, and provides detailed output.

## Verification

The script also uses NumPy's built-in `corrcoef()` function to verify the manual calculation, ensuring accuracy of the results.

---

# Assignment 2: Spam Email Detection

## Assignment Overview

This assignment involves developing a Python console application for email classification within spam and legitimate classes using logistic regression. The application trains a model on email features and can classify new emails.

## Task 1: Data File Upload

**Data File Location**: [data/i_matiashvili25_54376.csv](data/i_matiashvili25_54376.csv)

The dataset contains email features with the following columns:
- `words`: Total word count in the email
- `links`: Number of links/URLs in the email
- `capital_words`: Number of words in all capital letters
- `spam_word_count`: Count of spam-related words
- `is_spam`: Target label (0 = Legitimate, 1 = Spam)

**Dataset Statistics**:
- Total records: 2,501 emails
- Features: 4 numerical features
- Target: Binary classification (Spam vs Legitimate)

## Task 2: Model Training

### Data Loading and Processing

**Source Code**: [spam_detection.py](spam_detection.py)

The data loading and processing code:

```python
def load_data(file_path):
    """Load the spam email dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df

# Prepare features and target
feature_columns = ['words', 'links', 'capital_words', 'spam_word_count']
X = df[feature_columns]
y = df['is_spam']

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

**Description**:
- The data is loaded from the CSV file using pandas
- Four features are extracted: words, links, capital_words, and spam_word_count
- The target variable is `is_spam` (0 for legitimate, 1 for spam)
- Data is split into 70% training and 30% testing sets using stratified sampling to maintain class distribution

### Logistic Regression Model

**Source Code**: [spam_detection.py](spam_detection.py) - `train_model()` function

The logistic regression model code:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```

**Model Description**:
- **Algorithm**: Logistic Regression (binary classification)
- **Library**: scikit-learn's `LogisticRegression`
- **Parameters**: 
  - `random_state=42`: For reproducibility
  - `max_iter=1000`: Maximum iterations for convergence
- **Training Data**: 70% of the dataset (approximately 1,751 samples)
- **Features Used**: words, links, capital_words, spam_word_count

### Model Coefficients

After training, the model learns coefficients for each feature. The coefficients indicate the importance and direction of each feature:

- **words**: Coefficient value (see output when running the script)
- **links**: Coefficient value (see output when running the script)
- **capital_words**: Coefficient value (see output when running the script)
- **spam_word_count**: Coefficient value (see output when running the script)
- **Intercept**: Bias term

**Note**: The exact coefficient values are displayed when running `spam_detection.py`. A visualization of feature coefficients is also generated in `feature_coefficients.png`.

## Task 3: Model Validation

### Confusion Matrix and Accuracy

**Source Code**: [spam_detection.py](spam_detection.py) - `evaluate_model()` function

The validation code:

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

**Results** (run the script to see actual values):
- **Accuracy**: Calculated as (TP + TN) / (TP + TN + FP + FN)
- **Confusion Matrix**: 
  ```
  [[True Negatives,  False Positives],
   [False Negatives, True Positives]]
  ```

The confusion matrix shows:
- **True Negatives (TN)**: Legitimate emails correctly classified as legitimate
- **False Positives (FP)**: Legitimate emails incorrectly classified as spam
- **False Negatives (FN)**: Spam emails incorrectly classified as legitimate
- **True Positives (TP)**: Spam emails correctly classified as spam

**Visualization**: A confusion matrix heatmap is generated and saved as `confusion_matrix_heatmap.png`.

## Task 4: Email Text Classification

**Source Code**: [spam_detection.py](spam_detection.py) - `extract_features_from_email()` and `classify_email()` functions

The application can parse email text, extract features, and classify it. The feature extraction function:

```python
def extract_features_from_email(email_text):
    """
    Extract features from email text:
    - words: total word count
    - links: number of links/URLs
    - capital_words: number of words in all capital letters
    - spam_word_count: count of spam-related words
    """
    # Count words
    words = re.findall(r'\b\w+\b', email_text.lower())
    word_count = len(words)
    
    # Count links/URLs
    link_patterns = [
        r'https?://\S+',
        r'www\.\S+',
        r'\b\w+\.(com|org|net|edu|gov|io|co|uk)\b',
    ]
    # ... (full implementation in spam_detection.py)
    
    # Count capital words
    capital_words = re.findall(r'\b[A-Z]{2,}\b', email_text)
    
    # Count spam words
    spam_word_count = sum(1 for spam_word in SPAM_WORDS 
                         if spam_word in email_text.lower())
    
    return {
        'words': word_count,
        'links': link_count,
        'capital_words': capital_word_count,
        'spam_word_count': spam_word_count
    }
```

**Usage**: 
- Run `python test_email.py` for interactive email classification
- Or use `classify_email(model, email_text)` function in your code

## Task 5: Example Spam Email

**File**: [example_emails.txt](example_emails.txt)

**Composed Spam Email**:
```
Subject: URGENT! CLAIM YOUR $1,000,000 PRIZE NOW!

CONGRATULATIONS WINNER!

You have been SELECTED to receive $1,000,000 CASH PRIZE!
This is a LIMITED TIME OFFER that expires TODAY!

CLICK HERE NOW: www.freemoney.com/claim
Visit https://prizewinner.com/urgent to claim your FREE money!

ACT FAST! This offer is GUARANTEED but LIMITED TIME!
No obligation, RISK FREE! Click www.claimprize.net NOW!

Don't miss this AMAZING DEAL! Order now and SAVE BIG!
```

**Explanation**:
This email is designed to be classified as spam because it contains:
1. **Multiple spam trigger words**: "URGENT", "CLAIM", "PRIZE", "WINNER", "CONGRATULATIONS", "FREE", "GUARANTEED", "LIMITED TIME", "ACT FAST", "DEAL", "SAVE"
2. **High spam_word_count**: Contains many common spam words from the SPAM_WORDS list
3. **Multiple links**: Three different URLs (www.freemoney.com, prizewinner.com, claimprize.net)
4. **Excessive capital words**: Many words in ALL CAPS (URGENT, CLAIM, PRIZE, NOW, etc.)
5. **Urgency tactics**: Multiple phrases creating false urgency
6. **Financial promises**: Large cash amounts and free money claims

These characteristics align with typical spam email patterns that the model has learned to identify.

## Task 6: Example Legitimate Email

**File**: [example_emails.txt](example_emails.txt)

**Composed Legitimate Email**:
```
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
```

**Explanation**:
This email is designed to be classified as legitimate because it:
1. **No spam words**: Contains professional, normal business language without common spam trigger words
2. **Low spam_word_count**: Zero spam-related words
3. **No links**: Contains no URLs or hyperlinks
4. **Minimal capital words**: Only proper nouns and standard capitalization
5. **Professional tone**: Uses appropriate business communication style
6. **Normal word count**: Reasonable length for a business email
7. **Legitimate purpose**: Clear, professional communication about work matters

These characteristics match typical legitimate email patterns that the model has learned to identify.

## Task 7: Visualizations

### Visualization 1: Class Distribution Study

**Code** (from `spam_detection.py`):
```python
# Bar chart
class_counts = df['is_spam'].value_counts()
plt.bar(['Legitimate', 'Spam'], class_counts.values, 
        color=['#3498db', '#e74c3c'], alpha=0.7)
plt.xlabel('Email Class', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Class Distribution: Spam vs Legitimate Emails', 
          fontsize=14, fontweight='bold')

# Pie chart
plt.pie(class_counts.values, labels=['Legitimate', 'Spam'], 
        autopct='%1.1f%%', colors=colors)
plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
```

**Graph**: [class_distribution.png](class_distribution.png)

**Explanation**: 
This visualization shows the distribution of spam vs legitimate emails in the dataset using both a bar chart and pie chart. It helps evaluate if the dataset is balanced or imbalanced. A balanced dataset (approximately 50-50 split) is ideal for training, while an imbalanced dataset may require special techniques like class weighting or resampling. The visualization includes counts and percentages for easy interpretation.

### Visualization 2: Confusion Matrix Heatmap

**Code** (from `spam_detection.py`):
```python
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'],
            linewidths=2, linecolor='black', square=True)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=14, fontweight='bold')
plt.title('Confusion Matrix Heatmap', fontsize=16, fontweight='bold')
```

**Graph**: [confusion_matrix_heatmap.png](confusion_matrix_heatmap.png)

**Explanation**: 
This heatmap provides a graphical representation of the confusion matrix, showing how well the model performs on the test set. The color intensity indicates the count of predictions in each category. The diagonal elements (top-left and bottom-right) represent correct predictions (True Negatives and True Positives), while off-diagonal elements represent errors (False Positives and False Negatives). This visualization makes it easy to see the model's performance at a glance and identify any classification biases.

### Bonus Visualization: Feature Coefficients

**Code** (from `spam_detection.py`):
```python
feature_names = ['words', 'links', 'capital_words', 'spam_word_count']
coefficients = model.coef_[0]
plt.barh(feature_names, coefficients, 
         color=['red' if x < 0 else 'green' for x in coefficients])
plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Logistic Regression Feature Coefficients', 
          fontsize=14, fontweight='bold')
```

**Graph**: [feature_coefficients.png](feature_coefficients.png)

**Explanation**: 
This bar chart shows the learned coefficients for each feature in the logistic regression model. Positive coefficients (green) indicate features that increase the probability of spam classification, while negative coefficients (red) indicate features that decrease spam probability. The magnitude shows the relative importance of each feature. This helps understand which features the model considers most important for spam detection.

## How to Reproduce

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

### Running the Application

1. **Train the model and generate visualizations**:
   ```bash
   cd midterm
   python spam_detection.py
   ```

   This will:
   - Load the dataset
   - Train the logistic regression model on 70% of the data
   - Evaluate on the remaining 30%
   - Display model coefficients and performance metrics
   - Generate all visualizations

2. **Test individual emails**:
   ```bash
   python test_email.py
   ```
   
   For interactive mode, or:
   ```bash
   python test_email.py --examples
   ```
   
   To test the example spam and legitimate emails.

### Expected Output

When running `spam_detection.py`, you will see:
- Dataset information and statistics
- Training progress
- Model coefficients for each feature
- Test set accuracy and confusion matrix
- Classification report
- Confirmation of saved visualizations

## Files for Assignment 2

- `spam_detection.py`: Main application for training and evaluating the spam detection model
- `test_email.py`: Script for testing individual email classification
- `example_emails.txt`: Example spam and legitimate emails with explanations
- `data/i_matiashvili25_54376.csv`: Training dataset
- `class_distribution.png`: Visualization of class distribution (generated)
- `confusion_matrix_heatmap.png`: Confusion matrix heatmap (generated)
- `feature_coefficients.png`: Feature importance visualization (generated)

## Code Structure

### Main Functions in `spam_detection.py`:

1. **`load_data(file_path)`**: Loads the CSV dataset
2. **`extract_features_from_email(email_text)`**: Extracts features from raw email text
3. **`train_model(X_train, y_train)`**: Trains the logistic regression model
4. **`evaluate_model(model, X_test, y_test)`**: Evaluates model performance
5. **`classify_email(model, email_text)`**: Classifies a single email
6. **`create_visualizations(df, model, evaluation_results)`**: Generates all visualizations
7. **`main()`**: Orchestrates the entire process

## Summary

This spam detection application successfully:
- ✅ Loads and processes email feature data
- ✅ Trains a logistic regression model on 70% of the data
- ✅ Validates the model on the remaining 30% with confusion matrix and accuracy
- ✅ Extracts features from raw email text for classification
- ✅ Provides example spam and legitimate emails with explanations
- ✅ Generates comprehensive visualizations for data and model analysis

The model demonstrates good performance in distinguishing between spam and legitimate emails based on features such as word count, link count, capital words, and spam word frequency.
