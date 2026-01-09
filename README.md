# AI Midterm Exam - Finding the Correlation

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

## Files in This Repository

- `calculate_correlation.py`: Python script that calculates Pearson's correlation coefficient and creates the visualization
- `requirements.txt`: Python package dependencies
- `correlation_plot.png`: Generated visualization graph (created when script is run)
- `README.md`: This report document

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
