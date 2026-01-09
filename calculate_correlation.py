"""
Pearson's Correlation Coefficient Calculator
This script calculates Pearson's correlation coefficient for the given data points
and creates a visualization of the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Blue dots (main data points) - coordinates from left to right
blue_dots = [
    (-9, 8),
    (-7, 6.68),
    (-5, 4.72),
    (-3, 5.03),
    (-1, 0.87),
    (1, -0.53),
    (3, -2.9),
    (5, -4.73),
    (7, -6.5),
    (9.1, -8.33)
]

# Green dots (X-axis markers)
green_dots = [
    (-9, 0), (-7, 0), (-5, 0), (-3, 0), (-1, 0),
    (1, 0), (3, 0), (5, 0), (7, 0), (9.1, 0)
]

# Red dots (Y-axis markers) - coordinates from top to bottom
red_dots = [
    (0, 7.87), (0, 6.71), (0, 5.1), (0, 4.8), (0, 0.87),
    (0, -0.5), (0, -2.85), (0, -4.67), (0, -6.5), (0, -8.3)
]


def calculate_pearson_correlation(x, y):
    """
    Calculate Pearson's correlation coefficient using the formula:
    r = Σ(xi - x̄)(yi - ȳ) / √[Σ(xi - x̄)² Σ(yi - ȳ)²]
    
    Parameters:
    x: array-like, x coordinates
    y: array-like, y coordinates
    
    Returns:
    r: float, Pearson's correlation coefficient
    """
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate numerator: Σ(xi - x̄)(yi - ȳ)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    
    # Calculate denominator: √[Σ(xi - x̄)² Σ(yi - ȳ)²]
    sum_sq_x = np.sum((x - x_mean) ** 2)
    sum_sq_y = np.sum((y - y_mean) ** 2)
    denominator = np.sqrt(sum_sq_x * sum_sq_y)
    
    # Calculate correlation coefficient
    if denominator == 0:
        return 0  # Avoid division by zero
    r = numerator / denominator
    
    return r


def create_visualization(blue_data, green_data, red_data, correlation_coef):
    """
    Create a visualization of the data points with correlation information.
    
    Parameters:
    blue_data: list of tuples, main data points
    green_data: list of tuples, X-axis markers
    red_data: list of tuples, Y-axis markers
    correlation_coef: float, Pearson's correlation coefficient
    """
    # Extract coordinates
    blue_x, blue_y = zip(*blue_data)
    green_x, green_y = zip(*green_data)
    red_x, red_y = zip(*red_data)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot green dots (X-axis markers)
    ax.scatter(green_x, green_y, c='green', s=100, marker='o', 
               label='X-axis markers', alpha=0.7, edgecolors='darkgreen', linewidths=1.5)
    
    # Plot red dots (Y-axis markers)
    ax.scatter(red_x, red_y, c='red', s=100, marker='o', 
               label='Y-axis markers', alpha=0.7, edgecolors='darkred', linewidths=1.5)
    
    # Plot blue dots (main data points)
    ax.scatter(blue_x, blue_y, c='blue', s=150, marker='o', 
               label='Data points', alpha=0.8, edgecolors='darkblue', linewidths=2)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis labels
    ax.set_xlabel('X ღერძი (X axis)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y ღერძი (Y axis)', fontsize=12, fontweight='bold')
    ax.set_title('Pearson\'s Correlation Coefficient Visualization', fontsize=14, fontweight='bold')
    
    # Add correlation coefficient as text
    correlation_text = f"Pearson's Correlation Coefficient: r = {correlation_coef:.4f}"
    ax.text(0.02, 0.98, correlation_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Set axis limits with some padding
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # Add formula text
    formula_text = "Formula: r = Σ(xi-x̄)(yi-ȳ) / √[Σ(xi-x̄)² Σ(yi-ȳ)²]"
    ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'correlation_plot.png'")
    
    # Show the plot
    plt.show()


def main():
    """
    Main function to calculate correlation and create visualization.
    """
    # Extract x and y coordinates from blue dots
    x_coords = [point[0] for point in blue_dots]
    y_coords = [point[1] for point in blue_dots]
    
    # Create a DataFrame for better data handling
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords
    })
    
    print("Data Points (Blue Dots):")
    print(df.to_string(index=False))
    print()
    
    # Calculate Pearson's correlation coefficient
    correlation = calculate_pearson_correlation(x_coords, y_coords)
    
    # Also verify using numpy's built-in function
    numpy_correlation = np.corrcoef(x_coords, y_coords)[0, 1]
    
    print("=" * 60)
    print("Pearson's Correlation Coefficient Calculation")
    print("=" * 60)
    print(f"Manual calculation: r = {correlation:.6f}")
    print(f"NumPy verification: r = {numpy_correlation:.6f}")
    print()
    
    # Calculate intermediate values for detailed explanation
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    
    print("Intermediate Calculations:")
    print(f"Mean of X (x̄): {x_mean:.4f}")
    print(f"Mean of Y (ȳ): {y_mean:.4f}")
    print()
    
    # Calculate components
    deviations_x = [x - x_mean for x in x_coords]
    deviations_y = [y - y_mean for y in y_coords]
    products = [dx * dy for dx, dy in zip(deviations_x, deviations_y)]
    sq_deviations_x = [dx ** 2 for dx in deviations_x]
    sq_deviations_y = [dy ** 2 for dy in deviations_y]
    
    numerator = sum(products)
    sum_sq_x = sum(sq_deviations_x)
    sum_sq_y = sum(sq_deviations_y)
    denominator = np.sqrt(sum_sq_x * sum_sq_y)
    
    print(f"Numerator: Σ(xi - x̄)(yi - ȳ) = {numerator:.4f}")
    print(f"Σ(xi - x̄)² = {sum_sq_x:.4f}")
    print(f"Σ(yi - ȳ)² = {sum_sq_y:.4f}")
    print(f"Denominator: √[Σ(xi - x̄)² × Σ(yi - ȳ)²] = {denominator:.4f}")
    print(f"Correlation: r = {numerator:.4f} / {denominator:.4f} = {correlation:.6f}")
    print("=" * 60)
    print()
    
    # Create visualization
    create_visualization(blue_dots, green_dots, red_dots, correlation)
    
    return correlation


if __name__ == "__main__":
    correlation_result = main()
