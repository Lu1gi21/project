import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns

# Add project directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models and data utilities
from Luis.SLR.slr import SimpleLinearRegression
from Luis.Gradient_Descent.gradientDescent import GradientDescentRegression
from Data.data import load_clean_vgsales, get_numeric_vgsales_columns

# Set a more appealing style for plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def experiment_predict_global_sales():
    """
    Experiment to predict Global_Sales based on NA_Sales using both regression models.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)
    
    # Prepare features and target
    X = df_numeric['NA_Sales'].values.reshape(-1, 1)
    y = df_numeric['Global_Sales'].values
    
    # Train Simple Linear Regression model
    slr_model = SimpleLinearRegression()
    slr_model.fit(X, y)
    slr_score = slr_model.score(X, y)
    slr_predictions = slr_model.predict(X)
    
    # Train Gradient Descent model
    gd_model = GradientDescentRegression(learning_rate=0.01, n_iterations=1000)
    gd_model.fit(X, y)
    gd_score = gd_model.score(X, y)
    gd_predictions = gd_model.predict(X)
    
    # Plot results
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Video Game Sales Prediction Analysis: North America vs Global", fontsize=16, fontweight='bold')
    
    # SLR Plot
    ax1 = plt.subplot(2, 2, 1)
    scatter = ax1.scatter(X, y, alpha=0.6, color='#3498db', label='Actual data points')
    line = ax1.plot(X, slr_predictions, color='#e74c3c', linewidth=2, label='Regression line')
    ax1.set_xlabel('North America Sales (millions)')
    ax1.set_ylabel('Global Sales (millions)')
    ax1.set_title(f'Simple Linear Regression\nR² = {slr_score:.4f}')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # Add formula text
    intercept, slope = slr_model.weights
    formula = f"Global Sales = {slope:.4f} × NA Sales + {intercept:.4f}"
    ax1.text(0.05, 0.95, formula, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # GD Plot
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(X, y, alpha=0.6, color='#3498db', label='Actual data points')
    ax2.plot(X, gd_predictions, color='#9b59b6', linewidth=2, label='Regression line')
    ax2.set_xlabel('North America Sales (millions)')
    ax2.set_ylabel('Global Sales (millions)')
    ax2.set_title(f'Gradient Descent Regression\nR² = {gd_score:.4f}')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')
    
    # Add formula text
    gd_intercept, gd_slope = gd_model.weights
    gd_formula = f"Global Sales = {gd_slope:.4f} × NA Sales + {gd_intercept:.4f}"
    ax2.text(0.05, 0.95, gd_formula, transform=ax2.transAxes, fontsize=11,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Residuals Plot
    ax3 = plt.subplot(2, 2, 3)
    residuals = y - slr_predictions
    ax3.scatter(X, residuals, alpha=0.6, color='#2ecc71')
    ax3.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
    ax3.set_xlabel('North America Sales (millions)')
    ax3.set_ylabel('Residuals (Actual - Predicted)')
    ax3.set_title('Simple Linear Regression Residuals\nOverpredictions above line, Underpredictions below')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanatory text
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    residual_text = f"Mean residual: {mean_residual:.4f}\nStd deviation: {std_residual:.4f}"
    ax3.text(0.05, 0.05, residual_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Cost History Plot
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(range(1, len(gd_model.cost_history) + 1), gd_model.cost_history, color='#e67e22', linewidth=2)
    ax4.set_xlabel('Iteration Number')
    ax4.set_ylabel('Mean Squared Error Cost')
    ax4.set_title('Gradient Descent Optimization\nCost Reduction During Training')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Mark final cost value
    final_cost = gd_model.cost_history[-1]
    ax4.annotate(f'Final cost: {final_cost:.4f}', 
                xy=(len(gd_model.cost_history), final_cost),
                xytext=(len(gd_model.cost_history)*0.8, final_cost*1.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the suptitle
    plt.show()
    
    print(f"SLR R² score: {slr_score:.4f}")
    print(f"GD R² score: {gd_score:.4f}")
    print(f"SLR Coefficients: {slr_model.weights}")
    print(f"GD Coefficients: {gd_model.weights}")

def experiment_predict_regional_sales():
    """
    Experiment to predict JP_Sales based on EU_Sales.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)
    
    # Prepare features and target
    X = df_numeric['EU_Sales'].values.reshape(-1, 1)
    y = df_numeric['JP_Sales'].values
    
    # Train model
    model = SimpleLinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    predictions = model.predict(X)
    
    # Create enhanced plot
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, alpha=0.5, color='#2c3e50', label='Game titles')
    plt.plot(X, predictions, color='#e74c3c', linewidth=3, label='Regression line')
    
    # Add titles and labels
    plt.title('Relationship Between European and Japanese Video Game Sales', fontsize=16, fontweight='bold')
    plt.xlabel('European Sales (millions)', fontsize=14)
    plt.ylabel('Japanese Sales (millions)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation with model information
    intercept, slope = model.weights
    equation = f"JP Sales = {slope:.4f} × EU Sales + {intercept:.4f}"
    model_info = f"R² = {score:.4f}\n{equation}"
    plt.annotate(model_info, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
                 fontsize=12, verticalalignment='top')
    
    # Add correlation information
    correlation = np.corrcoef(X.flatten(), y)[0, 1]
    plt.annotate(f"Correlation: {correlation:.4f}", xy=(0.05, 0.80), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
                 fontsize=12, verticalalignment='top')
    
    # Mark the maximum values
    max_eu_index = np.argmax(X)
    max_jp_index = np.argmax(y)
    
    plt.scatter(X[max_eu_index], y[max_eu_index], color='green', s=100, edgecolor='black', zorder=5)
    plt.annotate('Highest EU Sales', xy=(X[max_eu_index], y[max_eu_index]),
                 xytext=(X[max_eu_index], y[max_eu_index]*1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, ha='center')
    
    plt.scatter(X[max_jp_index], y[max_jp_index], color='orange', s=100, edgecolor='black', zorder=5)
    plt.annotate('Highest JP Sales', xy=(X[max_jp_index], y[max_jp_index]),
                 xytext=(X[max_jp_index]*1.2, y[max_jp_index]),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, ha='center')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print(f"R² score for predicting JP_Sales from EU_Sales: {score:.4f}")
    print(f"Coefficients: {model.weights}")
    print(f"Correlation between JP_Sales and EU_Sales: {correlation:.4f}")

def experiment_sales_by_year():
    """
    Experiment to analyze the trend of Global_Sales over Year.
    """
    # Load and prepare data
    df = load_clean_vgsales()
    df_numeric = get_numeric_vgsales_columns(df)
    
    # Group by year and get average global sales
    yearly_sales = df_numeric.groupby('Year')['Global_Sales'].mean().reset_index()
    
    # Also get count of games per year for bubble size
    yearly_count = df_numeric.groupby('Year').size().reset_index(name='Count')
    yearly_data = yearly_sales.merge(yearly_count, on='Year')
    
    # Prepare features and target
    X = yearly_data['Year'].values.reshape(-1, 1)
    y = yearly_data['Global_Sales'].values
    
    # Train model
    model = SimpleLinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    predictions = model.predict(X)
    
    # Create enhanced plot
    plt.figure(figsize=(14, 10))
    
    # Scatter plot with bubble size representing number of games
    scatter = plt.scatter(yearly_data['Year'], yearly_data['Global_Sales'], 
               s=yearly_data['Count']*2, alpha=0.6, 
               c=yearly_data['Year'], cmap='viridis', 
               edgecolor='black', linewidth=1)
    
    # Add regression line
    plt.plot(X, predictions, color='red', linewidth=3, 
             linestyle='--', label='Trend line')
    
    # Add title and labels
    plt.title('Average Global Video Game Sales Trend by Year (1980-2020)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Release Year', fontsize=14)
    plt.ylabel('Average Global Sales (millions)', fontsize=14)
    
    # Add colorbar to show the year gradient
    cbar = plt.colorbar(scatter)
    cbar.set_label('Release Year', fontsize=12)
    
    # Add annotations for significant years
    # Find peak sales year
    peak_idx = np.argmax(yearly_data['Global_Sales'])
    peak_year = yearly_data.iloc[peak_idx]['Year']
    peak_sales = yearly_data.iloc[peak_idx]['Global_Sales']
    
    plt.annotate(f'Peak: {peak_year:.0f}\n{peak_sales:.2f}M', 
                xy=(peak_year, peak_sales),
                xytext=(peak_year-5, peak_sales*1.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add trend information
    intercept, slope = model.weights
    trend_text = f"Trend Equation: Sales = {slope:.4f} × Year + {intercept:.4f}\nR² = {score:.4f}"
    
    if slope > 0:
        trend_direction = f"Sales increasing by {slope:.4f}M per year"
    else:
        trend_direction = f"Sales decreasing by {abs(slope):.4f}M per year"
        
    plt.figtext(0.15, 0.15, trend_text + '\n' + trend_direction, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), 
                fontsize=12)
    
    # Add legend to explain bubble size
    plt.plot([], [], 'o', ms=5, mec='black', mfc='none', alpha=0.6, label='Small: Few Games')
    plt.plot([], [], 'o', ms=10, mec='black', mfc='none', alpha=0.6, label='Medium: More Games')
    plt.plot([], [], 'o', ms=20, mec='black', mfc='none', alpha=0.6, label='Large: Many Games')
    
    plt.legend(title='Bubble Size = Number of Games', loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print(f"R² score for predicting Global_Sales trend by Year: {score:.4f}")
    print(f"Coefficients: {model.weights}")
    print(f"Peak average sales year: {peak_year:.0f}, with {peak_sales:.4f}M average sales")

if __name__ == "__main__":
    print("Experiment 1: Predicting Global Sales from NA Sales")
    experiment_predict_global_sales()
    
    print("\nExperiment 2: Predicting JP Sales from EU Sales")
    experiment_predict_regional_sales()
    
    print("\nExperiment 3: Analyzing Sales Trends by Year")
    experiment_sales_by_year()
