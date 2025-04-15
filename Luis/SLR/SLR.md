# Simple Linear Regression

## Definition
Simple linear regression is a statistical method that aims to find a linear relationship between an independent variable (predictor) and a dependent variable (response). This technique establishes a straight line that best represents the relationship between these two variables, allowing us to make predictions about the dependent variable based on values of the independent variable.

## Mathematical Formulation

The equation of a simple linear regression model is:

$$\hat{y} = a + bx$$

Where:
- $\hat{y}$ is the predicted value of the dependent variable
- $x$ is the independent variable
- $a$ is the y-intercept (the value of $\hat{y}$ when $x = 0$)
- $b$ is the slope coefficient (the change in $\hat{y}$ for a unit change in $x$)

## Least Squares Regression Line (LSRL)

The most common method to determine the parameters $a$ and $b$ is the method of least squares, which minimizes the sum of squared differences between the observed values and the predicted values.

The formulas for calculating these parameters are:

$$b = \frac{S_{xy}}{S_{xx}} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{\sum(xy) - \frac{\sum{x}\sum{y}}{n}}{\sum(x^2) - \frac{(\sum{x})^2}{n}}$$

$$a = \bar{y} - b\bar{x}$$

Where:
- $\bar{x}$ is the mean of the independent variable ($\bar{x} = \frac{\sum{x}}{n}$)
- $\bar{y}$ is the mean of the dependent variable ($\bar{y} = \frac{\sum{y}}{n}$)
- $n$ is the number of data points

## Interpreting the Regression Line

The parameters of the regression line can be interpreted as follows:

- $a$ (y-intercept): The predicted value of the dependent variable when the independent variable equals zero.
- $b$ (slope): The change in the dependent variable for every one-unit increase in the independent variable.

## Applications

Simple linear regression can be used for:

1. **Prediction**: Estimating unknown values of the dependent variable based on known values of the independent variable.
2. **Understanding relationships**: Quantifying how changes in the independent variable relate to changes in the dependent variable.
3. **Trend analysis**: Identifying patterns and trends over time.

## Assumptions

For simple linear regression to be valid, several assumptions should be met:

1. **Linearity**: The relationship between variables is linear.
2. **Independence**: Observations are independent of each other.
3. **Homoscedasticity**: The variance of residuals is constant across all levels of the independent variable.
4. **Normality**: The residuals follow a normal distribution.

## Coefficient of Determination (R²)

R² measures the proportion of variance in the dependent variable that can be explained by the independent variable. It ranges from 0 to 1, where:
- R² = 0: The model explains none of the variability of the response data
- R² = 1: The model explains all the variability of the response data

## Sources

1. Newcastle University. "Simple Linear Regression." Retrieved from https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/simple-linear-regression.html

2. DATAtab. "Linear Regression." Retrieved from https://datatab.net/tutorial/linear-regression
