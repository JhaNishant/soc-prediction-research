import argparse
import logging

import cudf
import cupy as cp
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cuml import LinearRegression as cuLinearRegression
from cuml.metrics.regression import (mean_absolute_error, mean_squared_error,
                                     r2_score)
from cuml.model_selection import train_test_split as cuml_train_test_split
from cuml.preprocessing import PolynomialFeatures


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting script...')

    # Argument parser for command line interaction
    parser = argparse.ArgumentParser(
        description="Train a Polynomial Regression model."
    )
    # Input dataset file path
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file."
    )
    # Output model file path
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the output file for the trained model."
    )
    # Output plot file path
    parser.add_argument(
        "plot_file",
        type=str,
        help="Path to the output file for the prediction plot."
    )
    # Output coefficients file path
    parser.add_argument(
        "coefficients_file",
        type=str,
        help="Path to the output file for the coefficients."
    )

    args = parser.parse_args()

    input_file = args.input_file
    model_file = args.model_file
    plot_file = args.plot_file
    coefficients_file = args.coefficients_file

    # Load dataset
    data = pd.read_csv(input_file)

    target_column = 'soc_percent'

    # Split data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Convert pandas dataframes to cuDF dataframes for GPU computation
    X = cudf.DataFrame.from_pandas(X)
    y = cudf.Series(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = cuml_train_test_split(
        X, y, test_size=0.2, random_state=42)

    best_degree = None
    best_mse = np.inf

    # Set the maximum degree of the polynomial features
    max_degree = 10

    for degree in range(2, max_degree + 1):
        # Create polynomial features
        polynomial_features = PolynomialFeatures(degree=degree)
        X_train_poly = polynomial_features.fit_transform(X_train)
        X_test_poly = polynomial_features.transform(X_test)

        # Train a linear regression model
        model = cuLinearRegression()
        model.fit(X_train_poly, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test_poly)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Degree {degree}: Mean Squared Error: {mse}")

        # Check if the current model's MSE is the best so far
        if mse < best_mse:
            best_mse = mse
            best_degree = degree
        else:
            # If the MSE starts increasing, stop training further models
            break

    logging.info(f"Best degree: {best_degree}")
    logging.info(f"Best MSE: {best_mse}")

    # Create polynomial features with the best degree
    polynomial_features = PolynomialFeatures(degree=best_degree)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_test_poly = polynomial_features.transform(X_test)

    # Initialize Polynomial Regression model
    poly_reg = cuLinearRegression()

    # Train model
    poly_reg.fit(X_train_poly, y_train)
    logging.info('Model trained.')

    # Predict on test data
    y_pred = poly_reg.predict(X_test_poly)

    # Convert cuDF series to cupy array for further computation
    y_pred = cp.asarray(y_pred)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Mean Squared Error: {mse}')

    # Save trained model
    joblib.dump(poly_reg, model_file)
    logging.info(f'Model saved to {model_file}.')

    # Convert cuDF series to cupy array for plotting
    y_test = cp.asarray(y_test)

    # Compute residuals
    residuals = y_test.get() - y_pred.get()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Actual vs Predicted plot
    sns.scatterplot(
        x=y_test.get(),
        y=y_pred.get(),
        ax=ax[0],
        edgecolor=None,
        alpha=0.6,
        s=100)
    ax[0].set_xlabel('Actual Values', fontsize=12)
    ax[0].set_ylabel('Predicted Values', fontsize=12)
    ax[0].set_title('Actual vs Predicted Values', fontsize=14)
    # Add a line for perfect predictions
    lims = [
        np.min([ax[0].get_xlim(), ax[0].get_ylim()]),  # min of both axes
        np.max([ax[0].get_xlim(), ax[0].get_ylim()]),  # max of both axes
    ]
    ax[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    # Residual plot
    sns.scatterplot(
        x=y_pred.get(),
        y=residuals,
        ax=ax[1],
        edgecolor=None,
        alpha=0.6,
        s=100)
    ax[1].axhline(y=0, color='black', linestyle='--')
    ax[1].set_xlabel('Predicted Values', fontsize=12)
    ax[1].set_ylabel('Residuals', fontsize=12)
    ax[1].set_title('Residual Plot', fontsize=14)

    # Calculate and display regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_text = f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nR2: {r2:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # Place a text box with metrics in each subplot
    ax[0].text(
        0.05,
        0.95,
        metrics_text,
        transform=ax[0].transAxes,
        verticalalignment='top',
        bbox=props)
    ax[1].text(
        0.05,
        0.95,
        metrics_text,
        transform=ax[1].transAxes,
        verticalalignment='top',
        bbox=props)

    plt.tight_layout()
    plt.savefig(plot_file)
    logging.info(f'Plot saved as {plot_file}.')

    # Get the coefficients for each term in the polynomial regression equation
    coefficients = poly_reg.coef_
    equation = generate_equation(polynomial_features, coefficients)

    # Save coefficients and equation to a CSV file
    coefficients_df = pd.DataFrame(
        {'Term': equation.keys(), 'Coefficient': equation.values()})
    coefficients_df.to_csv(coefficients_file, index=False)
    logging.info(f'Coefficients saved to {coefficients_file}.')

    logging.info('Finished script.')


def generate_equation(polynomial_features, coefficients):
    # Get the feature names from the polynomial features
    feature_names = polynomial_features.get_feature_names_out()

    # Remove the constant term (intercept) from the feature names
    feature_names = feature_names[1:]

    # Create a dictionary to store the equation
    equation = {}

    # Generate the equation by combining the feature names and coefficients
    for feature, coefficient in zip(feature_names, coefficients):
        equation[feature] = coefficient

    return equation


if __name__ == '__main__':
    main()
