import argparse
import logging

import cupy as cp
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from cuml.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting script...')

    # Argument parser for command line interaction
    parser = argparse.ArgumentParser(
        description="Train a Lasso regression model."
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
    # Output coefficients file path
    parser.add_argument(
        "coeff_file",
        type=str,
        help="Path to the output file for the model's coefficients."
    )
    # Output plot file path
    parser.add_argument(
        "plot_file",
        type=str,
        help="Path to the output file for the prediction plot."
    )

    args = parser.parse_args()

    input_file = args.input_file
    model_file = args.model_file
    coeff_file = args.coeff_file
    plot_file = args.plot_file

    # Load dataset
    data = pd.read_csv(input_file)

    target_column = 'soc_percent'

    # Split data into features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Convert dataframes to CuPy arrays for GPU computation
    X = cp.asarray(X)
    y = cp.asarray(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize Lasso regression model
    lasso = Lasso(alpha=0.1)

    # Train model
    lasso.fit(X_train, y_train)
    logging.info('Model trained.')

    # Predict on test data
    y_pred = lasso.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Mean Squared Error: {mse}')

    # Save trained model
    joblib.dump(lasso, model_file)
    logging.info(f'Model saved to {model_file}.')

    # Save coefficients
    coeff_df = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])
    coeff_df.to_csv(coeff_file)
    logging.info(f'Coefficients saved to {coeff_file}.')

    # Plot actual vs predicted values
    plt.scatter(y_test.get(), y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values for Lasso Regression')
    plt.savefig(plot_file)
    logging.info(f'Plot saved as {plot_file}.')

    logging.info('Finished script.')


if __name__ == '__main__':
    main()
