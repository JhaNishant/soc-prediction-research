import argparse
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting script...')

    parser = argparse.ArgumentParser(
        description="Perform regression analysis and generate relevant plots."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file.",
    )
    parser.add_argument(
        "output_file_predictions",
        type=str,
        help="Path to the output file for the predictions.",
    )
    parser.add_argument(
        "output_file_coefficients",
        type=str,
        help="Path to the output file for the feature importance.",
    )
    parser.add_argument(
        "predicted_vs_actual_plot",
        type=str,
        help="Path to the output file for the predicted vs actual plot.",
    )
    parser.add_argument(
        "regression_plot",
        type=str,
        help="Path to the output file for the regression plot.",
    )
    parser.add_argument(
        "rf_model",
        type=str,
        help="Path to the output file for the trained random forest model.",
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file_predictions = args.output_file_predictions
    output_file_coefficients = args.output_file_coefficients
    predicted_vs_actual_plot = args.predicted_vs_actual_plot
    regression_plot = args.regression_plot
    rf_model = args.rf_model

    # Load the dataset
    data = pd.read_csv(input_file)

    target_column = 'soc_percent'

    # Separate features from the target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create a random forest regressor object
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, verbose=2)

    # Train the model using the training sets
    rf.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(rf, rf_model)
    logging.info(f'rf_model saved to {rf_model}.')

    # Use the model to make predictions on the test data
    y_pred = rf.predict(X_test)

    # Save predictions to a CSV file
    predictions = pd.DataFrame(data={'Actual': y_test, 'Predicted': y_pred})
    predictions.to_csv(output_file_predictions, index=False)
    logging.info(f'Predictions saved to {output_file_predictions}.')

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', data=predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.plot([predictions['Actual'].min(), predictions['Actual'].max()],
             [predictions['Actual'].min(), predictions['Actual'].max()],
             color='red', lw=2)

    # Calculate and display regression metrics
    mae = mean_absolute_error(predictions['Actual'], predictions['Predicted'])
    mse = mean_squared_error(predictions['Actual'], predictions['Predicted'])
    r2 = r2_score(predictions['Actual'], predictions['Predicted'])
    plt.text(
        0.05,
        0.85,
        f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nR2: {r2:.2f}',
        transform=plt.gca().transAxes,
        bbox=dict(
            facecolor='white',
            alpha=0.5))

    plt.savefig(predicted_vs_actual_plot)
    logging.info(
        f'Predicted vs Actual plot saved to {predicted_vs_actual_plot}.')

    # Get the feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Save the feature importances to a CSV file
    feature_importances = pd.DataFrame(
        data={
            'Feature': X.columns[indices],
            'Importance': importances[indices]})
    feature_importances.to_csv(output_file_coefficients, index=False)
    logging.info(f'Feature importances saved to {output_file_coefficients}.')

    # Plot the feature importances (only the top 10)
    plt.figure(figsize=(12, 12))  # Increased the size of the image
    plt.title("Feature importances")
    plt.barh(range(10), importances[indices][:10],
             color="r", align="center")
    plt.yticks(range(10), X.columns[indices][:10], rotation=45)
    # Adjusting the left margin to make more space for feature names
    plt.subplots_adjust(left=0.30)
    plt.ylim([-1, 10])
    plt.gca().invert_yaxis()  # To display the highest importance on top
    plt.savefig(regression_plot)
    logging.info(f'Regression plot saved to {regression_plot}.')

    logging.info('Finished script.')


if __name__ == '__main__':
    main()
