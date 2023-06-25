import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabtransformertf.models.tabtransformer import TabTransformer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting script...')

    parser = argparse.ArgumentParser(
        description="Perform regression analysis using TabTransformer model."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file.",
    )
    parser.add_argument(
        "plot_file",
        type=str,
        help="Path to the output file for the predicted vs actual plot.",
    )
    parser.add_argument(
        "tabtransformer_model",
        type=str,
        help="Path to the output file for the trained TabTransformer model.",
    )

    args = parser.parse_args()

    input_file = args.input_file
    plot_file = args.plot_file
    tabtransformer_model = args.tabtransformer_model

    # Load the dataset
    data = pd.read_csv(input_file)

    target_column = 'soc_percent'

    # Separate features from the target
    X = data.drop(target_column, axis=1).values
    y = data[target_column].values

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the TabTransformer model
    model = TabTransformer(out_dim=1, out_activation='linear')

    # Compile the TabTransformer model
    model.compile(loss='mean_squared_error', optimizer=Adam())

    # Print the model summary
    model.summary()

    # Define early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True)

    # Train the model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        callbacks=[early_stop])

    # Save the trained model to a file
    model.save(tabtransformer_model)
    logging.info(f'TabTransformer model saved to {tabtransformer_model}.')

    # Use the model to make predictions on the test data
    y_pred = model.predict(X_test)

    # Actual vs Predicted plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(
        x=y_test,
        y=y_pred.flatten(),
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
    residuals = y_test - y_pred.flatten()
    sns.scatterplot(
        x=y_pred.flatten(),
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
    mae = mean_absolute_error(y_test, y_pred.flatten())
    mse = mean_squared_error(y_test, y_pred.flatten())
    r2 = r2_score(y_test, y_pred.flatten())

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

    logging.info('Script finished.')


if __name__ == "__main__":
    main()
