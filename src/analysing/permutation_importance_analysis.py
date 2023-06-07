import argparse
import logging

import eli5
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting script...')

    parser = argparse.ArgumentParser(
        description="Perform permutation importance analysis."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file."
    )
    parser.add_argument(
        "output_file_csv",
        type=str,
        help="Path to the output file for the "
             "permutation importance scores in CSV format.",
    )
    parser.add_argument(
        "output_file_png",
        type=str,
        help="Path to the output file for the "
             "permutation importance scores in PNG format.",
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the trained model file."
    )

    args = parser.parse_args()

    input_file = args.input_file
    model_file = args.model_file
    output_file_csv = args.output_file_csv
    output_file_png = args.output_file_png

    data = pd.read_csv(input_file)

    target_column = 'soc_percent'

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf = joblib.load(model_file)
    logging.info(f'Trained model loaded from {model_file}.')

    perm = PermutationImportance(rf, random_state=42).fit(X_test, y_test)

    feature_importances = eli5.explain_weights_df(
        perm, feature_names=X_test.columns.tolist())

    feature_importances = feature_importances.sort_values(
        by='weight', ascending=False)

    feature_importances.to_csv(output_file_csv, index=False)
    logging.info(f'Permutation importance scores saved to {output_file_csv}.')

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='weight',
        y='feature',
        data=feature_importances.head(10))
    plt.title('Top 10 Permutation Importance Scores')

    # Get the maximum 'weight' value to set the xlim of the plot
    max_weight = feature_importances['weight'].head(10).max()

    # Add the weight values in front of the bars
    for i, v in enumerate(feature_importances['weight'].head(10)):
        plt.text(v + 0.01, i, str(round(v, 2)),
                 color='black', fontweight='bold')

    # Set the xlim of the plot to be slightly larger than the maximum 'weight'
    # value
    plt.xlim(0, max_weight + 0.1)

    plt.savefig(output_file_png)
    logging.info(
        f'Top 10 Permutation Importance Scores plot saved to '
        f'{output_file_png}.')

    logging.info('Finished script.')


if __name__ == '__main__':
    main()
