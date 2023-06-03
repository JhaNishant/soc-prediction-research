import argparse

import eli5
import joblib
import pandas as pd
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Perform permutation importance analysis."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file."
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the trained model file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file for the permutation importance scores.",
    )

    args = parser.parse_args()

    input_file = args.input_file
    model_file = args.model_file
    output_file = args.output_file

    # Load the dataset
    data = pd.read_csv(input_file)

    target_column = 'soc_percent'

    # Separate features from the target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Load the trained model
    rf = joblib.load(model_file)

    # Define and fit the Permutation Importance object
    perm = PermutationImportance(rf, random_state=42).fit(X_test, y_test)

    # Save the permutation importance scores to a DataFrame
    feature_importances = eli5.explain_weights_df(
        perm, feature_names=X_test.columns.tolist())

    # Save the permutation importance scores to a CSV file
    feature_importances.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
