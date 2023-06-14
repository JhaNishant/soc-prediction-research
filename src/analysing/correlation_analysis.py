import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting script...')

    parser = argparse.ArgumentParser(
        description="Calculate the correlation coefficients between "
                    "'soc_percent' and other features.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input dataset file.")
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file for the correlation coefficients.",
    )
    parser.add_argument(
        "correlation_bar_graph",
        type=str,
        help="Path to the output file for the correlation bar graph.",
    )
    parser.add_argument(
        "scatterplot_matrix",
        type=str,
        help="Path to the output file for the scatter plot matrix.",
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    correlation_bar_graph = args.correlation_bar_graph
    scatterplot_matrix = args.scatterplot_matrix

    # Load the dataset
    df = pd.read_csv(input_file)

    # Calculate the correlation matrix
    corr_matrix = df.corr(method="pearson")

    # Get the correlation coefficients between 'soc_percent' and other features
    soc_percent_corr = corr_matrix["soc_percent"]

    # Sort correlation coefficients in descending order
    soc_percent_corr_sorted = soc_percent_corr.sort_values(ascending=False)

    # Create a new DataFrame with two columns: predictor_names and
    # correlation_values
    output_df = pd.DataFrame({
        'predictor_names': soc_percent_corr_sorted.index,
        'correlation_values': soc_percent_corr_sorted.values
    })

    # Save the sorted result to a file with a proper header
    output_df.to_csv(output_file, index=False)
    logging.info(f"Saved correlation coefficients to {output_file}")

    # Threshold for filtering correlation coefficients
    threshold = 0.4

    # Filter out features with a low correlation
    significant_corr = soc_percent_corr[abs(soc_percent_corr) >= threshold]

    # Create a bar graph for features with a correlation coefficient above the
    # threshold
    plt.figure(figsize=(10, 6))
    bar_colors = ['g' if corr > 0 else 'r' for corr in significant_corr]
    bars = significant_corr.plot(kind='bar', color=bar_colors)
    plt.title("Correlation Coefficients (|r| >= {})".format(threshold))
    plt.xlabel("Predictors")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(rotation=45, ha='right')

    # Add a horizontal black line at y=0
    plt.axhline(0, color='black', linewidth=1)

    # Add correlation values above bars
    for i, (index, value) in enumerate(significant_corr.items()):
        bars.text(
            i,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value > 0 else "top",
            fontsize=8)

    # Add legend for positive and negative correlations
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='r', lw=4)]
    plt.legend(custom_lines, ['Positive Correlation', 'Negative Correlation'])

    plt.tight_layout()
    plt.savefig(correlation_bar_graph)
    logging.info(f"Saved correlation bar graph to {correlation_bar_graph}")

    # Create a scatter plot matrix for the significant_corr columns
    significant_columns = list(significant_corr.index)
    scatter_df = df[significant_columns]

    sns.pairplot(scatter_df)
    plt.savefig(scatterplot_matrix)
    logging.info(f"Saved scatter plot matrix to {scatterplot_matrix}")

    logging.info('Finished script.')


if __name__ == "__main__":
    main()
