# Example usage for correlation matrix here we have taken clinical correlation matrix csv file
corr_matrix_path = "clinical_corr_2.csv"
k = 0.04  # Example value for k
cutoff_betti = 0.5  # Example cutoff Betti value
results_df_clinical = process_correlation_matrix(corr_matrix_path, k, cutoff_betti)
