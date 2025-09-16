import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

class ClusterAnalysisPlotter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.analysis_df = None

    def load_and_prepare_data(self):
        """Loads Excel file and prepares analysis DataFrame."""
        self.analysis_df = pd.read_excel(self.file_path, sheet_name='Analysis')
        self.analysis_df['Wavenumber 1'] = self.analysis_df['Wavenumber 1'].round()
        self.analysis_df['Wavenumber 2'] = self.analysis_df['Wavenumber 2'].round()
        self.analysis_df.sort_values(by=['Wavenumber 1', 'Wavenumber 2'], inplace=True)
        self.analysis_df.sort_values(by='Sum of Mean Heights', ascending=False, inplace=True)
        self.analysis_df['Priority'] = range(1, len(self.analysis_df) + 1)

    @staticmethod
    def find_best_r2(concentrations, mean_peak_heights):
        """Finds the subset of data with the highest R² for linear regression."""
        best_r2 = -np.inf
        best_slope = best_intercept = None
        best_subset = None

        for i in range(len(concentrations) - 2):
            sub_conc = concentrations[i:]
            sub_heights = mean_peak_heights[i:]
            slope, intercept, r_value, _, _ = linregress(sub_conc, sub_heights)
            print(f"Subset {i+1}: Concentrations {sub_conc}, Mean Peak Heights {sub_heights}, R²={r_value**2:.6f}")
            if r_value**2 > best_r2:
                best_r2 = r_value**2
                best_slope = slope
                best_intercept = intercept
                best_subset = (sub_conc, sub_heights)

        return best_r2, best_slope, best_intercept, best_subset

    def plot_best_fit_ratio_transformed(self, cluster1, cluster2, priority, label):
        """Returns a plot, regression stats, and subset info for a cluster ratio."""
        ratio_df = self.analysis_df[
            (self.analysis_df['Cluster 1'] == cluster1) & 
            (self.analysis_df['Cluster 2'] == cluster2)
        ]
        if ratio_df.empty:
            raise ValueError(f"No data for Cluster {cluster1}/{cluster2}")

        valid_ratio_df = ratio_df.dropna(subset=['Mean Peak Height Ratio', 'Concentration'])
        valid_ratio_df = valid_ratio_df[valid_ratio_df['Mean Peak Height Ratio'] > 0]
        if valid_ratio_df.empty:
            raise ValueError(f"No valid data for linear regression in Cluster {cluster1}/{cluster2}")

        # Convert concentrations if they are in string format like '10^-1'
        if valid_ratio_df['Concentration'].dtype == object:
            try:
                valid_ratio_df.loc[:, 'Converted Concentration'] = valid_ratio_df['Concentration'].apply(
                    lambda x: float(str(x).replace("10^", "1E"))
                )
            except ValueError as e:
                raise ValueError(f"Error converting 'Concentration' values: {e}")
        else:
            valid_ratio_df.loc[:, 'Converted Concentration'] = valid_ratio_df['Concentration']

        # Log transformation
        valid_ratio_df.loc[:, 'Log Concentration'] = np.log10(valid_ratio_df['Converted Concentration'])
        valid_ratio_df = valid_ratio_df.dropna(subset=['Log Concentration'])
        if valid_ratio_df.empty:
            raise ValueError(f"No valid log concentrations for Cluster {cluster1}/{cluster2}")

        log_concentrations = valid_ratio_df['Log Concentration'].values
        mean_peak_heights = valid_ratio_df['Mean Peak Height Ratio'].values
        std_peak_heights = valid_ratio_df['Std Peak Height Ratio'].values

        # Find best fit
        best_r2, best_slope, best_intercept, best_subset = self.find_best_r2(log_concentrations, mean_peak_heights)

        # Record subset R² details
        subset_details = []
        for i in range(len(log_concentrations) - 2):
            sub_conc = log_concentrations[i:]
            sub_heights = mean_peak_heights[i:]
            slope, intercept, r_value, _, _ = linregress(sub_conc, sub_heights)
            subset_details.append(
                f"Subset {i + 1}: Concentrations {sub_conc}, Mean Peak Heights {sub_heights}, R²={r_value**2:.6f}"
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(log_concentrations, mean_peak_heights, yerr=std_peak_heights, fmt='o', capsize=5, color='blue', label='Concentrations vs Peak_height')

        if best_subset is not None:
            best_log_concentrations, _ = best_subset
            regression_line = best_slope * best_log_concentrations + best_intercept
            ax.plot(best_log_concentrations, regression_line, 'r--',
                    label=f'Best Linear fit: y={best_slope:.6f}x + {best_intercept:.6f}, R²={best_r2:.6f}')

        ax.set_xlabel('Log(Concentration)')
        ax.set_ylabel('Mean Peak Height Ratio')
        ax.set_title(f'Clusters: {cluster1}/{cluster2} - Mean Wavenumbers: {ratio_df["Wavenumber 1"].mean():.2f}/{ratio_df["Wavenumber 2"].mean():.2f} - {label}')
        ax.legend(loc='upper left')
        plt.tight_layout()

        return fig, best_r2, best_slope, best_intercept, "\n".join(subset_details)

    def generate_plots(self, top_n=60):
        """Generates plots for top N cluster ratios based on sum of peak heights."""
        unique_ratios = self.analysis_df[['Cluster 1', 'Cluster 2', 'Sum of Mean Heights', 'Priority']].drop_duplicates()
        top_ratios = unique_ratios.head(top_n)
        print(f"Generating plots for top {len(top_ratios)} clusters.")

        for _, row in top_ratios.iterrows():
            cluster1 = row['Cluster 1']
            cluster2 = row['Cluster 2']
            priority = row['Priority']
            label = f"Priority: {priority}, Sum: {row['Sum of Mean Heights']}"

            try:
                fig, best_r2, best_slope, best_intercept, subset_details = self.plot_best_fit_ratio_transformed(
                    cluster1, cluster2, priority, label
                )
                print(f"Generated plot for Cluster {cluster1}/{cluster2} with R²={best_r2:.6f}")
                yield fig, subset_details
            except Exception as e:
                error_msg = f"Error plotting Cluster {cluster1}/{cluster2}: {e}"
                print(error_msg)
                yield None, error_msg


if __name__ == "__main__":
    file_path = 'Ratios_analysis.xlsx'
    plotter = ClusterAnalysisPlotter(file_path)
    plotter.load_and_prepare_data()
    plotter.generate_plots(top_n=60)
