import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

class SlopeAnalysis:
    def __init__(self, file_paths, output_dirs):
        self.file_path = file_paths
        self.output_dirs = output_dirs
        self.analysis_dfs = None
        self.slope_collection = {}
        self.wavenumber_ratios = {}

    def load_and_prepare_data(self):
        self.analysis_dfs = pd.read_excel(self.file_path, sheet_name='Analysis')
        self.analysis_dfs['Wavenumber 1'] = self.analysis_dfs['Wavenumber 1'].round()
        self.analysis_dfs['Wavenumber 2'] = self.analysis_dfs['Wavenumber 2'].round()
        self.analysis_dfs.sort_values(by=['Wavenumber 1', 'Wavenumber 2'], inplace=True)
        self.analysis_dfs.sort_values(by='Sum of Mean Heights', ascending=False, inplace=True)
        self.analysis_dfs['Priority'] = range(1, len(self.analysis_dfs) + 1)
        self.slope_collection = {priority: [] for priority in self.analysis_dfs['Priority'].unique()}

    def find_best_slope(self, concentrations, mean_peak_heights):
        slope_values = []
        for i in range(len(concentrations) - 2):
            sub_conc = concentrations[i:]
            sub_heights = mean_peak_heights[i:]
            slope, intercept, _, _, _ = linregress(sub_conc, sub_heights)
            slope_values.append(slope)
        return slope_values

    def collect_slope_values(self, cluster1, cluster2, wavenumber1, wavenumber2, priority):
        try:
            ratio_df = self.analysis_dfs[(self.analysis_dfs['Cluster 1'] == cluster1) & (self.analysis_dfs['Cluster 2'] == cluster2)]
            if ratio_df.empty:
                print(f"No data for Cluster {cluster1}/{cluster2}")
                return

            valid_ratio_df = ratio_df.dropna(subset=['Mean Peak Height Ratio', 'Concentration'])
            valid_ratio_df = valid_ratio_df[valid_ratio_df['Mean Peak Height Ratio'] > 0]

            if valid_ratio_df.empty:
                print(f"No valid data for linear regression in Cluster {cluster1}/{cluster2}")
                return

            valid_ratio_df.sort_values(by='Concentration', inplace=True)
            valid_ratio_df['Log Concentration'] = np.log10(valid_ratio_df['Concentration'])

            log_concentrations = valid_ratio_df['Log Concentration'].values
            mean_peak_heights = valid_ratio_df['Mean Peak Height Ratio'].values
            slope_values_for_priority = self.find_best_slope(log_concentrations, mean_peak_heights)

            self.slope_collection[priority] = slope_values_for_priority
            self.wavenumber_ratios[priority] = f"{wavenumber1}/{wavenumber2}"

        except Exception as e:
            print(f"Error occurred while processing Cluster {cluster1}/{cluster2}: {e}")

    def process_all_priorities(self):
        unique_ratios = self.analysis_dfs[['Cluster 1', 'Cluster 2', 'Wavenumber 1', 'Wavenumber 2', 'Sum of Mean Heights', 'Priority']].drop_duplicates()
        for _, row in unique_ratios.iterrows():
            self.collect_slope_values(row['Cluster 1'], row['Cluster 2'], row['Wavenumber 1'], row['Wavenumber 2'], row['Priority'])

    def plot_grouped_slopes(self, max_plots=60):
        os.makedirs(self.output_dirs, exist_ok=True)
        max_subsets = max(len(slopes) for slopes in self.slope_collection.values())
        standard_subset_labels = [f"Subset {k+1} ({max_subsets + 2 - k} conc.)" for k in range(max_subsets)]
        priorities = sorted(self.slope_collection.keys())
        plot_count = 0

        for i in range(0, len(priorities), 5):
            if plot_count >= max_plots:
                break
            fig, ax = plt.subplots(figsize=(10, 6))
            for priority in priorities[i:i+5]:
                if plot_count >= max_plots:
                    break
                slope_values = self.slope_collection[priority]
                slope_values_padded = slope_values + [None] * (max_subsets - len(slope_values))


                # Calculate the mean of Wavenumber 1 and Wavenumber 2 for this priority
                wavenumber_df = self.analysis_dfs[self.analysis_dfs['Priority'] == priority]
                wavenumber1_mean = wavenumber_df['Wavenumber 1'].mean()
                wavenumber2_mean = wavenumber_df['Wavenumber 2'].mean()

                # Update the legend to include the mean values
                label = (f"Priority {priority} "
                        f"({self.wavenumber_ratios[priority]}, "
                        f"Mean: {wavenumber1_mean:.1f}/{wavenumber2_mean:.1f})")
                
                ax.plot(standard_subset_labels, slope_values_padded, marker='o', linestyle='-', markersize=6, linewidth=1.5, label=f'Priority {priority} ({self.wavenumber_ratios[priority]})')

            ax.set_xlabel('Subsets')
            ax.set_ylabel('Slope')
            ax.set_title(f'Slope Values for Priorities {priorities[i]} to {priorities[min(i+4, len(priorities)-1)]}')
            ax.set_xticks(range(len(standard_subset_labels)))
            ax.set_xticklabels(standard_subset_labels, rotation=45)
            ax.legend(title="Wavenumber Ratios", loc="upper right")

            filename = f'grouped_slope_values_priorities_{priorities[i]}_to_{priorities[min(i+4, len(priorities)-1)]}.png'
            fig_path = os.path.join(self.output_dirs, filename)
            plt.show()  # Display the plot interactively
            plt.close(fig)
            plot_count += 1

        print(f"Grouped plots saved in '{self.output_dirs}' directory, up to a maximum of {max_plots} plots.")

# Example usage
if __name__ == "__main__":
    analysis = SlopeAnalysis('Probe_ratios_analysis_processed_oligoSERS_Guanin.xlsx', 'Grouped_Slopes_Guanin')
    analysis.load_and_prepare_data()
    analysis.process_all_priorities()
    analysis.plot_grouped_slopes()
