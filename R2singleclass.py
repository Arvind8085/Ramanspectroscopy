import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


class R2Analysis:
    def __init__(self, file_path, output_dir):
        self.file_path = file_path
        self.output_dir = output_dir
        self.analysis_df = None
        self.r2_collection = {}
        self.cluster_ratios = {}

    def load_and_process_data(self):
        """Load and preprocess data from the Excel file."""
        self.analysis_df = pd.read_excel(self.file_path, sheet_name='Analysis')
        self.analysis_df['Wavenumber 1'] = self.analysis_df['Wavenumber 1'].round()
        self.analysis_df['Wavenumber 2'] = self.analysis_df['Wavenumber 2'].round()
        self.analysis_df.sort_values(by=['Wavenumber 1', 'Wavenumber 2'], inplace=True)
        self.analysis_df.sort_values(by='Sum of Mean Heights', ascending=False, inplace=True)
        self.analysis_df['Priority'] = range(1, len(self.analysis_df) + 1)
        self.r2_collection = {int(priority): [] for priority in self.analysis_df['Priority'].unique()}

    @staticmethod
    def find_best_r2(concentrations, mean_peak_heights):
        """Calculate R² values for progressively smaller subsets."""
        r2_values = []
        for i in range(len(concentrations) - 2):  # Ensure at least 3 concentrations
            sub_conc = concentrations[i:]
            sub_heights = mean_peak_heights[i:]
            _, _, r_value, _, _ = linregress(sub_conc, sub_heights)
            r2_values.append(r_value**2)
        return r2_values

    def collect_r2_values(self, cluster1, cluster2, priority, Wavenumber1, Wavenumber2):
        """Collect R² values for a given cluster pair and priority."""
        try:
            ratio_df = self.analysis_df[
                (self.analysis_df['Cluster 1'] == cluster1) &
                (self.analysis_df['Cluster 2'] == cluster2)
            ]
            if ratio_df.empty:
                print(f"No data for Cluster {cluster1}/{cluster2}")
                return

            valid_ratio_df = ratio_df.dropna(subset=['Mean Peak Height Ratio', 'Concentration'])
            valid_ratio_df = valid_ratio_df[valid_ratio_df['Mean Peak Height Ratio'] > 0]
            if valid_ratio_df.empty:
                print(f"No valid data for Cluster {cluster1}/{cluster2}")
                return

            valid_ratio_df = valid_ratio_df.sort_values(by='Concentration')  # Avoid SettingWithCopyWarning
            valid_ratio_df['Log Concentration'] = np.log10(valid_ratio_df['Concentration'])

            log_concentrations = valid_ratio_df['Log Concentration'].values
            mean_peak_heights = valid_ratio_df['Mean Peak Height Ratio'].values

            r2_values_for_priority = self.find_best_r2(log_concentrations, mean_peak_heights)

            if r2_values_for_priority:
                self.r2_collection[int(priority)] = r2_values_for_priority
                self.cluster_ratios[int(priority)] = f"{cluster1}/{cluster2}"
            else:
                print(f"No R² values calculated for Cluster {cluster1}/{cluster2}")

        except Exception as e:
            print(f"Error processing Cluster {cluster1}/{cluster2}: {e}")

    def generate_r2_collection(self):
        """Iterate through unique cluster pairs and calculate R² values."""
        unique_ratios = self.analysis_df[['Cluster 1', 'Cluster 2', 'Priority', 'Wavenumber 1', 'Wavenumber 2']].drop_duplicates()
        for _, row in unique_ratios.iterrows():
            priority = int(row['Priority'])
            self.collect_r2_values(row['Cluster 1'], row['Cluster 2'], priority, row['Wavenumber 1'], row['Wavenumber 2'])

    def plot_r2_values(self, max_plots=60):
        """Plot R² values and display them in a grid layout."""
        priorities = list(self.r2_collection.keys())
        displayed_plots = 0
        mean_wn1 = self.analysis_df["Wavenumber 1"].mean()
        mean_wn2 = self.analysis_df["Wavenumber 2"].mean()

        for i in range(0, len(priorities), 12):  # Display 3x4 grid at a time
            if displayed_plots >= max_plots:
                break

            fig, axes = plt.subplots(3, 4, figsize=(18, 12))
            fig.subplots_adjust(hspace=0.5, wspace=0.4)
            axes = axes.flatten()

            for j, priority in enumerate(priorities[i:i + 12]):
                if displayed_plots >= max_plots:
                    break

                r2_values = self.r2_collection.get(priority, [])
                if not r2_values:
                    print(f"No R² values for Priority {priority}. Skipping.")
                    continue

                subset_labels = [f"Subset {k + 1} ({len(r2_values) + 2 - k} conc.)" for k in range(len(r2_values))]
                ax = axes[j]

                ax.bar(np.arange(len(subset_labels)), r2_values, color='skyblue')
                ax.set_xticks(np.arange(len(subset_labels)))
                ax.set_xticklabels(subset_labels, rotation=45)
                ax.set_xlabel('Subsets')
                ax.set_ylabel('R² Values')
                ax.set_title(
                    f'R² for Priority {priority}\nCluster: {self.cluster_ratios.get(priority, "?")}, '
                    f'WNs: {mean_wn1:.2f}/{mean_wn2:.2f}'
                )
                displayed_plots += 1

            # Hide any unused axes
            for k in range(j + 1, 12):
                fig.delaxes(axes[k])

            plt.tight_layout()
            plt.show()

        print(f"Displayed total plots: {displayed_plots}")

    def run_analysis(self):
        """Execute the full analysis pipeline."""
        self.load_and_process_data()
        self.generate_r2_collection()
        self.plot_r2_values()


# Example usage
if __name__ == "__main__":
    file_path = 'Ratios_analysis.xlsx'
    output_dir = 'output_plots_R2_GuaninT'

    analysis = R2Analysis(file_path, output_dir)
    analysis.run_analysis()
