import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

class BarPlotGenerator1:
    def __init__(self, file_path1):
        self.file_path1 = file_path1
        self.cluster_stats_df1 = pd.read_excel(file_path1)
        self.df_peaks1 = self._prepare_dataframe1()

    def _prepare_dataframe1(self):
        """
        Prepare the DataFrame for plotting by rounding the wavenumber and selecting relevant columns.
        """
        self.cluster_stats_df1['mean_wavenumber'] = self.cluster_stats_df1['mean_wavenumber'].round(0)
        return self.cluster_stats_df1[['Concentration', 'mean_wavenumber', 'mean_AUC', 'std_AUC']]

    @staticmethod
    def _plot_bar_and_regression1(df_subset1, ax1, wavenumber1):
        """
        Plot a single bar plot with regression for a given wavenumber.
        """
        df_subset1 = df_subset1.copy()
        df_subset1['Concentration'] = df_subset1['Concentration'].apply(lambda x: float(str(x).replace("10^", "1E")))

        concentrations = df_subset1['Concentration'].unique()
        concentrations.sort()
        concentrations_log = np.log10(concentrations)

        # Group by concentration and calculate mean and standard deviation
        grouped1 = df_subset1.groupby('Concentration')
        height_means1 = grouped1['mean_AUC'].mean().reindex(concentrations)
        height_stds1 = grouped1['std_AUC'].mean().reindex(concentrations)

        # Plot bar chart with error bars or bars only
        if not height_stds1.isnull().all():
            ax1.errorbar(concentrations_log, height_means1, yerr=height_stds1, fmt='o', color='red',
                        ecolor='blue', capsize=5, label='Mean AUC ± Std Dev')
        else:
            ax1.bar(concentrations_log, height_means1, color='red', alpha=0.6, width=0.05, label='Mean AUC (No Std Dev)')

        # Perform linear regression on mean values
        slope, intercept, r_value, p_value, std_err = linregress(concentrations_log, height_means1)
        reg_line = slope * concentrations_log + intercept
        ax1.plot(concentrations_log, reg_line, color='green', linestyle='--', label='Regression Line')

        # Add R² and slope label
        r_squared = r_value ** 2
        ax1.text(0.05, 0.95, f'$R^2$ = {r_squared:.6f}\nSlope = {slope:.6f}',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        # Set labels and title
        ax1.set_title(f'Wavenumber {int(wavenumber1)}')
        ax1.set_xlabel('Log Concentration')
        ax1.set_ylabel('Mean AUC')
        ax1.legend(loc='upper right')

    def save_2_bars_in_pairs1(self, title):
        """
        Generate and save bar plots for two wavenumbers at a time in a 1x2 grid layout.
        """
        unique_wavenumbers1 = self.df_peaks1['mean_wavenumber'].unique()
        
        # Iterate through each wavenumber
        for wavenumber1 in unique_wavenumbers1:
            fig1, ax1 = plt.subplots(figsize=(8, 6))  # Create a single plot
            fig1.suptitle(f"{title} - Wavenumber {int(wavenumber1)}", fontsize=16)
            
            try:
                # Filter the DataFrame for the current wavenumber
                subset_df1 = self.df_peaks1[self.df_peaks1['mean_wavenumber'] == wavenumber1]
                
                # Plot the bar chart and regression line for this wavenumber
                self._plot_bar_and_regression1(subset_df1, ax1, wavenumber1)
            
            except Exception as e:
                print(f"Skipping wavenumber {wavenumber1} due to error: {e}")
                continue
            
            fig1.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include the title
            
            plt.show()  # Display the plot
            plt.close(fig1)  # Close the figure to free memory


