import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

class BarPlotGenerator:
    def __init__(self, file_path4):
        self.file_path4 = file_path4
        self.cluster_stats_df4 = pd.read_excel(file_path4)
        self.df_peaks4 = self._prepare_dataframe()

    def _prepare_dataframe(self):
        """
        Prepare the DataFrame for plotting by rounding the wavenumber and selecting relevant columns.
        """
        self.cluster_stats_df4['mean_wavenumber'] = self.cluster_stats_df4['mean_wavenumber'].round(0)
        return self.cluster_stats_df4[['Concentration', 'mean_wavenumber', 'mean_height', 'std_height']]

    @staticmethod
    
    def _plot_bar_and_regression(df_subset, ax, wavenumber):
        """
        Plot a single bar plot with regression for a given wavenumber.
        """
        # Convert concentration to float and apply log10 transformation
        df_subset = df_subset.copy()
        df_subset['Concentration'] = df_subset['Concentration'].apply(lambda x: float(str(x).replace("10^", "1E")))

        concentrations = df_subset['Concentration'].unique()
        concentrations.sort()
        concentrations_log = np.log10(concentrations)

        # Group by concentration and calculate mean and standard deviation
        grouped = df_subset.groupby('Concentration')
        height_means = grouped['mean_height'].mean().reindex(concentrations)
        height_stds = grouped['std_height'].mean().reindex(concentrations)

        # Plot mean peak height as red dots with error bars
        if not height_stds.isnull().all():
            ax.errorbar(concentrations_log, height_means, yerr=height_stds, fmt='o', color='red',
                        ecolor='blue', capsize=5, label='Mean Height ± Std Dev')
        else:
            ax.plot(concentrations_log, height_means, 'o', color='red', label='Mean Height (No Std Dev)')

        # Perform linear regression and plot the regression line
        slope, intercept, r_value, p_value, std_err = linregress(concentrations_log, height_means)
        reg_line = slope * concentrations_log + intercept
        ax.plot(concentrations_log, reg_line, color='green', linestyle='--', label='Regression Line')

        # Set labels and title
        ax.set_title(f'Wavenumber {int(wavenumber)}')
        ax.set_xlabel('Log Concentration')
        ax.set_ylabel('Mean Height')

        # Add R² and slope annotation
        r_squared = r_value**2
        ax.text(0.05, 0.95, f'$R^2$ = {r_squared:.6f}\nSlope = {slope:.6f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        ax.legend(loc='upper right')


    def save_single_bar_plots(self, title):
        """
        Generate and save bar plots for each wavenumber one at a time.
        """
        unique_wavenumbers = self.df_peaks4['mean_wavenumber'].unique()
        
        # Iterate through each wavenumber
        for wavenumber in unique_wavenumbers:
            fig, ax = plt.subplots(figsize=(8, 6))  # Create a single plot
            fig.suptitle(f"{title} - Wavenumber {int(wavenumber)}", fontsize=16)
            
            try:
                # Filter the DataFrame for the current wavenumber
                subset_df = self.df_peaks4[self.df_peaks4['mean_wavenumber'] == wavenumber]
                
                # Plot the bar chart and regression line for this wavenumber
                self._plot_bar_and_regression(subset_df, ax, wavenumber)
            
            except Exception as e:
                print(f"Skipping wavenumber {wavenumber} due to error: {e}")
                continue
            
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to include the title
            
            plt.show()  # Display the plot
            plt.close(fig)  # Close the figure to free memory


