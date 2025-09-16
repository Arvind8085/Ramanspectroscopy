import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Baseline_and_filters import apply_butterworth, baseline_als_optimized
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.integrate import simps

class PeakAnalysis2:
    def __init__(self, file_path6):
        self.file_path6 = file_path6
        self.df6 = pd.read_excel(file_path6)
        self.concentration_row6 = self.df6.iloc[0]
        self.df6 = self.df6.drop(0).reset_index(drop=True)
        self.intensity_columns6 = [col for col in self.df6.columns if 'Probe' in col]
        self.concentration_mapping6 = {col: self.concentration_row6[col] for col in self.intensity_columns6}
        self.detected_wavenumbers6 = []
        self.peak_heights6 = []
        self.peak_concentrations6 = []
        self.peak_aucs6 = []
        self.intensity_labels_16 = []

    @staticmethod
    def asymmetric_pseudo_voigt6(x, amplitude, center, sigma_g, sigma_l, asymmetry, eta):
        sigma_g_left = sigma_g * (1 + asymmetry)
        sigma_g_right = sigma_g * (1 - asymmetry)
        sigma_l_left = sigma_l * (1 + asymmetry)
        sigma_l_right = sigma_l * (1 - asymmetry)
        
        left_side = (x < center)
        g_left = np.exp(-((x[left_side] - center) ** 2) / (2 * sigma_g_left ** 2))
        l_left = (sigma_l_left ** 2) / ((x[left_side] - center) ** 2 + sigma_l_left ** 2)
        g_right = np.exp(-((x[~left_side] - center) ** 2) / (2 * sigma_g_right ** 2))
        l_right = (sigma_l_right ** 2) / ((x[~left_side] - center) ** 2 + sigma_l_right ** 2)
        
        pseudo_voigt = np.zeros_like(x)
        pseudo_voigt[left_side] = (1 - eta) * g_left + eta * l_left
        pseudo_voigt[~left_side] = (1 - eta) * g_right + eta * l_right

        return amplitude * pseudo_voigt

    def process_column6(self, intensity_column6, index):
        intensity_data6 = self.df6[intensity_column6].tolist()
        wavenumber_data6 = self.df6['Wavenumber'].tolist()
        concentration6 = self.concentration_mapping6[intensity_column6]
        
        data = np.array([intensity_data6, wavenumber_data6]).T
        filtered_data = data[(data[:, 1] >= 355) & (data[:, 1] <= 1900)]
        original_intensity = filtered_data[:, 0]
        
        order = 9
        cutoff_frequency = 0.1
        filtered_intensity = apply_butterworth(original_intensity, order, cutoff_frequency)
        baseline = baseline_als_optimized(filtered_intensity, smth=1e8, p=0.0001)
        baseline_corrected_intensity = original_intensity - baseline
        difference = filtered_intensity - original_intensity
        noise = np.std(difference)
        filtered_intensity_2 = apply_butterworth(baseline_corrected_intensity, order, cutoff_frequency)
        baseline2 = baseline_als_optimized(filtered_intensity_2, smth=1e8, p=0.0001)
        noise_level2 = noise + baseline2
        above_noise_peaks, _ = find_peaks(filtered_intensity_2, height=noise_level2)
        above_noise_data1 = filtered_intensity_2[above_noise_peaks]
        above_noise_x = filtered_data[:, 1][above_noise_peaks]
        colors = plt.cm.viridis(np.linspace(0, 1, len(above_noise_peaks)))
        for peak_index in range(len(above_noise_peaks)):
            x_peak = above_noise_x[peak_index]
            y_peak = above_noise_data1[peak_index]
            fit_range = (filtered_data[:, 1] >= (x_peak - 20)) & (filtered_data[:, 1] <= (x_peak + 20))
            x_data_fit = filtered_data[:, 1][fit_range]
            y_data_fit = filtered_intensity_2[fit_range]

            initial_guess = [y_peak, x_peak, 5, 5, 0, 0.5]
            bounds = ([0, x_peak - 20, 0, 0, -1, 0], [y_peak * 2, x_peak + 20, 50, 50, 1, 1])
            
            popt, pcov = curve_fit(self.asymmetric_pseudo_voigt6, x_data_fit, y_data_fit, p0=initial_guess, bounds=bounds)
            fitted_curve = self.asymmetric_pseudo_voigt6(filtered_data[:, 1], *popt)
            auc = simps(fitted_curve, filtered_data[:, 1])


            self.detected_wavenumbers6.append(x_peak)
            self.peak_heights6.append(y_peak)
            self.peak_concentrations6.append(concentration6)
            self.peak_aucs6.append(auc)
            self.intensity_labels_16.append(intensity_column6)

    def perform_clustering6(self):
        peak_data = np.array([self.detected_wavenumbers6, self.peak_heights6]).T
        lowest_bic = np.inf
        best_gmm = None
        bic_values = []

        for n_clusters in range(1, 60):
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', init_params='kmeans', random_state=46)
            gmm.fit(peak_data)
            bic = gmm.bic(peak_data)
            bic_values.append(bic)
            
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm

        optimal_clusters = best_gmm.n_components
        cluster_labels = best_gmm.predict(peak_data)
        return cluster_labels

    def save_results(self, cluster_labels):
        df_results = pd.DataFrame({
            'Intensity_Column': self.intensity_labels_16,
            'Wavenumber': self.detected_wavenumbers6,
            'Peak Height': self.peak_heights6,
            'Concentration': self.peak_concentrations6,
            'AUC': self.peak_aucs6,
            'Cluster': cluster_labels
        })

        grouped_1 = df_results.groupby(['Cluster', 'Concentration']).agg(
            mean_wavenumber=('Wavenumber', 'mean'),
            mean_height=('Peak Height', 'mean'),
            mean_AUC=('AUC', 'mean'),
            std_wavenumber=('Wavenumber', 'std'),
            std_height=('Peak Height', 'std'),
            std_AUC=('AUC', 'std')
        ).reset_index()

        output_file = 'peak_data_before_statistics.xlsx'
        with pd.ExcelWriter(output_file) as writer:
            df_results.to_excel(writer, sheet_name='Dataset 1 Peaks', index=False)
        print(f'Peak data has been saved to {output_file}')

        output_file = 'peak_statistics_R4.xlsx'
        with pd.ExcelWriter(output_file) as writer:
            grouped_1.to_excel(writer, sheet_name='Dataset 1 Stats', index=False)
        print(f'Statistics have been saved to {output_file}')

    def run(self):
        for index, intensity_column6 in enumerate(self.intensity_columns6):
            self.process_column6(intensity_column6, index)
        cluster_labels = self.perform_clustering6()
        self.save_results(cluster_labels)

