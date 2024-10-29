
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib import rc
import os
import matplotlib.ticker as ticker

# Set font properties
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Standard plot settings
PLOT_SETTINGS = {
    'figsize': (4, 4),
    'dpi': 300,
    'spine_color': 'black',
    'spine_linewidth': 2.5,
    'fontsize': 15
}

# Gaussian function definition
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Load and normalize data for LENS and CSP
def load_and_normalize_data(lens_file_path, cs_file_path):
    lens_data = np.load(lens_file_path)['arr_0']  
    cs_data = np.loadtxt(cs_file_path).T  

    max_value = np.max(lens_data)
    second_max_value = np.max(lens_data[lens_data < max_value])
    lens_data = np.clip(lens_data / second_max_value, 0, 1)  

    cs_data_min, cs_data_max = np.min(cs_data), np.max(cs_data)
    cs_data = (cs_data - cs_data_min) / (cs_data_max - cs_data_min)

    return lens_data, cs_data

# Calculate time differences between peaks in LENS and CSP
def calculate_max_time_deltas(lens_data, cs_data):
    time_conversion = 10 / 5000  
    deltas = []

    for lens_row, cs_row in zip(lens_data, cs_data):
        lens_max_idx = np.argmax(lens_row)
        cs_max_idx = np.argmax(cs_row)
        delta_time = (cs_max_idx - lens_max_idx) * time_conversion
        deltas.append(delta_time)

    return np.array(deltas)

# Fit Gaussian to data using curve_fit or minimize
def fit_gaussian(x, y, p0):
    try:
        popt, _ = curve_fit(gaussian, x, y, p0=p0, maxfev=5000)
    except RuntimeError:
        def gaussian_residuals(params, x, y):
            A, mu, sigma = params
            return np.sum((gaussian(x, A, mu, sigma) - y) ** 2)
        result = minimize(gaussian_residuals, p0, args=(x, y), method='BFGS')
        popt = result.x
    return popt

# Plot histogram and fit Gaussian curves
def plot_histogram_with_gaussian_fits(deltas, output_path, max_delay=4, bin_width=0.5, color='gray'):
    deltas = deltas[(deltas > -max_delay) & (deltas < max_delay)]
    bins = np.arange(-max_delay - bin_width / 2, max_delay + bin_width / 2, bin_width)
    hist, bin_edges = np.histogram(deltas, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'])
    plt.bar(bin_centers, hist, width=bin_width, color=color, alpha=0.6, edgecolor='black')

    central_mask = (bin_centers > -2.0) & (bin_centers < 2.0)
    p0_central = [np.max(hist), 0, 0.8]
    popt_central = fit_gaussian(bin_centers[central_mask], hist[central_mask], p0_central)
    x_smooth_central = np.linspace(bin_centers[central_mask].min(), bin_centers[central_mask].max(), 100)
    plt.plot(x_smooth_central, gaussian(x_smooth_central, *popt_central), 'black', lw=2)

    right_mask = (bin_centers > 0.5) & (bin_centers < 4)
    p0_right = [np.max(hist[right_mask]), 2.5, 0.5]
    popt_right = fit_gaussian(bin_centers[right_mask], hist[right_mask], p0_right)
    x_smooth_right = np.linspace(bin_centers[right_mask].min(), bin_centers[right_mask].max(), 100)
    plt.plot(x_smooth_right, gaussian(x_smooth_right, *popt_right), 'blue', lw=2)

    plt.fill_between(x_smooth_right, gaussian(x_smooth_right, *popt_right), color='blue', alpha=0.3, hatch='///')
    plt.ylabel('Frequency ($\times 10^3$)', fontsize=PLOT_SETTINGS['fontsize'])
    plt.xlabel('CS peak - LENS peak [ns]', fontsize=PLOT_SETTINGS['fontsize'])
    plt.xticks(fontsize=PLOT_SETTINGS['fontsize'])
    plt.yticks(fontsize=PLOT_SETTINGS['fontsize'])
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Plot data with summed trace and KDE
def plot_data_with_sum(data, sum_data, trace_color, sum_color, label, output_path, time_unit='ns'):
    fig, (ax, ax_kde) = plt.subplots(1, 2, figsize=(10, 6), dpi=300, width_ratios=[3, 1])
    time_conversion = 10 / 5000
    x = np.linspace(0, len(data[0]) * time_conversion, len(data[0]))

    segments = [np.column_stack([x, data[p, :]]) for p in range(data.shape[0])]
    lc = LineCollection(segments, colors=trace_color, linewidths=0.1, alpha=0.2)
    ax.add_collection(lc)

    sum_data = (sum_data - np.min(sum_data)) / (np.max(sum_data) - np.min(sum_data))
    ax.plot(x, sum_data, color=sum_color, linewidth=3, label=f'Sum of {label.capitalize()} Values (Normalized)')

    flattened_data = data.flatten()
    thresholds = [0.4, 0.25, 0.1]
    colors = ['darkblue', 'lightblue', 'lightskyblue'] if label == 'lens' else ['darkred', 'lightcoral', 'lightsalmon']

    ax_kde.set_ylim(ax.get_ylim())
    for idx, threshold in enumerate(thresholds):
        kde_data = flattened_data[(flattened_data > thresholds[idx]) & (flattened_data <= thresholds[idx - 1])] if idx else flattened_data[flattened_data > threshold]
        sns.kdeplot(y=kde_data, bw_adjust=0.8, linewidth=1.5, color=colors[idx], gridsize=50, ax=ax_kde, fill=True)

    ax.set_ylabel('LENS' if label == 'lens' else 'CSP', size=20)
    ax.set_xlabel(f'Time [{time_unit}]', size=20)
    ax_kde.set_xticklabels([])
    ax_kde.set_yticks([])
    plt.subplots_adjust(wspace=0.08, hspace=0.3)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Main analysis function
def analyze_replicas(base_dir, replicas, max_delay=4, bin_width=0.5):
    for replica in replicas:
        lens_file_path = os.path.join(base_dir, replica, 'LENS', 'lens_scnemd.npz')
        cs_file_path = os.path.join(base_dir, replica, 'CS', 'CS.dat')
        output_dir = os.path.join(base_dir, f'output_peak_analysis_{replica}')
        
        os.makedirs(output_dir, exist_ok=True)
        
        lens_data, cs_data = load_and_normalize_data(lens_file_path, cs_file_path)
        deltas = calculate_max_time_deltas(lens_data, cs_data)
        
        plot_histogram_with_gaussian_fits(deltas, os.path.join(output_dir, 'histogram_with_gaussian_fits.png'), max_delay=max_delay, bin_width=bin_width)
        plot_data_with_sum(lens_data, np.sum(lens_data, axis=0), 'dodgerblue', 'darkblue', 'lens', os.path.join(output_dir, 'lens_with_kde.png'))
        plot_data_with_sum(cs_data, np.sum(cs_data, axis=0), 'red', 'darkred', 'csp', os.path.join(output_dir, 'csp_with_kde.png'))

# Example usage
replica_dirs = [f'replica_{i}' for i in range(1, 6)]
base_directory = '.'
analyze_replicas(base_directory, replica_dirs)
