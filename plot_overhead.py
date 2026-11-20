import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'Source Code Pro'}
mpl.rc('font', **font)


# Function to read CSV file and extract columns
def read_csv(filename):
    x = []
    y = []
    z = []
    
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            x.append(float(row[0]))  # x values
            y.append(float(row[1])*1000)  # y values
            z.append(float(row[2])*1000)  # z values (standard deviation)
    
    return np.array(x), np.array(y), np.array(z)

# Function to plot x, y with error bars using z
def plot_with_error_bars(x, y, z):
    plt.errorbar(x, y, yerr=z, fmt='o-', capsize=5, label="Data with error bars")
    plt.xlabel('Number of classifiers', fontsize=14)
    plt.ylabel('Execution time (ms)', fontsize=14)
    plt.grid(True)
    plt.savefig("outfigs/overhead.pdf",  bbox_inches='tight')

# Main script to read the CSV file and plot the data
if __name__ == "__main__":
    # Replace with your actual file path
    filename = 'results/overhead.csv'
    
    # Read data from CSV file
    x, y, z = read_csv(filename)
    
    # Plot data with error bars
    plot_with_error_bars(x, y, z)