import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'Source Code Pro'}
mpl.rc('font', **font)


csv_file = 'results_realdata/varying_lambda.csv'

x = []
y_FIRST = []
y_LAST = []
y_OPTBUILD = []
y_OPTALG = []
y_SOA0 = []
y_SOA1 = []
y_SOA2 = []
y_SOA3 = []

# Open and read the CSV file
with open(csv_file, newline='') as file:
    reader = csv.reader(file)
    
    # Loop through each row and extract the relevant columns
    for row in reader:
        x.append(float(row[0]))  # First column is x
        y_FIRST.append(float(row[1]))  # Second column is y1
        y_LAST.append(float(row[3]))  # Fourth column is y2
        y_OPTBUILD.append(float(row[5]))  # Sixth column is y3
        y_OPTALG.append(float(row[7]))
        y_SOA0.append(float(row[9]))
        y_SOA1.append(float(row[11]))
        y_SOA2.append(float(row[13]))
        y_SOA3.append(float(row[15]))

# Merge SOA
y_SOA_merged = []
for i in range(0,len(y_SOA0)):
    y_SOA_merged.append(min(y_SOA0[i], y_SOA1[i], y_SOA2[i], y_SOA3[i]))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each line
plt.plot(x, y_OPTBUILD, label='FIRST', linestyle=':', color='b', linewidth=3)
plt.plot(x, y_LAST, label='BOUNDARY', linestyle='--', color='g', linewidth=3)
plt.plot(x, y_OPTALG, label='OPT-ALG', color='y', linewidth=3)
plt.plot(x, y_SOA_merged, label='SKIP', linestyle='-.', color='r', linewidth=3)

# Set axes (REAL DATA: y range is auto-scaled, the WCET units are now real MACs)
plt.ylim(auto=True)
plt.xlim((0, max(x)*1.03))

# Add labels and title
plt.xlabel('measured $\\lambda$ (real UCI HAR stream)', fontsize=18)
plt.ylabel('Average execution time (MACs)', fontsize=18)

# Show legend
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Display the plot
plt.grid(True)
plt.savefig("outfigs_realdata/lambda_single_1.pdf", bbox_inches='tight')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each line
plt.plot(x, y_SOA0, label='SKIP(0)', linestyle='-.', color='b', linewidth=3)
plt.plot(x, y_SOA1, label='SKIP(1)', linestyle=':', color='r', linewidth=3)
plt.plot(x, y_SOA2, label='SKIP(2)', linestyle=(0, (3.5, 1, 5)),  color='#550', linewidth=3)
plt.plot(x, y_SOA3, label='SKIP(3)', linestyle='-.', color='#055', linewidth=3)

# Set axes (REAL DATA: y range is auto-scaled)
plt.ylim(auto=True)
plt.xlim((0, max(x)*1.03))

# Add labels and title
plt.xlabel('measured $\\lambda$ (real UCI HAR stream)', fontsize=18)
plt.ylabel('Average execution time (MACs)', fontsize=18)

# Show legend
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Display the plot
plt.grid(True)
plt.savefig("outfigs_realdata/lambda_single_2.pdf", bbox_inches='tight')

print("# Statistics:")

print("OPTALG vs FIRST min", -np.min(np.subtract(y_OPTALG, y_OPTBUILD) / y_OPTBUILD)*100)
print("OPTALG vs BOUNDARY min", -np.min(np.subtract(y_OPTALG, y_LAST) / y_LAST)*100)
print("OPTALG vs SOA min", -np.min(np.subtract(y_OPTALG, y_SOA_merged) / y_SOA_merged)*100)

print("OPTALG vs FIRST mean", -np.mean(np.subtract(y_OPTALG, y_OPTBUILD) / y_OPTBUILD)*100)
print("OPTALG vs BOUNDARY mean", -np.mean(np.subtract(y_OPTALG, y_LAST) / y_LAST)*100)
print("OPTALG vs SOA mean", -np.mean(np.subtract(y_OPTALG, y_SOA_merged) / y_SOA_merged)*100)

print("OPTALG vs FIRST max", -np.max(np.subtract(y_OPTALG, y_OPTBUILD) / y_OPTBUILD)*100)
print("OPTALG vs BOUNDARY max", -np.max(np.subtract(y_OPTALG, y_LAST) / y_LAST)*100)
print("OPTALG vs SOA max", -np.max(np.subtract(y_OPTALG, y_SOA_merged) / y_SOA_merged)*100)
