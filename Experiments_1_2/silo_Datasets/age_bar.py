import pandas as pd,os
import matplotlib.pyplot as plt
silo_data = '/home/tanurima/germany/brain_age_parcels/kunal/Filtered_Datasets'

# Read the CSV files
df1 = pd.read_csv(os.path.join(silo_data,'CamCAN/Train_CamCAN.csv').replace('Train','Test'))
df2 = pd.read_csv(os.path.join(silo_data,'eNki/Train_eNki.csv').replace('Train','Test'))
df3 = pd.read_csv(os.path.join(silo_data,'SALD/Train_SALD.csv').replace('Train','Test'))

# Create a figure with three subplots (one below the other)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

# Plot age distribution for the first dataset
axes[0].hist(df1['age'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('CamCAN Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

# Plot age distribution for the second dataset
axes[1].hist(df2['age'], bins=20, color='salmon', edgecolor='black')
axes[1].set_title('eNki Age Distribution')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')

# Plot age distribution for the third dataset
axes[2].hist(df3['age'], bins=20, color='lightgreen', edgecolor='black')
axes[2].set_title('SALD Age Distribution')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Frequency')

# Adjust layout so titles and labels don't overlap
plt.tight_layout()
#plt.title('')

# Display the plots
#plt.show()
plt.savefig('bar-plot-age-te-distribution.png')
