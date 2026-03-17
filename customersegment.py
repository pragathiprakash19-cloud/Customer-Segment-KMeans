# Customer Segmentation Project

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("customer_data.csv")

# Display first rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Info")
print(df.info())

# Check missing values
print("\nMissing Values")
print(df.isnull().sum())

# Check column names (IMPORTANT)
print("\nColumns:")
print(df.columns)

# Select relevant columns (update if needed)
X = df[['Age', 'Annual Income', 'Spending Score']]

# Scaling (improves clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot Age vs Income
plt.scatter(df['Age'], df['Annual Income'])
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.title("Age vs Income")
plt.show()

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Apply KMeans (Optimal = 5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.scatter(df['Annual Income'],
            df['Spending Score'],
            c=df['Cluster'])

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:,1], centroids[:,2],
            s=200, c='red', label='Centroids')

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.legend()
plt.show()

# Cluster summary
print("\nCluster Summary")
print(df.groupby('Cluster').mean())

# Save result
df.to_csv("customer_segments.csv", index=False)

print("Customer segmentation completed successfully!")