# 📊 Customer Segmentation Analysis using K-Means Clustering
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Step 1: Load and explore dataset
data = pd.read_csv("Mall_Customers.csv")
print("Dataset Overview:")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\nFirst few rows:")
print(data.head())

# Step 2: Feature selection and preparation
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
print(f"\nSelected features for clustering: {list(features.columns)}")

# Step 3: Feature scaling using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("\nFeatures successfully standardized")

# Step 4: Determining optimal clusters using Elbow Method
inertia_values = []
k_range = range(1, 12)  # Extended range to k=11

for k in k_range:
    model = KMeans(n_clusters=k, init='k-means++', random_state=123, n_init=10)
    model.fit(scaled_features)
    inertia_values.append(model.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10,6))
plt.plot(k_range, inertia_values, marker='s', linewidth=2, markersize=8)
plt.title("Elbow Method for Optimal Number of Clusters", fontsize=14, fontweight='bold')
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()
print("Elbow method plot saved as 'elbow_method.png'")

# Step 5: Apply K-Means clustering (using k=4 instead of 5)
optimal_k = 4
final_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=123, n_init=10)
cluster_labels = final_model.fit_predict(scaled_features)

# Step 6: Add cluster assignments to original dataset
data['Customer_Segment'] = cluster_labels
print(f"\nClustering completed with {optimal_k} segments")

# Step 7: 2D Scatter Plot (Income vs Spending Score)
plt.figure(figsize=(12,8))
scatter_plot = sns.scatterplot(
    data=data, 
    x='Annual Income (k$)', 
    y='Spending Score (1-100)',
    hue='Customer_Segment', 
    palette='viridis',  # Different color palette
    s=120,  # Larger points
    alpha=0.8
)
plt.title("Customer Segmentation Analysis (Income vs Spending Behavior)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Annual Income (k$)", fontsize=12)
plt.ylabel("Spending Score (1-100)", fontsize=12)
plt.legend(title='Customer Segments', title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('customer_segments_2d.png', dpi=300, bbox_inches='tight')
plt.close()
print("2D segmentation plot saved as 'customer_segments_2d.png'")

# Step 8: 3D Visualization with enhanced styling
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

# Create 3D scatter plot
colors = ['red', 'blue', 'green', 'orange']  # Custom color scheme
for i in range(optimal_k):
    cluster_data = data[data['Customer_Segment'] == i]
    ax.scatter(cluster_data['Age'], 
              cluster_data['Annual Income (k$)'], 
              cluster_data['Spending Score (1-100)'],
              c=colors[i], 
              label=f'Segment {i}',
              s=100,
              alpha=0.7)

ax.set_xlabel("Age (years)", fontsize=12)
ax.set_ylabel("Annual Income (k$)", fontsize=12)
ax.set_zlabel("Spending Score (1-100)", fontsize=12)
ax.set_title("3D Customer Segmentation Visualization", fontsize=14, fontweight='bold')
ax.legend(title="Customer Segments")

# Rotate view for better perspective
ax.view_init(elev=20, azim=45)
plt.savefig('customer_segments_3d.png', dpi=300, bbox_inches='tight')
plt.close()
print("3D segmentation plot saved as 'customer_segments_3d.png'")

# Step 9: Detailed Cluster Analysis
print("\n" + "="*60)
print("CUSTOMER SEGMENT ANALYSIS")
print("="*60)

segment_summary = data.groupby('Customer_Segment')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].agg(['mean', 'std', 'count'])
print("\nDetailed Segment Statistics:")
print(segment_summary.round(2))

# Step 10: Business Insights
print("\n" + "="*40)
print("BUSINESS INSIGHTS BY SEGMENT")
print("="*40)

for segment in range(optimal_k):
    segment_data = data[data['Customer_Segment'] == segment]
    avg_age = segment_data['Age'].mean()
    avg_income = segment_data['Annual Income (k$)'].mean()
    avg_spending = segment_data['Spending Score (1-100)'].mean()
    count = len(segment_data)
    
    print(f"\nSegment {segment} ({count} customers):")
    print(f"  • Average Age: {avg_age:.1f} years")
    print(f"  • Average Income: ${avg_income:.1f}k")
    print(f"  • Average Spending Score: {avg_spending:.1f}/100")
    
    # Simple business interpretation
    if avg_income > 60 and avg_spending > 60:
        print(f"  • Profile: High-Value Customers (Premium Target)")
    elif avg_income < 40 and avg_spending < 40:
        print(f"  • Profile: Budget-Conscious Customers")
    elif avg_spending > 60:
        print(f"  • Profile: High Spenders (Marketing Focus)")
    else:
        print(f"  • Profile: Moderate Customers (Growth Potential)")

print("\n" + "="*60)