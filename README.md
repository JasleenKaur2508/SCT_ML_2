👥 Customer Segmentation using K-Means Clustering

🎯 Task 2: Advanced customer segmentation analysis using K-Means clustering with 2D & 3D visualizations

📊 Project Overview

This project applies K-Means clustering to perform customer segmentation analysis using the Mall Customers Dataset. The goal is to identify distinct customer groups based on their demographics and spending behavior, enabling businesses to design targeted marketing strategies.

🎯 What This Project Does

Customer Segmentation Analysis using K-Means

Dataset: Mall Customers (200 records)

Features: Age, Annual Income, Spending Score

Algorithm: K-Means Clustering

Output: Customer segments with profiles, 2D and 3D visualizations

Key Features
✅ Elbow Method: Determines the optimal number of clusters
✅ 2D Visualization: Income vs Spending Score scatter plot
✅ 3D Visualization: Age vs Income vs Spending Score plot
✅ Cluster Profiles: Segment-wise analysis & insights
✅ Business Recommendations: Targeted strategies for each segment

📁 Files Description

task2.py → Main clustering script

Mall_Customers.csv → Dataset file

requirements.txt → Python dependencies

README.md → Documentation

elbow_method.png → Elbow curve visualization

customer_segments_2d.png → 2D clustering result

customer_segments_3d.png → 3D clustering result

🛠️ Tech Stack

🐍 Python 3.9+
📊 Pandas · NumPy · Matplotlib · Seaborn
🤖 Scikit-learn (K-Means)
🎨 mpl_toolkits.mplot3d (3D Visualization)

🚀 Setup and Installation
Prerequisites

Python 3.9 or higher

Virtual environment (recommended)

Installation Steps

Clone the repository

git clone <your-repository-url>
cd SCT_ML_2


Create and activate virtual environment

python3 -m venv .venv
source .venv/bin/activate


Install dependencies

pip install -r requirements.txt

🎯 Running the Script
cd SCT_ML_2
source .venv/bin/activate
python3 task2.py


Expected Output

elbow_method.png: Shows optimal number of clusters

customer_segments_2d.png: Income vs Spending Score clusters

customer_segments_3d.png: 3D Age-Income-Spending visualization

Segment profiles with business insights

📊 Results Summary

Optimal Clusters: 4 customer segments identified

🎯 Segment 0: Budget-Conscious Customers (11%)

Avg Age: 46.6 yrs | Income: $27.2k | Spending Score: 18.8/100

Strategy: Essential products & value deals

🎯 Segment 1: Moderate Customers (49%)

Avg Age: 28.1 yrs | Income: $61.1k | Spending Score: 48.4/100

Strategy: Upselling & cross-selling opportunities

🎯 Segment 2: High Spenders (12%)

Avg Age: 25.2 yrs | Income: $25.8k | Spending Score: 76.9/100

Strategy: Financing options & trendy products

🎯 Segment 3: Stable Customers (28%)

Avg Age: 55.5 yrs | Income: $56.4k | Spending Score: 50.9/100

Strategy: Focus on quality & reliability

🔧 Issues Solved

Cleaned and preprocessed dataset

Handled scaling with StandardScaler

Determined clusters using Elbow Method

Generated high-quality 2D & 3D plots

Added actionable insights for each cluster

📈 Key Insights

Income vs Spending shows clear segmentation patterns

Young customers show higher spending even with lower income

Older customers are more conservative spenders

Businesses can design targeted marketing strategies based on these profiles

🚧 Future Enhancements

🔹 Try other clustering methods (DBSCAN, Hierarchical)
🔹 Add more features (occupation, purchase history)
🔹 Build a Streamlit dashboard for interactive segmentation
🔹 Connect to live customer databases
🔹 Predict customer lifetime value (CLV)

🆘 Troubleshooting

ModuleNotFoundError: Activate venv & install dependencies

FileNotFoundError: Ensure Mall_Customers.csv is in the same folder

Plot display issues: Check matplotlib backend for 3D

🎓 Learning Outcomes

Applied Unsupervised Learning (K-Means)

Used Elbow Method for model evaluation

Created 2D & 3D data visualizations

Derived business insights from clustering

🔥 Customer segmentation is a key tool for business intelligence & targeted marketing!

Made using Python · Machine Learning · Data Visualization
