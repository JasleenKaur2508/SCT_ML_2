# 📊 Customer Segmentation Analysis using K-Means Clustering

A comprehensive machine learning project that performs **customer segmentation analysis** using the **K-Means clustering algorithm** to identify distinct customer groups based on their demographics and spending behavior.

---

## 📊 Project Overview

This project implements **customer segmentation** using **unsupervised machine learning (K-Means clustering)**.  
The goal is to identify groups of mall customers with similar characteristics and provide **business insights** for targeted marketing.

---

## 🎯 What This Project Does

### **Customer Segmentation Analysis**
- **Dataset**: Mall Customers Dataset (200 customers)  
- **Algorithm**: K-Means Clustering  
- **Features**: Age, Annual Income, Spending Score  
- **Output**: Distinct customer segments with **2D & 3D visualizations**  

### **Key Features**
- ✅ **Elbow Method**: Automatically determines optimal number of clusters  
- ✅ **2D Visualization**: Income vs Spending Score scatter plot  
- ✅ **3D Visualization**: Age vs Income vs Spending Score interactive plot  
- ✅ **Cluster Profiles**: Statistical analysis of each segment  
- ✅ **Business Insights**: Marketing strategies for each group  

---

## 📁 Files Description

- `task2.py` – Main analysis script  
- `Mall_Customers.csv` – Customer dataset (200 records)  
- `requirements.txt` – Python dependencies  
- `README.md` – Documentation file  
- `elbow_method.png` – Elbow method visualization  
- `customer_segments_2d.png` – 2D scatter plot  
- `customer_segments_3d.png` – 3D visualization  

---

## 🛠️ Tech Stack

🐍 **Python**  
📊 **Pandas** · **NumPy** · **Matplotlib** · **Seaborn**  
🤖 **Scikit-learn (K-Means Clustering)**  
🎨 **3D Visualization** (`mpl_toolkits.mplot3d`)  

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8 or higher  
- Virtual environment (recommended)  

### Installation Steps

1. **Clone the repository**  
   ```bash
   git clone <your-repository-url>
   cd SCT_ML_2
   ```

**Important**: Make sure to run the script from within the SCT_ML_2 directory so it can find the Mall_Customers.csv file.

### **Expected Output:**
1. **Elbow Method Plot**: Shows optimal number of clusters (K=4)
2. **2D Scatter Plot**: Customer segments based on Income vs Spending Score
3. **3D Visualization**: Age vs Income vs Spending Score interactive plot
4. **Cluster Profiles**: Statistical summary of each customer segment

---

## 📊 Results Summary

### **Customer Segmentation Results:**
- **Optimal Clusters**: 4 customer segments identified
- **Segments Identified**: Different customer profiles based on age, income, and spending patterns

### **Cluster Analysis:**
   ```
Segment 0: Budget-Conscious Customers → Older, low income, minimal spending
Segment 1: Moderate Customers → Young professionals, good income, moderate spending
Segment 2: High Spenders → Young, lower income but high spending tendency
Segment 3: Stable Customers → Mature, moderate income & spending
 
   ```

---

## 📈 Key Insights

### **Customer Behavior Patterns:**
💰 Income vs Spending: Strong correlation with lifestyle choices

👥 Age Groups: Younger customers tend to spend more despite income differences

🎯 Target Segments: High-value and underserved customer groups identified

📊 Business Strategies: Targeted marketing campaigns and product positioning

---

## 🚧 Future Enhancements

🔹 **Advanced Clustering**: Try DBSCAN, Hierarchical Clustering  
🔹 **Feature Engineering**: Add more customer attributes  
🔹 **Interactive Dashboard**: Create Streamlit web app  
🔹 **Real-time Analysis**: Connect to live customer data  
🔹 **Predictive Modeling**: Add customer lifetime value prediction  

---

## 🆘 Troubleshooting

If you encounter issues:

1. **ModuleNotFoundError**: Make sure the virtual environment is activated and packages are installed
2. **FileNotFoundError**: Ensure `Mall_Customers.csv` is in the same folder as the script
3. **Display Issues**: The scripts generate matplotlib plots - ensure you have a display environment
4. **3D Plot Issues**: Some environments may not support 3D plots - check your matplotlib backend

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Unsupervised Learning**: K-Means clustering implementation
- **Data Visualization**: 2D and 3D plotting techniques
- **Customer Analytics**: Business insights from customer data
- **Model Evaluation**: Elbow method for optimal cluster selection

---

🔥 **Customer segmentation is a powerful tool for business intelligence and targeted marketing strategies!**

---

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/) 
[![Machine Learning](https://img.shields.io/badge/Topic-Clustering-green)]()
[![Data Visualization](https://img.shields.io/badge/Topic-Data%20Visualization-orange)]()
