# 🛍️ RetailPulse – Customer Shopping Intelligence Platform

An end-to-end data analytics project that transforms customer shopping data into meaningful business insights through **Python, SQL, and an interactive Streamlit dashboard**.

RetailPulse explores customer purchasing patterns, product performance, subscription behavior, marketing effectiveness, and customer segments to help businesses make data-driven decisions.

## 📌 Project Overview

The project uses a customer shopping behavior dataset containing **3,900 customer records and 18 attributes**. It combines exploratory data analysis, data cleaning, SQL-based business analysis, interactive visualizations, and machine learning-based customer segmentation.

The goal is to understand customer behavior, identify purchasing trends, and uncover opportunities to improve marketing and sales strategies.

## 🛠️ Tech Stack

- **Python:** Pandas, NumPy, Scikit-learn
- **SQL:** PostgreSQL
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Machine Learning:** K-Means Clustering
- **Tools:** Jupyter Notebook, Git, GitHub

## 📊 Project Workflow

1. **Data Exploration & Cleaning**
   - Examined dataset structure, data types, and statistical summaries.
   - Identified missing values and performed data cleaning.
   - Standardized column names and prepared analytical features.
   - Created age groups and other useful features for analysis.

2. **SQL Business Analysis**

   Developed SQL queries to answer 10 business-focused questions:

   - Revenue comparison between male and female customers.
   - Identifying customers who used discounts but spent above average.
   - Top 5 products based on average customer ratings.
   - Comparing average purchase amounts across shipping methods.
   - Evaluating subscription impact on customer spending.
   - Products with the highest discount usage rates.
   - Customer classification into New, Returning, and Loyal segments.
   - Top 3 most purchased products within each category.
   - Subscription behavior among repeat buyers.
   - Revenue contribution across different age groups.

3. **Interactive Streamlit Dashboard**

   RetailPulse provides six analytical modules:

   | Module | Key Focus |
   |---|---|
   | Executive Overview | Revenue, purchase trends, and key KPIs |
   | Customer Intelligence | Age, gender, location, and customer behavior |
   | Product & Sales | Product performance and purchasing patterns |
   | Marketing Intelligence | Discounts, promotions, and subscriptions |
   | Customer Segmentation | Behavioral segmentation using K-Means |
   | Data Quality & SQL | Data quality checks and SQL analysis |

4. **Customer Segmentation Using Machine Learning**

   - Applied K-Means clustering to identify behavioral customer groups.
   - Used numerical and categorical customer attributes.
   - Applied StandardScaler and OneHotEncoder for preprocessing.
   - Compared segment sizes, average spending, and purchase behavior.
   - Visualized customer segments and enabled segment data export.

## 📂 Project Structure

```text
RetailPulse/
│
├── app.py
├── customer_shopping_behavior.csv
├── customer_shopping_behavior.ipynb
├── retail_analysis.sql
├── requirements.txt
└── README.md
```

*File names may vary depending on your repository.*

## 🚀 Run the Project Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/RetailPulse.git
cd RetailPulse
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Launch the Streamlit dashboard**

```bash
streamlit run app.py
```

**4. Upload the dataset**

Upload `customer_shopping_behavior.csv` through the sidebar to explore the dashboard.

## 💡 Key Learning Outcomes

- Translating business questions into SQL queries.
- Performing exploratory data analysis and data cleaning.
- Building interactive dashboards for business stakeholders.
- Analyzing customer purchasing and marketing behavior.
- Applying machine learning for customer segmentation.
- Communicating analytical findings through visualizations.

## 🎯 Project Objective

RetailPulse demonstrates how data analytics and machine learning can be combined to understand customer behavior, evaluate business performance, and support better retail decision-making.

---

**Built with Python, SQL, Machine Learning, and Streamlit.**
