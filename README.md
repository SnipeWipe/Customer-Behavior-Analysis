# 🛍️ Customer Shopping Behavior Analysis

An end-to-end **Customer Shopping Behavior Analysis** project using **Python, Pandas, and PostgreSQL**. The project focuses on cleaning and transforming customer shopping data, performing exploratory data analysis, engineering useful business features, and loading the processed dataset into PostgreSQL for further analytics and visualization.

---

## 📌 Project Overview

Understanding customer shopping behavior is important for businesses to improve customer engagement, marketing strategies, product offerings, and sales performance.

This project analyzes customer purchasing information such as:

* Customer demographics
* Product categories and items
* Purchase amounts
* Customer reviews
* Subscription status
* Discounts
* Shipping preferences
* Payment methods
* Purchase frequency
* Previous purchases

The raw dataset is cleaned and transformed using **Pandas**, followed by feature engineering and loading the final dataset into **PostgreSQL**.

---

## 🗂️ Dataset

The project uses the following dataset:

**`customer_shopping_behavior.csv`**

### Dataset Size

* **Rows:** 3,900
* **Original Columns:** 18
* **Final Columns:** 19
* **Missing Values:** 37 in `Review Rating`

### Original Columns

| Column                 | Description                             |
| ---------------------- | --------------------------------------- |
| Customer ID            | Unique customer identifier              |
| Age                    | Customer age                            |
| Gender                 | Customer gender                         |
| Item Purchased         | Product purchased                       |
| Category               | Product category                        |
| Purchase Amount (USD)  | Purchase amount in USD                  |
| Location               | Customer location                       |
| Size                   | Product size                            |
| Color                  | Product color                           |
| Season                 | Purchase season                         |
| Review Rating          | Customer review rating                  |
| Subscription Status    | Whether the customer has a subscription |
| Shipping Type          | Shipping method selected                |
| Discount Applied       | Whether a discount was applied          |
| Promo Code Used        | Whether a promotional code was used     |
| Previous Purchases     | Number of previous purchases            |
| Payment Method         | Payment method                          |
| Frequency of Purchases | Customer purchase frequency             |

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **SQLAlchemy**
* **PostgreSQL**
* **psycopg2**
* **pgAdmin 4**
* **Jupyter Notebook**

---

## 🔄 Data Cleaning & Transformation

### 1. Load the Dataset

The dataset is loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv("customer_shopping_behavior.csv")
```

---

### 2. Exploratory Data Inspection

Initial inspection was performed using:

```python
df.head()
df.info()
df.describe(include='all')
df.isnull().sum()
```

This helped identify:

* Dataset dimensions
* Data types
* Unique categorical values
* Statistical distributions
* Missing values

---

### 3. Handling Missing Values

The dataset contained **37 missing values** in the `Review Rating` column.

Instead of removing these records, missing ratings were filled using the **median review rating within each product category**.

```python
df['Review Rating'] = df.groupby('Category')['Review Rating'].transform(
    lambda x: x.fillna(x.median())
)
```

This approach preserves the existing observations while using category-specific information for imputation.

---

### 4. Standardizing Column Names

Column names were converted to lowercase and spaces were replaced with underscores.

```python
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')

df = df.rename(
    columns={'purchase_amount_(usd)': 'purchase_amount'}
)
```

For example:

```text
Purchase Amount (USD)
```

was transformed into:

```text
purchase_amount
```

This makes the columns easier to use in Python and SQL.

---

## 🧮 Feature Engineering

### 1. Age Group

Customers were divided into four age groups using `pd.qcut()`:

```python
labels = ['Young Adult', 'Adult', 'Middle-Aged', 'Senior']

df['age_group'] = pd.qcut(
    df['age'],
    q=4,
    labels=labels
)
```

The resulting groups are:

* Young Adult
* Adult
* Middle-Aged
* Senior

Using quartiles ensures that the groups are based on the distribution of customers in the dataset.

---

### 2. Purchase Frequency in Days

The categorical purchase frequency was converted into an approximate number of days.

```python
frequency_mapping = {
    'Fortnightly': 14,
    'Weekly': 7,
    'Monthly': 30,
    'Quarterly': 90,
    'Bi-weekly': 14,
    'Annually': 365,
    'Every 3 Months': 90
}

df['purchase_frequency_days'] = (
    df['frequency_of_purchases'].map(frequency_mapping)
)
```

This creates a numerical feature:

```text
purchase_frequency_days
```

which can be used for quantitative analysis and customer segmentation.

---

## 🔍 Redundant Column Detection

The `discount_applied` and `promo_code_used` columns were compared:

```python
(df['discount_applied'] == df['promo_code_used']).all()
```

The result was:

```text
True
```

This indicates that both columns contain identical information for every record.

Therefore, the redundant `promo_code_used` column was removed:

```python
df = df.drop('promo_code_used', axis=1)
```

---

## 🗄️ PostgreSQL Integration

The cleaned dataset was loaded into a PostgreSQL database using **SQLAlchemy** and **psycopg2**.

### Required Libraries

```bash
pip install psycopg2-binary sqlalchemy
```

### Database Configuration

```python
from sqlalchemy import create_engine

username = "postgres"
password = "YOUR_PASSWORD"
host = "localhost"
port = "5432"
database = "customer_behavior"

engine = create_engine(
    f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
)
```

> **Security Note:** Do not commit your PostgreSQL password to GitHub. Use environment variables or a `.env` file for credentials.

---

### Load Data into PostgreSQL

```python
table_name = "customer"

df.to_sql(
    table_name,
    engine,
    if_exists="replace",
    index=False
)

print(
    f"Data successfully loaded into table '{table_name}' "
    f"in database '{database}'."
)
```

The processed DataFrame is stored in PostgreSQL as:

```text
Database: customer_behavior
Table: customer
```

---

## 📊 Final Dataset Structure

After cleaning and feature engineering, the dataset contains **19 columns**:

```text
customer_id
age
gender
item_purchased
category
purchase_amount
location
size
color
season
review_rating
subscription_status
shipping_type
discount_applied
previous_purchases
payment_method
frequency_of_purchases
age_group
purchase_frequency_days
```

---

## 📈 Potential Business Analysis

The processed dataset can be used to answer questions such as:

### Customer Analysis

* Which age groups purchase the most?
* Which locations have the highest number of customers?
* How does purchasing behavior vary by gender?
* Which customers have the highest number of previous purchases?

### Product Analysis

* Which products are purchased most frequently?
* Which categories generate the highest sales?
* Which sizes and colors are most popular?
* How do product ratings vary across categories?

### Sales Analysis

* What is the average purchase amount?
* Which customer segments spend the most?
* How does purchase frequency relate to spending?
* Which seasons generate higher purchase activity?

### Marketing Analysis

* How does discount usage relate to purchase behavior?
* What is the relationship between subscription status and purchasing frequency?
* Which customer segments could be targeted for retention campaigns?

### Operational Analysis

* Which shipping methods are most frequently selected?
* Which payment methods are most popular?
* Does shipping preference vary by customer segment?

---

## 📁 Suggested Project Structure

```text
customer-shopping-behavior/
│
├── customer_shopping_behavior.csv
├── customer_shopping_behavior.ipynb
├── README.md
│
└── requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
sqlalchemy
psycopg2-binary
```

---

## 🚀 Project Workflow

```text
Raw CSV Dataset
       │
       ▼
Data Loading
       │
       ▼
Data Inspection
       │
       ▼
Missing Value Treatment
       │
       ▼
Column Name Standardization
       │
       ▼
Feature Engineering
       │
       ├── Age Group
       │
       └── Purchase Frequency Days
       │
       ▼
Redundant Column Removal
       │
       ▼
Cleaned Dataset
       │
       ▼
PostgreSQL
       │
       ▼
SQL / BI Analysis
```

---

## 🎯 Key Skills Demonstrated

This project demonstrates practical experience with:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Missing-value treatment
* Categorical data handling
* Feature engineering
* Data quality checks
* Redundancy detection
* Pandas
* Python
* SQL
* PostgreSQL
* SQLAlchemy
* Database integration
* Preparing datasets for BI and analytics

---

## 🔮 Future Improvements

Potential extensions include:

* Building an interactive **Power BI dashboard**
* Performing customer segmentation using **RFM analysis**
* Creating customer lifetime value metrics
* Analyzing category-level revenue
* Building customer churn/retention indicators
* Creating SQL analytical queries
* Developing automated ETL pipelines
* Adding automated data-quality checks
* Deploying an interactive analytics application

---

## 👨‍💻 Author

**Akhilesh Pal**

Master's in Statistics | Data Analytics | Python | SQL | Power BI | Machine Learning

---

## ⭐ Project Objective

The primary objective of this project is to transform raw customer shopping data into a **clean, analysis-ready dataset** and integrate it with PostgreSQL so that it can be used for deeper SQL analysis, business intelligence, customer segmentation, and data-driven decision-making.
