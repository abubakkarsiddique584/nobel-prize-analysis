# 🏆 Nobel Prize Data Analysis

This repository contains a comprehensive exploratory data analysis (EDA) of the Nobel Prize dataset. The goal is to uncover patterns and trends in Nobel Prize awards over time, understand demographic and geographic distributions, and identify unique insights like the youngest/oldest winners, gender disparities, and category-specific trends.

---

## 📂 Dataset

The dataset used is `nobel_prize_data.csv`, which includes information about:
- Laureates' full names, birthplaces, and organizations
- Award year and category
- Gender
- Prize share
- Motivation
- Birth and death dates

---

## 📊 Key Analyses & Visualizations

- ✅ **Duplicate and Missing Data Checks**  
  Identified and removed duplicate rows, visualized missing values with a heatmap.

- 📈 **Prize Trends Over Time**  
  Yearly and cumulative award trends, average prize share analysis, and 5-year rolling averages.

- 📌 **Gender Distribution**  
  Pie chart of male vs. female laureates, with percentages by category.

- 🌍 **Geographic Breakdown**  
  Choropleth maps and bar charts showing country-wise prize distributions (modern borders and historical birth countries).

- 🎂 **Age at Time of Award**  
  Calculated winning age, identified youngest and oldest winners, and analyzed age trends over time and across categories.

- 📦 **Prize Category Insights**  
  Boxplots and regression plots show how winning age differs across Physics, Chemistry, Medicine, Literature, Peace, and Economics.

- 🏛 **Top Institutions and Cities**  
  Bar charts of the most awarded organizations and cities with Nobel affiliations.

- ☀️ **Sunburst Chart**  
  Hierarchical view: Country → City → Organization.

---

## 🧰 Tech Stack

- Python
- Pandas & NumPy
- Seaborn & Matplotlib
- Plotly (for interactive charts)

---

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/abubakkarsiddique584/nobel-prize-data-analysis.git
   cd nobel-prize-data-analysis
