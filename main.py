"""
Nobel Prize Data Quality Check
-----------------------------
This script loads `nobel_prize_data.csv`, then:
1. Reports duplicate rows.
2. Reports overall and per‑column NaN counts & percentages.
3. Prints which columns contain NaNs.
4. Generates a heatmap of missing‑value patterns.
5. Converts birth_date to datetime.
6. Converts prize_share to percentage.
7. Plots gender distribution with Plotly Donut Chart.
8. Finds first 3 female laureates.
9. Identifies repeat winners.
10. Visualizes prizes per category.
11. Identifies first Economics laureate.

Run:
    python nobel_data_quality_check.py
Requires:
    pandas, numpy, seaborn, matplotlib, plotly
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1) Load dataset
# --------------------------------------------------
CSV_PATH = Path("nobel_prize_data.csv")
if not CSV_PATH.exists():
    raise FileNotFoundError("CSV file not found. Check the path.")

df = pd.read_csv(CSV_PATH)
print("==================== DATA OVERVIEW ====================")
print(f"Rows, columns : {df.shape}\n")

# --------------------------------------------------
# 2) Duplicate row check
# --------------------------------------------------
dupe_count = df.duplicated().sum()
print("--- DUPLICATE ROWS ---")
print(f"Any duplicates? {'Yes' if dupe_count else 'No'}")
print(f"Total duplicate rows: {dupe_count}\n")

# --------------------------------------------------
# 3) NaN overview
# --------------------------------------------------
any_nans = df.isna().any().any()
print("--- NaN OVERVIEW ---")
print(f"Any NaNs in DataFrame? {'Yes' if any_nans else 'No'}")
print(f"Total NaNs in DataFrame: {df.isna().sum().sum()}\n")

nan_count = df.isna().sum()
nan_pct = (df.isna().mean() * 100).round(1)
nan_summary = pd.DataFrame({"NaN_Count": nan_count, "NaN_%": nan_pct}).sort_values("NaN_Count", ascending=False)
print("--- NaN per column (only columns with NaNs) ---")
print(nan_summary[nan_summary["NaN_Count"] > 0])
print()

# --------------------------------------------------
# 4) Missing‑value heatmap
# --------------------------------------------------
plt.figure(figsize=(14, 6))
sns.heatmap(df.isna(), cbar=False, yticklabels=False)
plt.title("Missing‑Value Heatmap • Nobel Prize Dataset")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5) Why NaNs
# --------------------------------------------------
print("--- WHY CERTAIN COLUMNS HAVE NaNs ---")
explanations = {
    "organization_*": "Individuals vs. organization laureates.",
    "birth_*": "Columns empty for institutions or historical data gaps.",
    "death_*": "Living laureates naturally have no death data.",
    "motivation": "Some early awards lack citation text.",
    "prize_share": "Sometimes not recorded for team/organisation awards."
}
for pattern, reason in explanations.items():
    print(f"Columns like '{pattern}' → {reason}")

# --------------------------------------------------
# 6) Convert birth_date to datetime
# --------------------------------------------------
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
print(df["birth_date"].head())
print(df["birth_date"].dtype)

# --------------------------------------------------
# 7) Convert prize_share to percentage
# --------------------------------------------------
df["share_percent"] = df["prize_share"].apply(lambda x: eval(x) * 100 if pd.notna(x) else np.nan)
print(df[["prize_share", "share_percent"]].head())

# --------------------------------------------------
# 8) Gender Distribution Donut Chart
# --------------------------------------------------
gender_counts = df["sex"].value_counts().reset_index()
gender_counts.columns = ["sex", "count"]
fig = px.pie(gender_counts, names="sex", values="count", hole=0.5, title="Percentage of Male vs. Female Laureates")
fig.update_traces(textinfo='percent+label')
fig.show()

female_pct = (df["sex"] == "Female").mean() * 100
print(f"\nPercentage of Nobel Prizes awarded to women: {female_pct:.2f}%")

# --------------------------------------------------
# 9) First 3 Female Laureates
# --------------------------------------------------
female_laureates = df[df["sex"] == "Female"]
female_laureates_sorted = female_laureates.sort_values(by=["year", "category"])
first_3_women = female_laureates_sorted.head(3)
print("\n--- FIRST 3 FEMALE NOBEL LAUREATES ---")
for i, row in first_3_women.iterrows():
    print(f"Name         : {row['full_name']}")
    print(f"Year         : {row['year']}")
    print(f"Category     : {row['category']}")
    print(f"Motivation   : {row['motivation']}")
    print(f"Birth Country: {row['birth_country']}")
    print(f"Organization : {row['organization_name']}\n")

# --------------------------------------------------
# 10) Repeat Winners
# --------------------------------------------------
winner_counts = df["full_name"].value_counts()
repeat_winners = winner_counts[winner_counts > 1]
print("\n--- REPEAT NOBEL PRIZE WINNERS ---")
for name in repeat_winners.index:
    entries = df[df["full_name"] == name]
    print(f"\n{name} — {len(entries)} prizes")
    for _, row in entries.iterrows():
        print(f"• {row['year']} - {row['category']} - {row['motivation']}")

# --------------------------------------------------
# 11) Prizes per Category (Bar Chart)
# --------------------------------------------------
category_counts = df["category"].value_counts().reset_index()
category_counts.columns = ["category", "count"]
fig = px.bar(
    category_counts,
    x="category",
    y="count",
    title="Number of Nobel Prizes Awarded by Category",
    color="count",
    color_continuous_scale="Aggrnyl",
)
fig.update_layout(coloraxis_showscale=False)
fig.show()

most_prizes = category_counts.iloc[0]
least_prizes = category_counts.iloc[-1]
print("\n--- CATEGORY SUMMARY ---")
print(f"Most awarded category   : {most_prizes['category']} ({most_prizes['count']})")
print(f"Least awarded category  : {least_prizes['category']} ({least_prizes['count']})")
print(f"Total categories        : {category_counts.shape[0]}")

# --------------------------------------------------
# 12) First Economics Nobel Prize
# --------------------------------------------------
df["category_normalized"] = df["category"].str.strip().str.lower()
econ_df = df[df["category_normalized"] == "economics"]
econ_df_sorted = econ_df.sort_values(by="year")

if not econ_df_sorted.empty:
    first_econ_prize = econ_df_sorted.iloc[0]
    print("\n--- FIRST ECONOMICS NOBEL PRIZE ---")
    print(f"Year         : {first_econ_prize['year']}")
    print(f"Laureate     : {first_econ_prize['full_name']}")
    print(f"Motivation   : {first_econ_prize['motivation']}")
    print(f"Birth Country: {first_econ_prize['birth_country']}")
else:
    print("\nNo Economics Nobel Prize data found.")

# --------------------------------------------------
# 11) Male and Female Winners by Category
# --------------------------------------------------

# Group by category and sex, count number of prizes
gender_category_counts = df.groupby(["category", "sex"]).size().reset_index(name="count")

# Create a Plotly grouped bar chart
fig = px.bar(
    gender_category_counts,
    x="category",
    y="count",
    color="sex",
    title="Number of Male and Female Nobel Laureates by Category",
    barmode="group",
    labels={"count": "Number of Laureates"},
    color_discrete_map={"Male": "steelblue", "Female": "hotpink"}
)

# Show chart
fig.show()

# Optional: Print Literature vs. Physics female counts
female_by_category = gender_category_counts[
    (gender_category_counts["sex"] == "Female")
].set_index("category")

print("\n--- FEMALE LAUREATES BY CATEGORY ---")
for cat in ["literature", "physics"]:
    if cat in female_by_category.index:
        count = female_by_category.loc[cat, "count"]
        print(f"{cat.capitalize()} : {count} female winners")
    else:
        print(f"{cat.capitalize()} : 0 female winners")

# --------------------------------------------------
# 12) Number of Prizes Awarded Over Time
# --------------------------------------------------

# Count number of prizes awarded per year
prizes_per_year = df["year"].value_counts().sort_index()

# Convert to DataFrame
prize_trend_df = prizes_per_year.reset_index()
prize_trend_df.columns = ["year", "num_prizes"]
prize_trend_df = prize_trend_df.sort_values("year")

# Compute a 5-year rolling average
prize_trend_df["rolling_avg"] = prize_trend_df["num_prizes"].rolling(window=5).mean()

# Plot using Matplotlib
plt.figure(figsize=(12, 6))
plt.scatter(prize_trend_df["year"], prize_trend_df["num_prizes"], alpha=0.6, label="Annual Prize Count")
plt.plot(prize_trend_df["year"], prize_trend_df["rolling_avg"], color="crimson", linewidth=2.5, label="5-Year Rolling Average")

# Set x-ticks every 5 years from 1900 to 2020

xticks = np.arange(1900, 2025, 5)
plt.xticks(xticks, rotation=45)

# Titles and labels
plt.title("Nobel Prizes Awarded Over Time", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Number of Prizes Awarded")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Convert birth_date and filter relevant columns
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

# Convert 'prize_share' to float percentage
df["share_percent"] = df["prize_share"].apply(lambda x: eval(x) * 100 if pd.notna(x) else np.nan)

# --------------------------------------------------
# 1) Count of prizes awarded per year
# --------------------------------------------------
prizes_per_year = df.groupby("year")["prize"].count()

# 2) Average share percentage per year
avg_share_per_year = df.groupby("year")["share_percent"].mean()

# 3) Apply 5-year rolling average
prizes_rolling = prizes_per_year.rolling(window=5).mean()
share_rolling = avg_share_per_year.rolling(window=5).mean()

# --------------------------------------------------
# 4) Plot using Matplotlib with dual Y-axes
# --------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Scatter plot: number of prizes per year
ax1.scatter(prizes_per_year.index, prizes_per_year.values, color="gray", alpha=0.5, label="Prizes per Year")
ax1.plot(prizes_rolling.index, prizes_rolling.values, color="blue", label="5-Year Rolling Avg (Prizes)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Prizes", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xticks(np.arange(1900, 2025, 5))  # every 5 years
ax1.set_title("Nobel Prizes Over Time & Average Share Percentage")

# Secondary Y-axis for average prize share
ax2 = ax1.twinx()
ax2.plot(share_rolling.index, share_rolling.values, color="red", label="5-Year Rolling Avg (Share %)")
ax2.set_ylabel("Average Prize Share (%)", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.invert_yaxis()  # Optional: invert for clarity (higher share % = fewer shared)

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")

plt.tight_layout()
plt.show()

# Drop NaN countries
df = df[df["birth_country_current"].notna()]

# Group by birth_country_current and count prizes
country_counts = (
    df.groupby("birth_country_current")["prize"]
    .count()
    .reset_index()
    .rename(columns={"birth_country_current": "country", "prize": "prizes"})
    .sort_values(by="prizes", ascending=False)  # descending order
)

# Plot with better readability
fig = px.bar(
    country_counts,
    x="prizes",
    y="country",
    orientation="h",
    title="Nobel Prizes by Country (Based on Birth Country - Modern Borders)",
    labels={"prizes": "Number of Prizes", "country": "Country"},
    color="prizes",
    color_continuous_scale="Viridis",
    height=1200  # increase height for better spacing
)

# Update layout for larger font sizes
fig.update_layout(
    yaxis=dict(tickfont=dict(size=12)),
    xaxis=dict(tickfont=dict(size=12)),
    title=dict(font=dict(size=18)),
    margin=dict(l=150, r=20, t=60, b=40)  # more left margin for long country names
)

fig.show()

# Remove rows without country info
df = df[df["birth_country_current"].notna()]

# Group by current birth country and count prizes
country_counts = (
    df.groupby("birth_country_current")["prize"]
    .count()
    .reset_index()
    .rename(columns={"birth_country_current": "country", "prize": "prizes"})
)

# Plotly Choropleth Map
fig = px.choropleth(
    country_counts,
    locations="country",
    locationmode="country names",  # Match country names
    color="prizes",
    hover_name="country",
    color_continuous_scale="Plasma",
    title="Number of Nobel Prizes by Country",
    projection="natural earth"
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    title_x=0.5
)

fig.show()

# Group and count the number of prizes per country and category
country_category_counts = (
    df
    .groupby(["birth_country", "category"])
    .size()
    .reset_index(name="prize_count")
    .rename(columns={"birth_country": "country"})
)

# Filter for top 20 countries by total prizes
top_countries = (
    df["birth_country"]
    .value_counts()
    .head(20)
    .index
)

# Filter dataset to include only those top 20 countries
country_category_counts = country_category_counts[country_category_counts["country"].isin(top_countries)]

# Create grouped or stacked bar chart
fig = px.bar(
    country_category_counts,
    x="country",
    y="prize_count",
    color="category",
    title="Nobel Prizes by Country and Category",
    labels={"prize_count": "Number of Prizes"},
    text="prize_count"
)

fig.update_layout(
    xaxis_title="Country",
    yaxis_title="Number of Prizes",
    barmode="stack",  # Change to "group" for side-by-side bars
    xaxis_tickangle=45
)

fig.show()

# Convert birth_date to datetime
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

# Group by country and category
country_category = (
    df.groupby(["birth_country", "category"])
      .size()
      .reset_index(name="prize_count")
)

# Pivot to get countries as rows and categories as columns
pivot_table = country_category.pivot_table(
    index="birth_country",
    columns="category",
    values="prize_count",
    fill_value=0
)

# Select relevant countries for analysis
selected_countries = [
    "United States of America", "Germany", "Japan",
    "France", "United Kingdom", "Australia", "Netherlands"
]

# Extract relevant rows
pivot_selected = pivot_table.loc[selected_countries]

# Store each country for comparison
us = pivot_selected.loc["United States of America"]
germany = pivot_selected.loc["Germany"]
japan = pivot_selected.loc["Japan"]
uk = pivot_selected.loc["United Kingdom"]
france = pivot_selected.loc["France"]
australia = pivot_selected.loc["Australia"]
netherlands = pivot_selected.loc["Netherlands"]

# 1. Germany and Japan weakest compared to USA
weakest_germany = (us - germany).sort_values(ascending=False)
weakest_japan = (us - japan).sort_values(ascending=False)

print("\nGermany weakest compared to USA:")
print(weakest_germany.head(3))

print("\nJapan weakest compared to USA:")
print(weakest_japan.head(3))

# 2. Where Germany > UK
germany_vs_uk = (germany - uk).sort_values(ascending=False)
print("\nGermany wins over UK in:")
print(germany_vs_uk[germany_vs_uk > 0])

# 3. France > Germany
france_vs_germany = (france - germany).sort_values(ascending=False)
print("\nFrance wins over Germany in:")
print(france_vs_germany[france_vs_germany > 0])

# 4. Australia's top category
australia_top_category = australia.sort_values(ascending=False).head(1)
print("\nAustralia's top Nobel category:")
print(australia_top_category)

# 5. Netherlands: category making up ≈ 50%
total_netherlands = netherlands.sum()
netherlands_pct = (netherlands / total_netherlands * 100).sort_values(ascending=False)
print("\nNetherlands category percentage:")
print(netherlands_pct)

print("\nCategory close to 50%:")
print(netherlands_pct[netherlands_pct >= 45])

# 6. USA vs France (Economics, Physics, Medicine)
print("\nUSA vs France in select categories:")
print(f"Economics: USA={us.get('economics', 0)}, France={france.get('economics', 0)}")
print(f"Physics: USA={us.get('physics', 0)}, France={france.get('physics', 0)}")
print(f"Medicine: USA={us.get('medicine', 0)}, France={france.get('medicine', 0)}")

# --------------------------------------------------
# Cumulative Prizes Won by Country Over Time
# --------------------------------------------------

# 1. Filter out missing birth_country_current or year
df_valid = df[df["birth_country_current"].notna() & df["year"].notna()]

# 2. Group by year and country and count prizes
prizes_per_year_country = df_valid.groupby(["year", "birth_country_current"]).size().reset_index(name="prize_count")

# 3. Pivot: rows = year, columns = country, values = prize counts
pivot_table = prizes_per_year_country.pivot(index="year", columns="birth_country_current", values="prize_count").fillna(0)

# 4. Compute cumulative sum across years
cumulative_prizes = pivot_table.cumsum()

# 5. Create interactive Plotly line chart
fig = px.line(
    cumulative_prizes,
    labels={"value": "Cumulative Prizes", "year": "Year", "birth_country_current": "Country"},
    title="Cumulative Nobel Prizes Won by Country Over Time"
)

fig.update_layout(
    xaxis=dict(dtick=5),  # tick every 5 years
    legend=dict(title="Country", traceorder="normal"),
    height=600
)

fig.show()

# --------------------------------------------------
# Top Research Organizations by Number of Laureates
# --------------------------------------------------

# 1. Filter out missing organization names
org_counts = df["organization_name"].dropna()

# 2. Count the frequency of each organization
top_orgs = org_counts.value_counts().head(20).reset_index()
top_orgs.columns = ["organization", "prize_count"]

# 3. Plot with Plotly
import plotly.express as px

fig = px.bar(
    top_orgs.sort_values("prize_count", ascending=True),  # horizontal chart
    x="prize_count",
    y="organization",
    orientation="h",
    title="Top Research Organizations Affiliated with Nobel Laureates",
    color="prize_count",
    color_continuous_scale="Blues"
)

fig.update_layout(
    xaxis_title="Number of Prizes",
    yaxis_title="Organization",
    coloraxis_showscale=False,
    height=600
)

fig.show()

# --------------------------------------------------
# Cities with Most Nobel-Winning Institutions
# --------------------------------------------------

# 1. Drop missing cities
city_counts = df["organization_city"].dropna()

# 2. Count city frequency and get top 20
top_cities = city_counts.value_counts().head(20).reset_index()
top_cities.columns = ["city", "prize_count"]

# 3. Plot using Plotly
fig = px.bar(
    top_cities.sort_values("prize_count", ascending=True),  # horizontal
    x="prize_count",
    y="city",
    orientation="h",
    title="Top 20 Cities of Nobel-Winning Research Institutions",
    color="prize_count",
    color_continuous_scale="YlGnBu"
)

fig.update_layout(
    xaxis_title="Number of Nobel Prizes",
    yaxis_title="City",
    coloraxis_showscale=False,
    height=600
)

fig.show()

# --------------------------------------------------
# Top Birth Cities of Nobel Laureates
# --------------------------------------------------

# 1. Drop missing birth cities
birth_city_counts = df["birth_city"].dropna()

# 2. Count frequency and get top 20
top_birth_cities = birth_city_counts.value_counts().head(20).reset_index()
top_birth_cities.columns = ["city", "laureate_count"]

# 3. Plot using Plotly with Plasma color scale
fig = px.bar(
    top_birth_cities.sort_values("laureate_count", ascending=True),
    x="laureate_count",
    y="city",
    orientation="h",
    title="Top 20 Birth Cities of Nobel Laureates",
    color="laureate_count",
    color_continuous_scale="Plasma"
)
fig.update_layout(
    xaxis_title="Number of Laureates",
    yaxis_title="Birth City",
    coloraxis_showscale=False,
    height=600
)
fig.show()

# --------------------------------------------------
# Sunburst Chart: Country → City → Organisation
# --------------------------------------------------

# Filter only rows with non-null organizations and countries
org_data = df.dropna(subset=["organization_name", "organization_country", "organization_city"])

# Group by Country > City > Organization
sunburst_data = (
    org_data.groupby(["organization_country", "organization_city", "organization_name"])
    .size()
    .reset_index(name="prize_count")
)

# Create sunburst chart
fig = px.sunburst(
    sunburst_data,
    path=["organization_country", "organization_city", "organization_name"],
    values="prize_count",
    title="Nobel Prizes: Country → City → Organization",
    color="prize_count",
    color_continuous_scale="Blues"
)

fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
fig.show()

# Convert birth_date to datetime (already done earlier, can be skipped or kept)
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

# Calculate winning age
df["winning_age"] = df["year"] - df["birth_date"].dt.year

# Preview result
print(df[["full_name", "birth_date", "year", "winning_age"]].head())

# Youngest winner
youngest_winner = df.loc[df["winning_age"].idxmin()]
print("\n----- YOUNGEST WINNER -----")
print(f"Name       : {youngest_winner['full_name']}")
print(f"Age        : {youngest_winner['winning_age']} years")
print(f"Year       : {youngest_winner['year']}")
print(f"Category   : {youngest_winner['category']}")
print(f"Motivation : {youngest_winner['motivation']}")

# Oldest winner
oldest_winner = df.loc[df["winning_age"].idxmax()]
print("\n----- OLDEST WINNER -----")
print(f"Name       : {oldest_winner['full_name']}")
print(f"Age        : {oldest_winner['winning_age']} years")
print(f"Year       : {oldest_winner['year']}")
print(f"Category   : {oldest_winner['category']}")
print(f"Motivation : {oldest_winner['motivation']}")

# Statistics
avg_age = df["winning_age"].mean()
age_75_percentile = df["winning_age"].quantile(0.75)

print(f"\nAverage winning age: {avg_age:.2f} years")
print(f"75% of laureates are younger than {age_75_percentile:.2f} years")

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df["winning_age"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Nobel Laureate Winning Ages")
plt.xlabel("Age at Time of Award")
plt.ylabel("Number of Laureates")
plt.axvline(avg_age, color='red', linestyle='--', label=f"Mean Age ≈ {avg_age:.0f}")
plt.axvline(age_75_percentile, color='green', linestyle='--', label=f"75th Percentile ≈ {age_75_percentile:.0f}")
plt.legend()
plt.tight_layout()
plt.show()

# Ensure birth_date is in datetime format and winning_age is calculated
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
df["winning_age"] = df["year"] - df["birth_date"].dt.year

# Drop rows with missing age
age_data = df["winning_age"].dropna()

# 1. Descriptive Statistics
print("=== Descriptive Statistics for Laureate Age at Time of Award ===")
print(age_data.describe())

# 2. Histograms with varying bin sizes
bin_sizes = [10, 20, 30, 50]

for bins in bin_sizes:
    plt.figure(figsize=(10, 5))
    sns.histplot(age_data, bins=bins, kde=True, color="skyblue")
    plt.title(f"Distribution of Laureate Age at Time of Award (Bins = {bins})")
    plt.xlabel("Age")
    plt.ylabel("Number of Laureates")
    plt.axvline(age_data.mean(), color='red', linestyle='--', label=f"Mean ≈ {age_data.mean():.1f}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Ensure datetime and age column are correct
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
df["winning_age"] = df["year"] - df["birth_date"].dt.year

# Drop missing values
age_trend_data = df[["year", "winning_age"]].dropna()

# Create a regplot with lowess smoothing
plt.figure(figsize=(12, 6))
sns.regplot(
    data=age_trend_data,
    x="year",
    y="winning_age",
    lowess=True,
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)
plt.title("Age at Time of Nobel Prize Over Time (Lowess Trend)")
plt.xlabel("Award Year")
plt.ylabel("Age at Award")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Recalculate age if needed
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
df["winning_age"] = df["year"] - df["birth_date"].dt.year

# Drop rows with missing data
age_by_category = df[["category", "winning_age"]].dropna()

# Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=age_by_category, x="category", y="winning_age", palette="Set2")

plt.title("Nobel Prize Winner Age Distribution by Category")
plt.ylabel("Age at Time of Award")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Ensure correct datetime and age columns
df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
df["winning_age"] = df["year"] - df["birth_date"].dt.year

# Drop rows with missing values
age_df = df[["year", "winning_age", "category"]].dropna()

# Use Seaborn's lmplot with one row per category
sns.lmplot(
    data=age_df,
    x="year",
    y="winning_age",
    row="category",
    lowess=True,
    height=3,
    aspect=2,
    scatter_kws={"s": 20, "alpha": 0.3},
    line_kws={"color": "red"}
)

plt.subplots_adjust(top=0.95)
plt.suptitle("Nobel Laureate Winning Age Trends by Category", fontsize=16)
plt.show()
