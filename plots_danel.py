import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

OUTPUT_DIR = "danel_plots_img"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created folder: {OUTPUT_DIR}")


# Load raw
df_raw = pd.read_csv("data/happiness_report.csv")

df = df_raw.iloc[2:].reset_index(drop=True)

df.columns = [
    "Rank", "Country", "Score", "Change",
    "GDP", "Social", "Life_expectancy",
    "Freedom", "Generosity", "Trust"
]

# Keep only useful columns
df = df[["Rank", "Country", "Score", "Change"]]

# Convert numeric
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df["Change"] = pd.to_numeric(df["Change"], errors="coerce")

df = df.dropna(subset=["Score"])

#1

X = df[["Change"]].values
y = df["Score"].values.reshape(-1, 1)

mask = ~np.isnan(X.flatten()) & ~np.isnan(y.flatten())
X = X[mask].reshape(-1, 1)
y = y[mask].reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6)
plt.plot(X, y_pred, color="red", linewidth=2)
plt.title("Happiness Score vs Yearly Change")
plt.xlabel("Change Over Prior Year")
plt.ylabel("Happiness Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/happiness_scatter_regression.png")
plt.close()

#2

top10 = df.sort_values("Score", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top10, x="Score", y="Country", palette="viridis")
plt.title("Top 10 Happiest Countries")
plt.xlabel("Happiness Score")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/happiness_top10.png")
plt.close()

#3

plt.figure(figsize=(10, 6))
sns.histplot(df["Score"], bins=20, kde=True, color="purple")
plt.title("Distribution of Happiness Scores")
plt.xlabel("Happiness Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/happiness_distribution.png")
plt.close()


print("✔ All graphs saved to folder:", OUTPUT_DIR)
