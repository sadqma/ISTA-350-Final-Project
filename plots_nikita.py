import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

OUTPUT_DIR = "nikita_plots_img"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created folder: {OUTPUT_DIR}")


df_raw = pd.read_csv("data/pisa_scores.csv", header=None)


# last clean table has EXACT 3 columns per row:
clean_rows = df_raw[df_raw.apply(lambda row: row.count(), axis=1) == 3]

# Reset and rename
df = clean_rows.reset_index(drop=True)
df.columns = ["Rank", "Country", "Score"]

# Convert numeric columns
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

# Drop NaN
df = df.dropna(subset=["Rank", "Score", "Country"])

#1

X = df[["Rank"]].values
y = df[["Score"]].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label="Countries")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression")

plt.title("PISA: Rank vs Score")
plt.xlabel("Rank (lower = better)")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pisa_scatter_regression.png")
plt.close()

#2

top10 = df.sort_values("Score", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top10, x="Score", y="Country", palette="magma")
plt.title("Top 10 Countries by PISA Score")
plt.xlabel("Score")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pisa_top10.png")
plt.close()

#3
plt.figure(figsize=(10, 6))
sns.histplot(df["Score"], bins=20, kde=True, color="blue")
plt.title("Distribution of PISA Scores")
plt.xlabel("Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pisa_score_distribution.png")
plt.close()


print("✔ All Nikita's PISA graphs created successfully in:", OUTPUT_DIR)
