"""One-off helper: build notebooks/01_eda.ipynb programmatically.

Kept in the repo so the EDA notebook is reproducible. Run via:
    uv run python scripts/_build_eda_notebook.py
then execute with nbconvert (see prepare step in the session log).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# 01 — Telco Churn EDA\n\n"
        "Quick exploratory pass over the **reference window** "
        "(`data/reference/reference_data.parquet`, first 60% of the cleaned data).\n\n"
        "Goal: sanity-check nulls, class balance, feature types, and a couple of "
        "churn relationships before modelling."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style='whitegrid')\n"
        "\n"
        "df = pd.read_parquet('../data/reference/reference_data.parquet')\n"
        "print('shape:', df.shape)\n"
        "df.head()"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Structure, dtypes, nulls")
)
cells.append(
    nbf.v4.new_code_cell("df.info()")
)
cells.append(
    nbf.v4.new_code_cell("df.describe(include='all').T")
)

cells.append(
    nbf.v4.new_markdown_cell("## Class balance")
)
cells.append(
    nbf.v4.new_code_cell(
        "counts = df['Churn'].value_counts().sort_index()\n"
        "ax = counts.plot(kind='bar', color=['#4c72b0', '#dd8452'])\n"
        "ax.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)\n"
        "ax.set_title(f'Churn class balance (rate={df[\"Churn\"].mean():.3f})')\n"
        "ax.set_ylabel('count')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Correlation — numerical features")
)
cells.append(
    nbf.v4.new_code_cell(
        "num_cols = df.select_dtypes(include='number').columns.tolist()\n"
        "corr = df[num_cols].corr()\n"
        "plt.figure(figsize=(7, 6))\n"
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,\n"
        "            square=True, cbar_kws={'shrink': 0.8})\n"
        "plt.title('Numerical feature correlation')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Churn rate by contract type")
)
cells.append(
    nbf.v4.new_code_cell(
        "rate = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)\n"
        "ax = rate.plot(kind='bar', color='#c44e52')\n"
        "ax.set_title('Churn rate by contract type')\n"
        "ax.set_ylabel('churn rate')\n"
        "ax.set_xlabel('')\n"
        "plt.xticks(rotation=20, ha='right')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "rate"
    )
)

nb["cells"] = cells

out = Path("notebooks/01_eda.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
