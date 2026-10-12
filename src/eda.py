"""Exploratory charts saved to images/."""
import matplotlib.pyplot as plt


def run(df, out="images"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby("Contract")["Churn"].mean().sort_values().plot.barh(ax=ax[0], color="#4C72B0")
    ax[0].set_title("Churn rate by contract"); ax[0].set_xlabel("churn rate")
    for v, c in [(0, "#55A868"), (1, "#C44E52")]:
        df[df.Churn == v]["tenure"].plot.hist(ax=ax[1], bins=30, alpha=.6, color=c,
                                              label="Churned" if v else "Stayed")
    ax[1].set_title("Tenure (months)"); ax[1].legend()
    df.groupby("InternetService")["Churn"].mean().sort_values().plot.barh(ax=ax[2], color="#8172B2")
    ax[2].set_title("Churn rate by internet service")
    fig.tight_layout(); fig.savefig(f"{out}/eda_overview.png", dpi=120); plt.close(fig)
