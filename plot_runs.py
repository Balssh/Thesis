import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_cols(df):
    for index, col in enumerate(columns):
        grids.append(sns.FacetGrid(data=df, col="tag", hue="tag", col_wrap=5))

        grids[index].map(sns.lineplot, "step", col)

    plt.show()


df = pd.read_csv("ExcelRuns/runs_smoothed.csv")
print(df.head)

columns = [col for col in df.columns if col not in ["step", "tag"]]
grids = []

# norm_df = df[
#     ~df["tag"].isin(
#         [
#             "SeparateNN_GOOD_NO_NORM_ADV",
#             "SeparateNN_NO_ADV_NORM",
#             "SeparateNN_GOOD_PLAIN",
#             "SeparateNN_PLAIN",
#             "SeparateNN_PLAIN_ANNEAL",
#         ]
#     )
# ]

no_gae_df = df[
    ~df["tag"].isin(
        [
            "SeparateNN_NO_GAE",
            "SeparateNN_GOOD_NO_GAE",
            "SeparateNN_GOOD_PLAIN",
            "SeparateNN_PLAIN",
            "SeparateNN_PLAIN_ANNEAL",
        ]
    )
]
plot_cols(no_gae_df)
