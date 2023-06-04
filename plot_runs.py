import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("ExcelRuns/runs_smoothed.csv")
print(df.head)

# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# AFISEAZA COMPARATIV REZULTATELE PENTRU TOATE EXPERIMENTELE
# columns = [col for col in df.columns if col not in ["step", "tag"]]
# grids = []
# for index, col in enumerate(columns):
#     grids.append(sns.FacetGrid(data=df, col="tag", hue="tag", col_wrap=5))
#
#     grids[index].map(sns.lineplot, "step", col)
# fig = plt.figure()
# sns.relplot(
#     data=df,
#     kind="line",
#     x="step",
#     y="episodic_return",
#     hue="tag",
#     style="tag",
#     size="tag",
# )
plt.show()
