import numpy as np
import pandas as pd
import seaborn as sns
import tensorboard as tb
from matplotlib import pyplot as plt
from packaging import version
from scipy import stats

experiment_ids = {
    "SharedNN": "84a77iJQTnGLxyiKnEQMsw",
    "SeparateNN_CONTROL_RUN": "7DgqMAzRRWOoBSX3gF8w0w",
    "SeparateNN_GOOD_095_GAMMA": "RQXzS3MBQHWLg6fe5hxMsQ",
    "SeparateNN_GOOD_CONTROL_RUN": "bqQYfLQ3S92QPkDv1jywMA",
    "SeparateNN_GOOD_NO_NORM_ADV": "aptbS8bVRNWhZLXXjW0E2w",
    "SeparateNN_GOOD_NO_ANNEAL": "hkEe7S2CQb6fQ4d7JAsefw",
    "SeparateNN_GOOD_NO_GAE": "he6QUEcySryzYpur21t6Wg",
    "SeparateNN_GOOD_NO_GRAD_NORM": "bhlRs3t2QOidVS46uqjJtA",
    "SeparateNN_GOOD_NO_VALUE_CLIP": "Jj3BVD5rSQan5jGotaAKAw",
    "SeparateNN_GOOD_PLAIN": "EDmAeP5nRiOkgeU3m6qW3w",
    "SeparateNN_NO_ADV_NORM": "RD7WFxUdSPyraZeDx69Ztg",
    "SeparateNN_NO_ANNEAL_RATE": "R5wMhxanQO2SzrQc4Mp8cw",
    "SeparateNN_NO_GAE": "bMnvMllwQZaWMoU0rl1gog",
    "SeparateNN_NO_VALUE_CLIP": "U2otyjyXQ1KBxzy5u8Q3Yw",
    "SeparateNN_PLAIN": "gPQ1RaL7RWC8sOGxRyK93A",
    "SeparateNN_PLAIN_ANNEAL": "Jo9GZKiKQQaRxdrOv2dJPw",
}
# for experiment in experiment_ids:
#     experiment_data = tb.data.experimental.ExperimentFromDev(experiment)
#     df = experiment_data.get_scalars()
#     print(df.head)
#     for frame in df["run"].unique():
#         print(frame)


def write_to_csv(data, path):
    data.to_csv(path, index=False)


experiment_data = tb.data.experimental.ExperimentFromDev("84a77iJQTnGLxyiKnEQMsw")
df = experiment_data.get_scalars()
df_pivot = df.pivot(index=["run", "step"], columns="tag", values="value")
df_pivot = df_pivot.reset_index()
df_pivot.columns = [
    col.split("/")[1] if "/" in col else col for col in df_pivot.columns
]
print(df_pivot)
averaged_df = df_pivot.groupby("step")["episodic_return"].mean().reset_index()
selected_columns = [col for col in df_pivot.columns if col not in ["run", "step"]]
for column in selected_columns:
    averaged_df[column] = df_pivot.groupby("step")[column].mean()

smoothed_df = averaged_df[d].rolling(window=100, min_periods=1).mean()
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# sns.lineplot(
#     data=df_pivot,
#     x="step",
#     y="smoothed_values",
# ).set_title("returns")
# plt.show()
# plt.plot(averaged_df["step"], smoothed_returns)
# plt.xlabel("Step")
# plt.ylabel("Returns/Step")
# plt.title("Smoothed Returns per Step")
# plt.show()
