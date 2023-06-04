import pandas as pd
import tensorboard as tb

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


def write_to_csv(data, path):
    data.to_csv(path, index=False)


dfs = []

for experiment_name, experiment_id in experiment_ids.items():
    print(f"Processing experiment {experiment_name}")
    experiment_data = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment_data.get_scalars()
    df_pivot = df.pivot(index=["run", "step"], columns="tag", values="value")
    df_pivot = df_pivot.reset_index()
    df_pivot.columns = [
        col.split("/")[1] if "/" in col else col for col in df_pivot.columns
    ]
    selected_columns = [col for col in df_pivot.columns if col not in ["run", "step"]]
    averaged_df = df_pivot.groupby("step")[selected_columns].mean().reset_index()
    # smoothed_df = averaged_df[selected_columns].rolling(window=100, min_periods=1).mean()
    smoothed_df = (
        averaged_df[selected_columns]
        .ewm(alpha=0.001, min_periods=1, adjust=True)
        .mean()
    )
    smoothed_df["tag"] = [f"{experiment_name}" for _ in range(len(smoothed_df.index))]
    smoothed_df["step"] = averaged_df["step"]

    write_to_csv(df_pivot, f"{experiment_name}_RAW.csv")
    # write_to_csv(smoothed_df, f"{experiment_name}_smoothed.csv")

    dfs.append(smoothed_df)
    print("----------------------DONE----------------------")

print("Writing the combine smoothed results")
combined_df = pd.concat(dfs, axis=0, ignore_index=True)
write_to_csv(combined_df, "runs_smoothed.csv")
