# %%
import mlxp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
reader = mlxp.Reader("./logs/", refresh=True)
query: str = "info.status == 'COMPLETE'"
df = pd.DataFrame(reader.filter(query_string=query))


# %%
def plot_last(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, key in zip(axes.flatten(), ["train_acc", "train_cce", "valid_acc", "valid_cce"]):
        ax.plot(df.iloc[-1]["scope." + key])
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(["Last"])


plot_last(df)


# %%
def plot_mean(df, idx):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, key in zip(axes.flatten(), ["train_acc", "train_cce", "valid_acc", "valid_cce"]):
        ax.plot(df.iloc[idx]["scope." + key].mean(-1))
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(["Mean"])


plot_mean(df, idx=-1)
