# %%
import matplotlib.pyplot as plt
import mlxp
import numpy as np
import pandas as pd

fig, axes = plt.subplots(2, 2, figsize=(10, 5))

reader = mlxp.Reader("./logs/", refresh=True)
query: str = "info.status == 'COMPLETE'"
res = pd.DataFrame(reader.filter(query_string=query))
axes[0, 0].plot(res["train.loss"].iloc[-1])
axes[0, 0].plot(res["train.loss"].iloc[-1])

axes[1, 0].plot(np.array(res["scope.train_cce"].iloc[-1]), label=np.arange(15))
axes[1, 1].plot(np.array(res["scope.train_acc"].iloc[-1]))
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(np.array(res["scope.valid_cce"].iloc[-1]), label=np.arange(15))
axes[0].plot(np.array(res["scope.valid_acc"].iloc[-1]), label=np.arange(15))
plt.legend()
plt.show()
