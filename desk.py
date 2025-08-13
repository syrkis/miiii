# %%
import matplotlib.pyplot as plt
import mlxp
import numpy as np
import pandas as pd

# %%
reader = mlxp.Reader("./logs/", refresh=True)
query: str = "info.status == 'COMPLETE'"
res = pd.DataFrame(reader.filter(query_string=query))
plt.plot(res["train.loss"].iloc[-1])

# %%
plt.plot(np.array(res["scope.valid_acc"].iloc[-1]).T.flatten())
