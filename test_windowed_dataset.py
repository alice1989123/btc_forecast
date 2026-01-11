import pandas as pd
import numpy as np
from btc_forecast.windowed_dataset import WindowedDataset  # <-- adjust path if different
import inspect
print("WindowedDataset loaded from:", inspect.getfile(WindowedDataset))
print("Source preview:\n", "\n".join(inspect.getsource(WindowedDataset).splitlines()[:40]))
df = pd.DataFrame({"close": np.arange(20, dtype=float)})
ds = WindowedDataset(df, input_width=5, label_width=3, shift=0, variables_used=["close"])

x, y = ds[0]
print("x:", x.squeeze().tolist())
print("y:", y.squeeze().tolist())