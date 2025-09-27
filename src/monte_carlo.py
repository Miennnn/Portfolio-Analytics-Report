
import numpy as np
import pandas as pd

def simulate_paths(start_value, mu, sigma, years, sims=5000, steps_per_year=12, contributions=0.0):
    dt = 1/steps_per_year
    steps = int(years*steps_per_year)
    paths = np.zeros((sims, steps+1))
    paths[:,0] = start_value
    for t in range(1, steps+1):
        z = np.random.normal(0,1,size=sims)
        growth = np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
        paths[:, t] = np.maximum(0.0, paths[:, t-1]*growth + contributions)
    return paths

def percentile_bands(paths: np.ndarray, ps=(5,50,95)):
    qs = np.percentile(paths, ps, axis=0)
    return pd.DataFrame(qs.T, columns=[f"p{p}" for p in ps])
