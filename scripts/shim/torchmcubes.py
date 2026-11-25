"""Drop-in replacement for torchmcubes (which needs compiling) using PyMCubes."""
import mcubes
import numpy as np
import torch


def marching_cubes(vol, thresh):
    v, f = mcubes.marching_cubes(vol.detach().cpu().numpy().astype(np.float32), float(thresh))
    v = v[:, ::-1].copy()  # torchmcubes reports xyz in the reverse of array-axis order
    return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(f.astype(np.int64))
