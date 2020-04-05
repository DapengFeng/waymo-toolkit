import numba
import numpy as np
from numba import boolean, float32, int64


@numba.jit(
    float32[:](float32[:, :], int64, int64, float32, boolean, int64), nopython=True, fastmath=True,
)
def ransac(
    data, goal_inliers, max_iterations, threshold, stop_at_goal, random_seed,
):
    best_ic = 0
    best_model = np.zeros(4, dtype=np.float32)
    np.random.seed(random_seed)
    total = len(data)
    for i in range(max_iterations):
        idx = np.random.choice(total, 3, replace=False)
        sample = data[idx]
        xyza = np.ones((3, 4), dtype=np.float32)
        xyza[:, :3] = sample
        m = np.linalg.svd(xyza)[-1][-1]
        m = m.astype(np.float32)
        m = np.ascontiguousarray(m)
        ic = 0
        for j in range(total):
            xyzs = np.ones((1, 4), dtype=np.float32)
            xyzs[:, :3] = data[j]
            xyzs = np.ascontiguousarray(xyzs)
            if (np.abs(xyzs @ m) < threshold)[0]:
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    return best_model
