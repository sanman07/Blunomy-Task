# LiDAR Wire Fitting

Processes drone-captured LiDAR point clouds of electricity lines, clusters the individual wires, and fits a 3D catenary model to each one.

## What it does

Takes a `.parquet` file of (x, y, z) LiDAR points and runs them through a pipeline: clean → cluster → project to 2D → fit catenary → reconstruct in 3D and plot.

## How it works

### Clustering (DBSCAN)
DBSCAN rather than K-means, because we don't know how many wires are in the scene. It also handles noisy/outlier points well — stray LiDAR returns just get labelled as noise instead of polluting a cluster.

### PCA projection
The catenary equation is 2D, but wires hang in 3D at arbitrary orientations. PCA on each cluster gives us the plane the wire lives in, and we project onto that before fitting.

### Catenary fitting
The catenary equation (`y = y0 + c*(cosh((x-x0)/c) - 1)`) is fit using `scipy.optimize.curve_fit`. `c` controls sag — small means droopy, large means taut. Starting guesses for `x0`, `y0`, and `c` are estimated from the projected data (span and sag range), which matters a lot for convergence on unusual wire geometries.

### 3D reconstruction
The fitted 2D curve is unprojected back into 3D via the inverse PCA transform.


