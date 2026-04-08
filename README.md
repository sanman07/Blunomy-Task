# LiDAR Wire Fitting

A Python package that takes drone LiDAR scans of power lines and fits a catenary curve to each individual wire.

## The problem

A LiDAR drone flies over power lines and records thousands of (x, y, z) points wherever its laser hits something. Some of those points are on wires, some are noise. The task is to figure out which points belong to which wire, and then fit the mathematical shape of a hanging cable — a catenary — to each one.

The tricky part is that the catenary equation is 2D, but the data is 3D. And the wires run diagonally across the scene at arbitrary angles, so you can't just pick two axes and hope for the best.

## Setup

```bash
git clone <repo-url>
cd Blunomy-Task
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

## Usage

Single file:
```bash
python scripts/run_pipeline.py --input data/lidar_cable_points_easy.parquet
```

All files, save plots to output folder:
```bash
python scripts/run_pipeline.py --input data/ --save-plots --output-dir output/
```

## My approach

### Separating the wires

My first attempt used DBSCAN clustering on all three dimensions. I chose DBSCAN over K-means because I don't know how many wires are in the scene upfront K-means requires you to specify that. DBSCAN finds clusters automatically based on density and labels stray points as noise rather than forcing them into a cluster.

The problem was that standard DBSCAN kept merging multiple wires into one cluster. After some investigation I realised why the separation between wires only happens in one specific direction, perpendicular to the wire. Z wasn't helping at all, it was just adding noise to the distance calculation.

So instead of clustering in 3D, I used PCA on the X-Y coordinates to find the direction perpendicular to the wires, projected all the points onto that single axis, and ran DBSCAN on just that one dimension. That worked much better — easy, hard and extrahard all separate into individual wires cleanly. Medium is still imperfect because it has two sets of wires at different heights, which confuses the cross-wire projection. A two-stage approach would fix it but I didn't get there in the time available.

### Projecting to 2D

For each wire cluster, I use PCA to find the plane the wire lives in and project onto it before fitting the catenary. One thing that caught me out here — PCA orders components by variance, and the wires span a large distance in X-Y but only about a metre in Z. So Z comes out as the third component, not the second. I initially took PC1 and PC2 which gave a flat top-down projection with no sag visible at all. Switching to PC1 and PC3 fixed it and  PC1 is along the wire length, PC3 captures the sag direction.

### Fitting the catenary

The catenary equation is:

```
y(x) = y0 + c * (cosh((x - x0) / c) - 1)
```

`c` controls how tight or flat the wire hangs — small means droopy, large means taut. I use `scipy.optimize.curve_fit` to find the best values for `x0`, `y0`, and `c`.

The starting guess matters a lot because curve_fit is a local optimiser a bad starting point leads to the wrong answer or just fails. I estimate the starting values from the data: midpoint of the wire for `x0`, minimum projected y for `y0`, and span divided by sag for `c`. The span/sag ratio made sense intuitively a wire with a large span and barely any sag should have a large c, and one that dips a lot relative to its span should have a small c.

I also hit a bug here with the plotting. I was deriving the x range to sample the curve from the curvature parameter. For relatively flat wires the curvature is very large, which produced a sampling range way beyond the actual wire the data points ended up as a tiny cluster in the corner of a massive empty plot. Fixed by just sampling over the actual range of the data instead.

### Getting back to 3D

Once the catenary is fitted in 2D, I sample points along it and project back into 3D using the stored PCA axes.


## Structure

```
Blunomy-Task/
├── pyproject.toml
├── requirements.txt
├── lidar_wire_fitting/
│   ├── __init__.py
│   ├── loader.py          # loads and validates parquet files
│   ├── clustering.py      # cross-wire PCA + DBSCAN
│   ├── fitting.py         # PCA projection and catenary fitting
│   └── visualisation.py   # 3D plots
├── scripts/
│   └── run_pipeline.py    # entry point
├── data/
└── output/
```

## Results

| File | Wires found | Notes |
|---|---|---|
| easy | 3 | works well |
| hard | 3 | fits are noisier, points are sparse |
| extrahard | 2 of 3 | third wire fit didn't converge |
| medium | 1 | should be 6 — two height bands confuse the cross-wire projection |

## Future fix with more time

**Medium clustering** — split by height band first, then apply the cross-wire clustering within each band. Should give 6 individual wire fits instead of one average.

**PCA component selection** — using PC1 and PC3 assumes Z is always the sag direction, which holds for this dataset but isn't general. A more robust approach would identify the sag direction from the data shape rather than assuming it.

**curve_fit robustness** — fitting fails silently on some clusters. A fallback when convergence fails, or better initial estimates, would make it more better.

**Parallelisation** — each wire is fitted independently so this is a natural fit for multiprocessing. A lot better for larger datasets.