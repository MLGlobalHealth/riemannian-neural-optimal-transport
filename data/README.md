# Continental point clouds

The two CSV files under `continental/` are unchanged copies of the original research repository's `data/land_points_past.csv` and `data/land_points_present.csv`. Each has 50,000 data rows and the header `lat,lon,plate_id`. The experiment recipe validates these exact hashes before training.

| File | Role | SHA256 |
|---|---|---|
| `continental/land_points_past.csv` | Source point cloud, described by the original notebook as 150 million years ago | `1afe72623f68d42ded0c1c90de223fb050f4298ff272ca6401d32cb40e1d2fee` |
| `continental/land_points_present.csv` | Present-day target point cloud | `17ff2ebfa9fa7acb367b9e21adb994e1a501777e97143decc8f437a5fa33357f` |

`lat` and `lon` are latitude/longitude in degrees. `plate_id` is the plate-polygon identifier assigned by the original data-generation notebook. The current transport training uses the coordinates; retaining plate IDs does not by itself reproduce the paper's plate-purity analysis.

The original `continental_drift_data.ipynb` specifies 50,000 points at 0 Ma and 150 Ma, requests coastlines and plate polygons from the GPlates Web Service, selects the `ZAHIROVIC2022` reconstruction model, samples points conditioned on land, and assigns plate IDs using polygon lookup. This describes the recovered notebook code and its intended data construction. The release does not rerun that external service or establish that a current service response would reproduce these bytes.

The packaged CSVs and their hashes are sufficient for the local runner; no network download is needed. The explicit recipe is [`paper/configs/continental.json`](../paper/configs/continental.json). Training, smoke checks and sample rendering are documented in the [experiment guide](../docs/experiments.md#figure-1-continental-drift). The full research repository preserves the original notebooks under `archive/original/`; the standalone distribution carries the data and provenance without depending on that archive.
