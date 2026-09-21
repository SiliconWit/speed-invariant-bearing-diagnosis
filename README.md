# Spindrift: speed-invariant bearing fault diagnosis

Bearing fault frequencies scale with shaft speed, so machines running at different
speeds see the same fault at different frequencies. This code studies envelope spectrum
representations that turn that change into a shift, and their effect on federated fault
diagnosis across machines that do not share their data.

## Requirements

Python 3.8 or later and a CPU; no GPU is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
```

Tests that read the recordings are skipped, with the reason, until the data is fetched.

## Data

Two public bearing datasets, CWRU and JNU. `data/fetch_data.py` downloads them into
`data/cwru` and `data/jnu` and checks every file against the checksums in
`data/MANIFEST.md`, which also gives their sources and citations.

## Running

```bash
python3 data/fetch_data.py
cd code
OMP_NUM_THREADS=1 python3 run_jnu.py        # federated grid on JNU, 90 runs
OMP_NUM_THREADS=1 python3 run_sim.py        # simulated speed sweep
OMP_NUM_THREADS=1 python3 run_sim.py --seeds 10 --ball-orders 4.86 4.90 4.94 4.98 5.02
                                            # seed spread and ball order sensitivity
OMP_NUM_THREADS=1 python3 run_controls.py   # shift identity and architecture control
OMP_NUM_THREADS=1 python3 run_arch.py       # architecture control with the order-tracking baseline
python3 analyse.py
python3 build_numbers.py
python3 make_figs.py
```

On an eight-core CPU the federated grid takes a few hours, the architecture runs most of
a day, and the simulated sweep about half an hour. The seed and ball order sweep repeats
the simulated sweep for each ball defect order with ten training seeds and takes a few
hours; `--nproc` sets the number of worker processes. When its output
`results/sim_ball_order.json` is present, the analysis adds the seed spread to the
numbers and writes `ball_order_table.tex`. Features are extracted from the
recordings on first use and cached in `data/cache`. `build_numbers.py` reads the recorded
shaft speeds from the CWRU files, so it also needs the data.

## Outputs

Everything is written to `results/`: one JSON file per experiment, `summary.json`,
`numbers.tex` (every reported value as a LaTeX macro), `jnu_table.tex`,
`ball_order_table.tex` (when that sweep has been run) and the figures in `results/figs/`. The environment variables `DATA_DIR`, `RESULTS_DIR` and `PUBLISH_DIR`
change these locations (see `code/paths.py`).

## Layout

```
code/features.py       log-frequency envelope spectrum, linear-axis control, order spectrum
code/sim.py            bearing fault simulator
code/data.py           JNU: splitting by time, windowing, feature cache
code/cwru.py           CWRU: the same, as the low speed-range control
code/models.py         encoder, projection, linear head
code/central.py        centralised training
code/fed.py            FedAvg, FedProx and prototype alignment
code/margin.py         prototype separation, radius, displacement and the margin ratio
code/run_*.py          experiments
code/analyse.py        aggregates the raw output into results/summary.json
code/build_numbers.py  results/summary.json to results/numbers.tex
code/figures.py        figures; code/make_figs.py draws them and the JNU table
code/paths.py          data and output locations
data/                  fetch script and manifest
tests/                 tests
```

## Citation

See `CITATION.cff`.

## License

MIT
