"""Where the code reads data and writes results.

Defaults follow the repository layout:

    data/jnu, data/cwru      recordings, fetched by data/fetch_data.py
    data/cache               extracted features, rebuilt on demand
    results/*.json           raw experiment output and the summary
    results/numbers.tex      every quoted number, as LaTeX macros
    results/jnu_table.tex    the JNU results table
    results/figs/            the figures

Three environment variables override the defaults:

    DATA_DIR      directory holding jnu/, cwru/ and cache/
    RESULTS_DIR   directory for the JSON files
    PUBLISH_DIR   directory for numbers.tex, jnu_table.tex and figs/

An optional file local_paths.py next to this one may set DATA, RESULTS or PUBLISH
directly; it is not part of the repository.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
PUBLISH = os.environ.get("PUBLISH_DIR", RESULTS)

try:
    from local_paths import *  # noqa: F401,F403
except ImportError:
    pass

JNU = os.path.join(DATA, "jnu")
CWRU = os.path.join(DATA, "cwru")
CACHE = os.path.join(DATA, "cache")
FIGS = os.path.join(PUBLISH, "figs")
