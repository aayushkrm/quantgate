"""Select the fixed evaluation subsets and write their item IDs to data/subsets/*.json.

Run once, with network access (it downloads the Hub splits named in quantgate.data.SPECS):

    .venv/bin/python scripts/make_subsets.py

The selection is seeded with quantgate.data.SUBSET_SEED, so rerunning gives the same IDs. Only
IDs and selection metadata are written. No dataset text goes into the repo.
"""

from quantgate.data import SPECS, SUBSET_SEED, SUBSETS_DIR, make_subset


def main() -> None:
    """Write one subset file per spec and print its path and size."""
    for spec in SPECS.values():
        path = make_subset(spec, subsets_dir=SUBSETS_DIR, seed=SUBSET_SEED)
        print(f"{spec.name}: {spec.n} IDs -> {path}")


if __name__ == "__main__":
    main()
