"""Mine frequent itemsets and association rules on the climate dataset.

Usage:
    python scripts/mine_association_rules.py --support 0.3 --confidence 0.5 \\
        --method confidence --discretization equal_frequency
"""

import argparse

import _bootstrap  # noqa: F401
from datamining import config
from datamining.association_rules import AssociationRules
from datamining.datasets import load_climate

METHODS = {
    "confidence": "get_best_rules",
    "lift": "get_best_rules_lift",
    "cosine": "get_best_rules_cosine",
}
COLUMNS = {"Temperature": "Temp", "Humidity": "Hum", "Rainfall": "Rain"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--support", type=float, default=0.3, help="minimum support ratio")
    p.add_argument("--confidence", type=float, default=0.5, help="minimum rule score")
    p.add_argument("--method", choices=METHODS, default="confidence")
    p.add_argument(
        "--discretization",
        choices=["equal_frequency", "equal_width"],
        default="equal_frequency",
    )
    p.add_argument("--classes", type=int, default=0, help="0 = Sturges-like default")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_climate(config.CLIMATE_RAW)
    ar = AssociationRules(data)
    discretize = getattr(ar, args.discretization)
    for col, prefix in COLUMNS.items():
        data[col] = discretize(data, col, prefix, args.classes)
    ar.setDataset(data)

    itemsets = ar.appriori(args.support)
    print(f"{len(itemsets)} frequent itemsets (support >= {args.support})")
    rules = getattr(ar, METHODS[args.method])(itemsets, args.confidence)
    for itemset, itemset_rules in rules.items():
        for lhs, rhs in itemset_rules:
            print(f"  {lhs} -> {rhs}")


if __name__ == "__main__":
    main()
