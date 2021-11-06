import sys
import pandas as pd
import logging

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",  # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    df = pd.read_csv(sys.argv[1], skiprows=1)
    res = [int(n.split('_')[0]) == int(p.split('-')[0]) for n, p in zip(df["name"], df["top1 class"])]
    logging.info(f"top-1: {sum(res) / len(res)}")