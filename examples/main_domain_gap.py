"""
Domain Gap Metric Calculator Script

This script serves as the entry point for calculating the Domain Gap Metric (CMD)
between source and target domains using the GapMetric class.

Usage:
Run this script to compute CMD distance based on the provided configuration.
Optionally, use the '--cfg' argument for the JSON config file path and '--tsk'
argument for the JSON task config file path.
"""

import time
from dqm.domain_gap.metrics import CMD, MMD, Wasserstein, ProxyADistance, FID, KLMVN
import argparse
from dqm.domain_gap.utils import load_config, display_resume


def main():
    """
    Main function for executing the Domain Gap Metric calculation.

    Args: None

    Returns: None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, help="path to config file")

    args = parser.parse_args()

    cfg = load_config(args.cfg)
    method = cfg["METHOD"]["name"]

    start = time.time()
    # Choose the appropriate method and compute the distance
    if method == "cmd":
        cmd = CMD()
        dist = cmd.compute(cfg)

    elif method == "klmvn":
        klmvn = KLMVN()
        dist = klmvn.compute_image_distance(cfg)

    elif method == "mmd":
        mmd = MMD()
        dist = mmd.compute(cfg)

    elif method == "wasserstein":
        wass = Wasserstein()
        if cfg["METHOD"]["dimension"] == "1D":
            dist = wass.compute_1D_distance(cfg)
        elif cfg["METHOD"]["dimension"] == "swd":
            dist = wass.compute_slice_wasserstein_distance(cfg)

    elif method == "proxy":
        pad = ProxyADistance()
        dist = pad.compute_image_distance(cfg)

    elif method == "fid":
        fid = FID()
        dist = fid.compute_image_distance(cfg)

    end = time.time()
    time_lapse = end - start
    display_resume(cfg, dist, time_lapse)


if __name__ == "__main__":
    main()
