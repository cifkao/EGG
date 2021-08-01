#!/usr/bin/env python3
import argparse

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, help="the upper input range (non-inclusive)")
    parser.add_argument("M", type=int, help="the maximum number of examples per result (class)")
    parser.add_argument("--prefix", type=str, default="sum.", nargs='?')
    parser.add_argument("--test-prob", type=float, default=0.1,
                        help="probability that a tuple ends up in test set")
    parser.add_argument("--resample", action="store_true",
                        help="balance training dataset by resampling exactly K tuples "
                             "with replacement for each class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N, M = args.N, args.M

    with open(args.prefix + "train", "w") as f_train, \
         open(args.prefix + "test", "w") as f_test:
        rng = np.random.default_rng(args.seed)
        # Loop over all possible results
        num_results = 2 * (N - 1) + 1
        for r in range(num_results):
            # Sample at most M possible input tuples without replacement.
            # Sample the first term within the intersection of [r-N+1, N) and [0, r],
            # this ensures that both terms are within [0, N) and sum up to r.
            a_range = range(max(0, r - N + 1), min(N, r + 1))
            a_samples = rng.choice(a_range, size=min(M, len(a_range)), replace=False)

            # Do random train-test split; it is important to do that here since we want different
            # tuples in each split
            test_set_mask = rng.binomial(n=1, p=args.test_prob, size=len(a_samples)).astype(bool)
            # Make sure at least one tuple ends up in the training set
            if (~test_set_mask).sum() == 0:
                test_set_mask[rng.choice(len(a_samples))] = False

            # Write test samples
            for a in a_samples[test_set_mask]:
                print(a, r - a, r, file=f_test)

            # Write training samples
            a_samples_train = a_samples[~test_set_mask]
            if args.resample:
                a_samples_train = rng.choice(a_samples_train, size=M, replace=True)
            for a in a_samples_train:
                print(a, r - a, r, file=f_train)