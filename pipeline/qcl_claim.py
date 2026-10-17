#!/usr/bin/env python3
"""Claim one mapping queue for a worker, atomically.

  python3 pipeline/qcl_claim.py WORK RUN      # prints the claimed queue path, or nothing when no queue is left (exit 2)

Moves exactly one file WORK/pending/<RUN>NN.txt to WORK/claimed/ with an atomic rename; two workers can never get the
same queue, and an empty pending folder never moves anything (a shell `mv` with an empty name would move the folder)."""
import os, sys
W, run = sys.argv[1], sys.argv[2]
os.makedirs(f"{W}/claimed", exist_ok=True)
for f in sorted(os.listdir(f"{W}/pending")) if os.path.isdir(f"{W}/pending") else []:
    if not (f.startswith(run) and f.endswith(".txt") and os.path.isfile(f"{W}/pending/{f}")): continue
    try: os.rename(f"{W}/pending/{f}", f"{W}/claimed/{f}")
    except FileNotFoundError: continue          # another worker took it first
    print(f"{W}/claimed/{f}"); sys.exit(0)
sys.exit(2)
