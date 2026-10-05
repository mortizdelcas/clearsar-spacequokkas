#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

DEFAULT_REFERENCE = "predictions/final.json"
DEFAULT_REFERENCE_MEMBER = "final.json"


def _sha256_iter(it):
    h = hashlib.sha256()
    for chunk in it:
        h.update(chunk)
    return h.hexdigest()


def sha256_file(p: Path) -> str:
    with open(p, "rb") as f:
        return _sha256_iter(iter(lambda: f.read(8 * 1024 * 1024), b""))


def sha256_zip_member(zip_path: Path, member: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        if member not in zf.namelist():
            raise SystemExit(
                f"member {member!r} not found in {zip_path} "
                f"(available: {zf.namelist()})")
        with zf.open(member) as f:
            return _sha256_iter(iter(lambda: f.read(8 * 1024 * 1024), b""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help="Reference predictions (.json), used when neither "
                         "--reference-zip nor --reference-sha256 is given.")
    ap.add_argument("--reference-zip", default=None)
    ap.add_argument("--reference-member", default=DEFAULT_REFERENCE_MEMBER)
    ap.add_argument("--reference-sha256", default=None,
                    help="If set, compare against this SHA-256 directly "
                         "(skips reading the ZIP).")
    args = ap.parse_args()

    cand = Path(args.candidate)
    if not cand.exists():
        print(f"[ERROR] candidate not found: {cand}")
        return 2
    if cand.suffix.lower() != ".json":
        print(f"[ERROR] candidate must be a .json file (got {cand.suffix})")
        return 2

    if args.reference_sha256:
        expected = args.reference_sha256.lower().strip()
        ref_label = f"<--reference-sha256> ({expected})"
    elif args.reference_zip is None:
        ref = Path(args.reference)
        if not ref.exists():
            print(f"[ERROR] reference not found: {ref}")
            return 2
        expected = sha256_file(ref)
        ref_label = str(ref)
    else:
        ref_zip = Path(args.reference_zip)
        if not ref_zip.exists():
            print(f"[ERROR] reference ZIP not found: {ref_zip}")
            return 2
        expected = sha256_zip_member(ref_zip, args.reference_member)
        ref_label = f"{ref_zip}::{args.reference_member}"

    actual = sha256_file(cand)

    print("=" * 72)
    print(f"Verifying {cand}")
    print("=" * 72)
    print(f"  reference: {ref_label}")
    print(f"  expected : {expected}")
    print(f"  actual   : {actual}")
    print()

    if actual == expected:
        print("[PASS] candidate is byte-equal to reference.")
        return 0
    print("[FAIL] candidate differs from reference.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
