#!/bin/bash
# The checksum preflight compares against. Run from the repository root:
#   EXPECTED_HARNESS_SUM=$(benchmark/scripts/harness-sum.sh) bash preflight.sh
cd "$(dirname "$0")/../src" || exit 1
# LC_ALL=C so macOS and Linux agree on the order; without it the two sides
# disagree on identical trees and the check cries wolf.
find . -name '*.py' -not -path '*__pycache__*' | LC_ALL=C sort | xargs cat 2>/dev/null | md5sum | cut -d' ' -f1
