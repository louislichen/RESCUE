#!/usr/bin/env bash
# Unexecuted protocol template: deliberately has no state-changing branch.
set -Eeuo pipefail
printf '%s\n' 'DISABLED: both baseline studies, retained union, fresh-data rule and final sources must be independently verified and frozen before any confirmation launch.' >&2
exit 64
