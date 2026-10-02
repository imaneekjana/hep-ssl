#!/usr/bin/env bash
# Run on the Mac. Creates/upload a new deployment; does NOT submit jobs.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/Users/clintli/Desktop/hep_ssl}"
for program in python3 ssh scp; do command -v "$program" >/dev/null; done
DEST="$(python3 "$KIT/build_deployment.py" --project "$PROJECT" --kit "$KIT")"
source "$DEST/location.env"
printf '\nCreated local deployment: %s\n' "$DEST"
# Check only file existence on the transfer host; no data is downloaded to the Mac.
ssh kli398@transfer.chtc.wisc.edu \
  "test -r /staging/k/kli398/colliderml-data-pairwise-2500.tar.gz && test -r /staging/k/kli398/hep_ssl.sif && mkdir '$STAGE_DIR'"
ssh kli398@ap2002.chtc.wisc.edu \
  "mkdir -p hep_ssl_chtc && mkdir '$DEPLOYMENT_REL'"
scp "$DEST/"* "kli398@ap2002.chtc.wisc.edu:$DEPLOYMENT_REL/"
# This small pointer is only a convenience; existing run directories are not overwritten.
ssh kli398@ap2002.chtc.wisc.edu \
  "printf '%s\n' '$DEPLOYMENT_REL' > hep_ssl_chtc/LATEST_PHASE1_DEPLOYMENT.txt"
printf '\nUPLOAD_COMPLETE: no jobs submitted.\n'
printf 'Local config: %s/pairwise_chtc.json\n' "$DEST"
printf 'SSH: ssh kli398@ap2002.chtc.wisc.edu\n'
printf 'Then: cd "$HOME/%s"\n' "$DEPLOYMENT_REL"
printf 'Prepare submit: condor_submit 01_prepare.sub\n'
