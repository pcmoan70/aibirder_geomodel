#!/usr/bin/env bash
# Pull Finnish vascular-plant, moss, fungi & lichen observations from
# FinBIF (laji.fi) — the GBIF mirror is incomplete for these taxa, so we
# go through the FinBIF API directly. Same yearly-windows pattern as
# download_finbif.sh.
#
# Set FINBIF_TOKEN before running.
set -euo pipefail

if [[ -z "${FINBIF_TOKEN:-}" ]]; then
    echo "error: FINBIF_TOKEN environment variable is not set." >&2
    echo "       Register: curl -X POST -d email=<addr> https://api.laji.fi/v0/api-users" >&2
    exit 1
fi

ARCHIVES_DIR="/media/pc/HD1/aibirder_model_data/gbif_archives"
mkdir -p "$ARCHIVES_DIR"
YEAR_NOW="$(date +%Y)"

# FinBIF taxon umbrellas (see CLASS_TO_GROUP in utils/finbif_download.py):
#   vascular_plants → MVL.343
#   mosses          → MVL.561
#   fungi           → MVL.233 (umbrella incl. lichens)
GROUPS=(vascular_plants mosses fungi)
LABEL=$(IFS=_ ; echo "${GROUPS[*]}")    # → vascular_plants_mosses_fungi

run_window() {
    local from="$1" to="$2"
    local out="$ARCHIVES_DIR/finbif_${LABEL}_${from}-${to}.csv.gz"
    echo "=== FinBIF pull  ${from}-${to}  →  $(basename "$out") ==="
    .venv/bin/python utils/finbif_download.py \
        --classes "${GROUPS[@]}" \
        --year-from "$from" --year-to "$to" \
        --max-accuracy 5000 \
        --mode list --page-size 5000 \
        --output "$out"
    echo
}

run_window 2000 2009

for year in $(seq 2010 "$YEAR_NOW"); do
    run_window "$year" "$year"
done

echo "Done. Archives:"
ls -lh "$ARCHIVES_DIR"/finbif_${LABEL}_*.csv.gz 2>/dev/null || true
