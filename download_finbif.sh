#!/usr/bin/env bash
# Pull all Finnish bird and mammal observations from FinBIF (laji.fi)
# in two date windows. Output CSVs go in the same archives folder as the
# GBIF zips so combine.py picks everything up uniformly.
#
# Set FINBIF_TOKEN before running:
#   export FINBIF_TOKEN="<token from api.laji.fi email>"
set -euo pipefail

if [[ -z "${FINBIF_TOKEN:-}" ]]; then
    echo "error: FINBIF_TOKEN environment variable is not set." >&2
    echo "       Register: curl -X POST -d email=<addr> https://api.laji.fi/v0/api-users" >&2
    exit 1
fi

ARCHIVES_DIR="/media/pc/HD1/aibirder_model_data/gbif_archives"
mkdir -p "$ARCHIVES_DIR"

# Today's year for the upper bound of the recent window.
YEAR_NOW="$(date +%Y)"

run_window() {
    local from="$1" to="$2"
    local out="$ARCHIVES_DIR/finbif_aves_mammalia_${from}-${to}.csv.gz"
    echo "=== FinBIF pull  ${from}-${to}  →  $(basename "$out") ==="
    # The python script is idempotent: if $out already covers the window
    # it returns immediately; if partial it resumes from the last
    # eventDate (with a small overlap so transient drops re-fetch a few
    # days). Connection errors are retried with exponential backoff.
    .venv/bin/python utils/finbif_download.py \
        --classes Aves Mammalia \
        --year-from "$from" --year-to "$to" \
        --max-accuracy 5000 \
        --mode list --page-size 5000 \
        --output "$out"
    echo
}

run_window 2000 2009

# Then 2010 onward, one year per file. Per-year is more robust:
# - if the API drops the connection mid-year, we resume from that year only;
# - one year fits comfortably under the API's pagination limits;
# - smaller files are easier to inspect / move around.
for year in $(seq 2010 "$YEAR_NOW"); do
    run_window "$year" "$year"
done

echo "Done. Archives:"
ls -lh "$ARCHIVES_DIR"/finbif_aves_mammalia_*.csv.gz 2>/dev/null || true
