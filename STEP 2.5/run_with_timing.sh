#!/bin/bash
# Timed run script - save as run_with_timing.sh

echo "⏱️  STEP 2.5 Timed Run"
echo "===================="
echo "Start time: $(date)"
echo ""

# Record start time
START_TIME=$(date +%s)

# Create timing log
echo "$(date): Starting STEP 2.5 process" > timing_log.txt

# First, fix the AOI processor issue
echo "🔧 Applying AOI processor fix..."
cp aoi_processor.py aoi_processor.py.backup
# You'll need to manually edit aoi_processor.py with the fix from above

# Run dry-run first to see what we're downloading
echo "🔍 Running dry-run to see requirements..."
echo "$(date): Starting dry-run" >> timing_log.txt

python download_dem_for_aois.py --dry-run

echo ""
echo "Press Enter to continue with actual download, or Ctrl+C to cancel..."
read

# Start actual download
echo "📥 Starting actual download..."
echo "$(date): Starting download" >> timing_log.txt

# Run with timing and progress monitoring
time python download_dem_for_aois.py \
    --buffer 2.0 \
    --max-concurrent 3 \
    --log-level INFO \
    2>&1 | tee download_output.log

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 STEP 2.5 Complete!"
echo "===================="
echo "End time: $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "$(date): Process completed in ${HOURS}h ${MINUTES}m ${SECONDS}s" >> timing_log.txt

# Show summary
echo ""
echo "📊 Performance Summary:"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"

# Check output size
OUTPUT_DIR="../STEP 3/data/raw/dem"
if [ -d "$OUTPUT_DIR" ]; then
    OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
    echo "Output size: $OUTPUT_SIZE"
fi

# Show what was created
if [ -d "$OUTPUT_DIR/mosaics" ]; then
    MOSAIC_COUNT=$(ls -1 "$OUTPUT_DIR/mosaics"/*.tif 2>/dev/null | wc -l)
    echo "DEM mosaics created: $MOSAIC_COUNT"
fi

echo ""
echo "💡 AWS Comparison:"
echo "Local time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Estimated AWS time: ~30-60 minutes"
echo "AWS cost estimate: ~\$10-15"

if [ $HOURS -gt 2 ]; then
    echo "🚀 Recommendation: AWS would be significantly faster"
elif [ $HOURS -gt 1 ]; then
    echo "⚖️  Recommendation: AWS marginally better, your choice"
else
    echo "✅ Recommendation: Local processing is fine"
fi