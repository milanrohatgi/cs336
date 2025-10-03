#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH --time=12:00:00
#SBATCH --output=/data/c-mrohatgi/web_quality/logs/submitter_%j.out
#SBATCH --error=/data/c-mrohatgi/web_quality/logs/submitter_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

BASE_DIR="/data/c-mrohatgi/web_quality"
mkdir -p $BASE_DIR/urls $BASE_DIR/warc $BASE_DIR/logs

MAX_JOBS=16

cat > $BASE_DIR/fetch_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=fetch_urls
#SBATCH --partition=a4-cpu
#SBATCH --qos=a4-cpu-qos
#SBATCH --time=30:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/c-mrohatgi/web_quality/logs/wget_job_%j.out
#SBATCH --error=/data/c-mrohatgi/web_quality/logs/wget_job_%j.err

cd /data/c-mrohatgi/web_quality/warc

# Run wget in parallel on 4 chunks
printf "%s\n" "$@" | xargs -n1 -P4 bash -c '
  f="$1"
  name=$(basename "$f" .txt)
  wget --timeout=3 \
       -i "$f" \
       --warc-file="$name" \
       -O /dev/null
' _
EOF

chmod +x $BASE_DIR/fetch_job.sh

count_running_jobs() {
    squeue -u $USER -h -t running,pending -n fetch_urls | wc -l
}

chunk_files=( $BASE_DIR/urls/chunk_*.txt )

for (( i=0; i<${#chunk_files[@]}; i+=4 )); do
    batch=( "${chunk_files[@]:i:4}" )

    skip=true
    for f in "${batch[@]}"; do
        name=$(basename "$f" .txt)
        if [ ! -f "$BASE_DIR/warc/${name}.warc.gz" ]; then
            skip=false
            break
        fi
    done
    if $skip; then
        echo "Skipping batch starting at ${batch[0]} (already done)"
        continue
    fi

    while [ $(count_running_jobs) -ge $MAX_JOBS ]; do
        echo "At max jobs ($(count_running_jobs)). Waiting..."
        sleep 5
    done

    echo "Submitting job for: ${batch[*]}"
    sbatch $BASE_DIR/fetch_job.sh "${batch[@]}"

    sleep 1
done

echo "All jobs submitted or queued. Check with: squeue -u $USER"
