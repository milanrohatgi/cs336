#!/bin/bash
# Submit all job scripts
for f in sbatch_jobs/*.sh; do
    sbatch $f
done
