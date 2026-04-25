#!/bin/bash
# Submit Prithvi extraction jobs in priority order for 2025 prediction.
# Waits for slots to free up before submitting the next batch.

PYTHON=/home/ec2-user/miniconda3/envs/Hackathon2026/bin/python
MAX_RUNNING=4
# 2025 first, then most-recent historical years down to 2015
YEARS="2025 2024 2023 2022 2021 2020 2019 2018 2017 2016 2015"

running_count() {
    local n
    n=$( { \
        aws sagemaker list-processing-jobs --status-equals InProgress \
            --query 'ProcessingJobSummaries[].ProcessingJobName' --output text 2>/dev/null
        aws sagemaker list-processing-jobs --status-equals Stopping \
            --query 'ProcessingJobSummaries[].ProcessingJobName' --output text 2>/dev/null
    } | tr '\t' '\n' | grep -v '^None$' | grep -v '^$' | wc -l )
    echo "${n:-0}"
}

for year in $YEARS; do
    for state in IA CO WI MO NE; do
        # Wait until a slot is free
        while [ "$(running_count)" -ge "$MAX_RUNNING" ]; do
            echo "$(date '+%H:%M:%S') Waiting for slot... ($(running_count)/$MAX_RUNNING running)"
            sleep 60
        done
        echo "$(date '+%H:%M:%S') Submitting $state $year"
        $PYTHON sagemaker_run.py --state $state --year $year
        sleep 3  # small gap to avoid throttling
    done
    echo "--- Submitted all states for $year ---"
done

echo "All jobs submitted."
