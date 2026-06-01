#!/bin/sh

export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# source /work/nakamura.souta/venv/bin/activate.csh
. /work/nakamura.souta/haptic/LSTM/venv/python3.8_cuda11/bin/activate

ARGS="$@"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="./Log/log_B$1_E$2_$timestamp.log"
mkdir -p ./Log/

echo "Test: $ARGS" > $log_file
python3 ./train_speaker_by_cross_model3.py $1 $2 >> "$log_file" 2>&1
