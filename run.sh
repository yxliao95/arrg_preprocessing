#!/bin/bash

python /home/yuxiang/liao/workspace/arrg_preprocessing/3_radlex_annotate.py
python /home/yuxiang/liao/workspace/util_proj/test_email.py --from_bash --content "radlex done"

# nohup /home/liao/workspace/arrg_prototype/src_x/run.sh > /home/liao/workspace/arrg_prototype/nohup.log 2>&1 &

# Check process and kill
# ps aux | grep <进程名>
# kill <PID>

# Client side:
# ssh -L 6007:localhost:6006 liao@10.96.125.95