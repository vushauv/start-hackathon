#!/usr/bin/env bash
# usage: ./run_par.sh <split-name> <data-dir> <out-dir> [workers]   (provider openai gpt-5.6-luna)
set -e; S=$1; D=$2; O=$3; W=${4:-6}; export HF_HUB_OFFLINE=1
mkdir -p $O
[ -f $O/evidence.jsonl ] || uv run pipeline retrieve --split $S --data $D --evidence $O/evidence.jsonl 2>&1 | grep -v Loading
python3 -c "
import json,sys;ids=[json.loads(l)['question_id'] for l in open('$D/questions.jsonl')]
for i in range($W): open('$O/ids_%d.txt'%i,'w').write(' '.join(ids[i::$W]))"
for i in $(seq 0 $((W-1))); do
  uv run pipeline answer --split $S --data $D --provider openai --model gpt-5.6-luna --evidence $O/evidence.jsonl \
     --results $O/r_$i.jsonl --question-ids $(cat $O/ids_$i.txt) > $O/answer_$i.log 2>&1 &
done; wait
cat $O/r_*.jsonl > $O/results.jsonl
uv run pipeline predict --split $S --data $D --provider openai --evidence $O/evidence.jsonl --results $O/results.jsonl --predictions $O/predictions.jsonl 2>&1 | tail -1
[ -f $D/gold.jsonl ] && uv run pipeline score --split $S --data $D --provider openai --predictions $O/predictions.jsonl 2>&1 | tail -7 | tee $O/score.txt || true
echo FINISHED
