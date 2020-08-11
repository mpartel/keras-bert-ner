#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p gpu
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

function on_exit {
    rm -f out-$SLURM_JOBID.tsv
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 model data_dir seq_len batch_size learning_rate epochs"
    exit 1
fi

model="$1"
data_dir="$2"
max_seq_length="$3"
batch_size="$4"
learning_rate="$5"
epochs="$6"

vocab="$(dirname "$model")/vocab.txt"
config="$(dirname "$model")/bert_config.json"

if [[ $model =~ "uncased" ]]; then
    caseparam="--do_lower_case"
elif [[ $model =~ "multilingual" ]]; then
    caseparam="--do_lower_case"
else
    caseparam=""
fi

rm -f logs/latest.out logs/latest.err
ln -s "$SLURM_JOBID.out" "logs/latest.out"
ln -s "$SLURM_JOBID.err" "logs/latest.err"

module purge
module load tensorflow
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun python ner.py \
    --vocab_file "$vocab" \
    --bert_config_file "$config" \
    --init_checkpoint "$model" \
    --learning_rate $learning_rate \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --batch_size $batch_size \
    --train_data "$data_dir/train.tsv" \
    --test_data "$data_dir/dev.tsv" \
    --output_file "out-$SLURM_JOBID.tsv" \
    $caseparam

result=$(python conlleval.py out-$SLURM_JOBID.tsv \
    | egrep '^accuracy' | perl -pe 's/.*FB1:\s+(\S+).*/$1/')

echo -n 'DEV-RESULT'$'\t'
echo -n 'init_checkpoint'$'\t'"$model"$'\t'
echo -n 'data_dir'$'\t'"$data_dir"$'\t'
echo -n 'max_seq_length'$'\t'"$max_seq_length"$'\t'
echo -n 'train_batch_size'$'\t'"$batch_size"$'\t'
echo -n 'learning_rate'$'\t'"$learning_rate"$'\t'
echo -n 'num_train_epochs'$'\t'"$epochs"$'\t'
echo 'FB1'$'\t'"$result"

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"
