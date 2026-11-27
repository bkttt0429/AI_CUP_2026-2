#!/usr/bin/env bash
# test of single-node steps inside a multi-node job
i=0; pids=()
for n in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  srun --nodes=1 --ntasks=1 --cpus-per-task=96 --gres=gpu:8 --nodelist=$n --output=logs/nano4/multitest-$SLURM_JOB_ID-$i.out \
    bash -c "export SLURM_JOB_ID=${SLURM_JOB_ID}0$i; echo node \$(hostname) fake id \$SLURM_JOB_ID cvd \$CUDA_VISIBLE_DEVICES gpus \$(nvidia-smi -L | wc -l) cpus \$(nproc) mem \$(free -g | awk 'NR==2{print \$4}') GB python \$(which python); timeout 150 python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(8,4,device=\"cuda\")); y.sum().backward(); torch.cuda.synchronize(); print(\"dp ok\", torch.cuda.device_count())'; sleep 20" &
  pids+=($!); i=$((i+1))
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
cat logs/nano4/multitest-$SLURM_JOB_ID-*.out
echo "steps rc $rc"
