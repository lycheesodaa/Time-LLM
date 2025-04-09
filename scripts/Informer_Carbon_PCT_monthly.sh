model_name='Informer'

train_epochs=100
learning_rate=0.001

master_port=55555
num_process=1
batch_size=16

#for feats in 0 10 20 30 40
for feats in 0 20
do
  #for seq_len in 2 4 6 8 10 12 14 16 18 20
  for seq_len in 2 8 14 20 # Currently have errors with such short seqlen, must add padding
  do
    for pred_len in 1 2 4 6 8 10 12 14 16 18
#    for pred_len in 1 18
#    for pred_len in 1
    do
      accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 1 run_main.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/carbon/res/ \
        --data_path merged_data.csv \
        --results_path ./results/carbon_monthly/ \
        --model_id ${model_name}_Carbon_monthly_${seq_len}_${pred_len} \
        --model $model_name \
        --data Carbon_Monthly \
        --features MS \
        --target 'Price' \
        --freq 'm' \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --feats_pct $feats \
        --d_model 512 \
        --n_heads 8 \
        --d_ff 2048 \
        --factor 5 \
        --dropout 0.05 \
        --no-scale \
        --des "carbon_monthly-feat${feats}%" \
        --itr 1 \
        --batch_size $batch_size \
        --learning_rate $learning_rate\
        --train_epochs $train_epochs \
        --model_comment 'full' | tee results/logs/${model_name}-Carbon-${pred_len}_batch${batch_size}.txt
    done
  done
done