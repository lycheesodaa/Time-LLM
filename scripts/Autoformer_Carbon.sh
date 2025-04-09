model_name='Autoformer'

train_epochs=100
learning_rate=0.0001

master_port=55555
num_process=1
batch_size=16

seq_len=96  # autoformer defaults to seq_len 96

# BEFORE RUNNING, MAKE SURE DATA IS UNTRANSFORMED (?)

# no scale
for pred_len in 1 2 3 4 5 7 14 21 28 30 35 40 45 50 55 60 70 80 90
#for pred_len in 90
do
  for feats in 0 2 4 6 8 12
#  for feats in 0 12
  do
    accelerate launch --num_processes $num_process --main_process_port $master_port --gpu_ids 0 run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/carbon/res_daily/ \
      --data_path merged_data_imputed_top${feats}.csv \
      --results_path ./results/carbon_daily/ \
      --model_id ${model_name}_Carbon_${seq_len} \
      --model $model_name \
      --data Carbon \
      --features MS \
      --target 'Price' \
      --freq 'd' \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in $((feats + 1)) \
      --dec_in $((feats + 1)) \
      --c_out $((feats + 1)) \
      --d_model 512 \
      --n_heads 8 \
      --d_ff 2048 \
      --factor 3 \
      --dropout 0.05 \
      --no-scale \
      --des "carbon_daily_window-${feats}" \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate\
      --train_epochs $train_epochs \
      --model_comment 'full' | tee results/logs/${model_name}-Carbon-${pred_len}_batch${batch_size}.txt
  done
done

# scale
for pred_len in 70 80 90
do
  for feats in 0 2 4 6 8 12
  do
    accelerate launch --num_processes $num_process --main_process_port $master_port --gpu_ids 0 run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/carbon/res_daily/ \
      --data_path merged_data_imputed_top${feats}.csv \
      --results_path ./results/carbon_daily/ \
      --model_id ${model_name}_Carbon_${seq_len} \
      --model $model_name \
      --data Carbon \
      --features MS \
      --target 'Price' \
      --freq 'd' \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in $((feats + 1)) \
      --dec_in $((feats + 1)) \
      --c_out $((feats + 1)) \
      --d_model 512 \
      --n_heads 8 \
      --d_ff 2048 \
      --factor 3 \
      --dropout 0.05 \
      --des "carbon_daily_window-${feats}" \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate\
      --train_epochs $train_epochs \
      --model_comment 'full' | tee results/logs/${model_name}-Carbon-${pred_len}_batch${batch_size}.txt
  done
done