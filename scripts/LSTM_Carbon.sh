#model_name=LSTM
#model_name=ConvLSTM
#model_name=GRU
#model_name=GRUAttention
model_name=$1

train_epochs=100
learning_rate=0.001

master_port=55555
num_process=1
batch_size=16

seq_len=512
d_model=200

#for pred_len in 1 2 3 4 5 7 14 21 28 30 35 40 45 50 55 60 70 80 90
for pred_len in 1 90
do
#  for feats in 0 2 4 6 8 12
  for feats in 0
  do
    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 1 run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/carbon/res_daily/ \
      --data_path merged_data_imputed.csv \
      --results_path ./results/carbon_daily/ \
      --model_id ${model_name}_Carbon_${seq_len}_${pred_len} \
      --model $model_name \
      --data Carbon \
      --features MS \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --feats_pct $feats \
      --target 'Price' \
      --d_model $d_model \
      --des "carbon_daily-${feats}" \
      --itr 1 \
      --batch_size $batch_size \
      --lradj 'TST' \
      --learning_rate $learning_rate\
      --train_epochs $train_epochs \
      --model_comment 'full' | tee results/logs/${model_name}-Carbon-${pred_len}_batch${batch_size}_dmodel${d_model}.txt
#      --no-scale \
  done
done
