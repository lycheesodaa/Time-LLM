model_name=LSTM
train_epochs=100
learning_rate=0.001

master_port=55555
num_process=1
batch_size=16
#d_model=200

comment='LSTM-Demand'
#for d_model in 160 180 200
for d_model in 200
do
#  for pred_len in 96 192 356
  for pred_len in 1 12 72
  do
    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 0 run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/demand/ \
      --data_path demand_data_all_cleaned.csv \
      --model_id Demand_512_${pred_len} \
      --model $model_name \
      --data Demand \
      --features MS \
      --seq_len 512 \
      --label_len 0 \
      --pred_len $pred_len \
      --enc_in 30 \
      --target 'actual' \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --batch_size $batch_size \
      --lradj 'TST'\
      --learning_rate $learning_rate\
      --train_epochs $train_epochs \
      --model_comment $comment | tee results/LSTM-Demand-${pred_len}_batch16_dmodel${d_model}.txt
  done
done
