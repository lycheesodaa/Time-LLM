model_name=LSTM

train_epochs=100
learning_rate=0.001

master_port=55555
num_process=1
batch_size=16

seq_len=1024
pred_len=720
d_model=200

comment='full'
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 1 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_full_daily.csv \
  --results_path ./results/demand_long/ \
  --model_id ${model_name}_Demand_${seq_len}_${pred_len}_long \
  --model $model_name \
  --data Demand \
  --features S \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --target 'system_demand_actual' \
  --no-scale \
  --des "demand_sg_long" \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate\
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/logs/${model_name}-Demand-${pred_len}_batch${batch_size}_dmodel${d_model}_long.txt
