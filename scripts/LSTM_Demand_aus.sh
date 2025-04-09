model_name=LSTM
#model_name=ConvLSTM
#model_name=GRU
#model_name=GRUAttention
#model_name=$1

train_epochs=100
learning_rate=0.001

master_port=55555
num_process=1
batch_size=16

seq_len=512
d_model=200

comment='full'
for run_name in '_top0' '_top5' '_top9' ''
#for run_name in ''
#for run_name in '_daily' '_daily_top9' '_daily_top5' '_daily_top0'
do
  for pred_len in 1 12 24 48 72 168 336
#  for pred_len in 1 7 14 30 60 180 365
  do
    echo "Running demand aus${run_name} pl${pred_len}..."

    # enc_in is 28 for daily and 25 for weekly/monthly
    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 0 run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/demand/ \
      --data_path demand_data_all_nsw${run_name}.csv \
      --results_path ./results/data_aus${run_name}/ \
      --model_id ${model_name}_Demand_aus_${seq_len}_${pred_len}${run_name} \
      --model $model_name \
      --data Demand \
      --features MS \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --target 'actual' \
      --no-scale \
      --des "demand_aus${run_name}" \
      --itr 1 \
      --d_model $d_model \
      --batch_size $batch_size \
      --lradj 'TST'\
      --learning_rate $learning_rate\
      --train_epochs $train_epochs \
      --model_comment $comment | tee results/logs/${model_name}-Demand-aus-${pred_len}_batch${batch_size}_dmodel${d_model}${run_name}.txt
  done
done
