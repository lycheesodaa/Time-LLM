model_name='LSTM'

train_epochs=100
learning_rate=0.001

master_port=55555
num_process=1
batch_size=16

seq_len=512
d_model=200

#for decomp in 'emd'
  for decomp in 'eemd'
#  for decomp in 'ceemdan'
#  for decomp in 'wavelet'
do
  for seq_len in 20
#  for seq_len in 8 14 20 # min length for wavelet is 8
  do
    for pred_len in 1 2 4 6 8 10 12 14 16 18
    do
      for feats_pct in 20
      do
        accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 1 run_main.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path ./dataset/carbon/res/ \
          --data_path merged_data.csv \
          --results_path ./results/carbon_monthly_decomp/ \
          --model_id ${model_name}_Carbon_${decomp}_${seq_len}_${pred_len} \
          --model $model_name \
          --data Carbon_Monthly \
          --feats_pct $feats_pct \
          --features MS \
          --seq_len $seq_len \
          --label_len 0 \
          --pred_len $pred_len \
          --target 'Price' \
          --d_model $d_model \
          --no-scale \
          --des "carbon_monthly-window-${decomp}-feat${feats_pct}" \
          --itr 1 \
          --batch_size $batch_size \
          --lradj 'TST' \
          --learning_rate $learning_rate\
          --train_epochs $train_epochs \
          --decomp_type $decomp \
          --model_comment 'full' | tee results/logs/${model_name}-Carbon-${pred_len}_batch${batch_size}_dmodel${d_model}.txt
      done
    done
  done
done
