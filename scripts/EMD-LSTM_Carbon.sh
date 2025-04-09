model_name='LSTM'

train_epochs=100
learning_rate=0.001

master_port=55555
num_process=1
batch_size=16

seq_len=512
d_model=200

#for pred_len in 1 2 3 4 5 7 14 21 28 30 35 40 45 50 55 60 70 80 90
#for pred_len in 1
for pred_len in 180 365 545
do
  for decomp in 'emd' 'eemd' 'ceemdan' 'wavelet'
#  for decomp in 'emd'
#  for decomp in 'eemd'
#  for decomp in 'ceemdan'
#  for decomp in 'wavelet'
  do
    echo "Running ${decomp} pl${pred_len}..."

    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 0 run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/carbon/res_daily/decomposed_data/ \
      --data_path "${decomp}_decomposition_${pred_len}.h5" \
      --results_path ./results/carbon_daily/${decomp}/ \
      --model_id ${model_name}_Carbon_${decomp}_${seq_len}_${pred_len} \
      --model $model_name \
      --data Carbon_Daily_Decomp \
      --features MS \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --target 'Price' \
      --d_model $d_model \
      --no-scale \
      --des "carbon_daily-${decomp}" \
      --itr 1 \
      --batch_size $batch_size \
      --lradj 'TST' \
      --learning_rate $learning_rate\
      --train_epochs $train_epochs \
      --model_comment 'full' | tee results/logs/${model_name}-Carbon-${pred_len}_batch${batch_size}_dmodel${d_model}.txt
  done
done

##for pred_len in 1 2 3 4 5 7 14 21 28 30 35 40 45 50 55 60 70 80 90
#for pred_len in 180 365 545
#do
#  for decomp in 'emd' 'eemd' 'ceemdan' 'wavelet'
#  do
#    accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --gpu_ids 0 run_main.py \
#      --task_name long_term_forecast \
#      --is_training 1 \
#      --root_path ./dataset/carbon/res_daily/ \
#      --data_path merged_data_imputed_top0.csv \
#      --results_path ./results/carbon_daily/ \
#      --model_id ${model_name}_Carbon_${seq_len}_${pred_len} \
#      --model $model_name \
#      --data Carbon \
#      --features S \
#      --seq_len $seq_len \
#      --label_len 0 \
#      --pred_len $pred_len \
#      --enc_in 1 \
#      --target 'Price' \
#      --d_model $d_model \
#      --no-scale \
#      --des "carbon_daily-${decomp}" \
#      --itr 1 \
#      --batch_size $batch_size \
#      --lradj 'TST' \
#      --learning_rate $learning_rate\
#      --train_epochs $train_epochs \
#      --decomp_type $decomp \
#      --model_comment 'full' | tee results/logs/${model_name}-Carbon-${pred_len}_batch${batch_size}_dmodel${d_model}.txt
#  done
#done
