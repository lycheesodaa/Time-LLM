model_name=LSTM
train_epochs=100
learning_rate=0.001

master_port=55555
num_process=2
batch_size=8
d_model=160

comment='LSTM-Demand'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_96 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate\
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-96_batch16_dmodel160.txt


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_192 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-192_batch16_dmodel160.txt

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_356 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-356_batch16_dmodel160.txt

d_model=180

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_96 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate\
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-96_batch16_dmodel180.txt


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_192 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-192_batch16_dmodel180.txt

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_356 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-356_batch16_dmodel180.txt

d_model=200

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_96 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate\
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-96_batch16_dmodel200.txt


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_192 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-192_batch16_dmodel200.txt

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/demand/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id Demand_512_356 \
  --model $model_name \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 30 \
  --target 'actual' \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/LSTM-Demand-356_batch16_dmodel200.txt