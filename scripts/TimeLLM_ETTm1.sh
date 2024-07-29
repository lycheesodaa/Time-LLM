model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llama_layers=8

master_port=55555
num_process=2
batch_size=8
d_model=16
d_ff=32

comment='TimeLLM-ETTm1'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate\
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/ETTm1-96_Llama8_batch16_dmodel16_dff32.txt

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment | tee results/ETTm1-192_Llama8_batch16_dmodel16_dff32.txt
#
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_512_336 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 336 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size $batch_size \
#  --learning_rate $learning_rate \
#  --lradj 'TST'\
#  --learning_rate 0.001 \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --patience 20 \
#  --model_comment $comment
#
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm1.csv \
#  --model_id ETTm1_512_720 \
#  --model $model_name \
#  --data ETTm1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 720 \
#  --factor 3 \
#  --enc_in 7 \
#  --dec_in 7 \
#  --c_out 7 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model $d_model \
#  --d_ff $d_ff \
#  --batch_size $batch_size \
#  --learning_rate $learning_rate \
#  --lradj 'TST'\
#  --learning_rate 0.001 \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --patience 20 \
#  --model_comment $comment