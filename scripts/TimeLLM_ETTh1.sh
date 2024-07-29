model_name=GPT2
train_epochs=50
learning_rate=0.01
llama_layers=6
llm_dim=768

master_port=05555
num_process=2
batch_size=8
d_model=16
d_ff=32

comment='TimeLLM-ETTh1'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --llm_model $model_name \
  --llm_dim $llm_dim \
  --data ETTh1 \
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
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_192 \
#  --model $model_name \
#  --llm_model $model_name \
#  --llm_dim $llm_dim \
#  --data ETTh1 \
#  --features M \
#  --seq_len 512 \
#  --label_len 48 \
#  --pred_len 192 \
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
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment

#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_336 \
#  --model $model_name \
#  --data ETTh1 \
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
#  --lradj 'COS'\
#  --learning_rate 0.001 \
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment
#
#accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_512_720 \
#  --model $model_name \
#  --data ETTh1 \
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
#  --llm_layers $llama_layers \
#  --train_epochs $train_epochs \
#  --model_comment $comment