#!/bin/bash

cuda=$1

# CUDA_VISIBLE_DEVICES=0 python run_multi_cho.py --do_train --do_eval --do_lower_case --task_name alphanli --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 32 --model_type bert --load_model_path bert-base-uncased --output_dir alphanli_bert --fp16 --warmup_proportion 0.2
# HellaSWAG 2e-5 32 3 0.1 128
# CUDA_VISIBLE_DEVICES=1 python run_multi_cho.py --do_train --do_eval --do_lower_case --task_name hellaswag --num_train_epochs 3 --learning_rate 2e-5 --train_batch_size 32 --model_type bert --load_model_path bert-base-uncased --output_dir hellaswag_bert --fp16 --warmup_proportion 0.1 && \
# CUDA_VISIBLE_DEVICES=1 python run_sent_clas.py --do_train --do_eval --do_lower_case --task_name sst-2 --num_train_epochs 3 --learning_rate 2e-5 --train_batch_size 32 --model_type bert --load_model_path bert-base-uncased --output_dir sst-2_bert --fp16 --warmup_proportion 0.06 && \
# CUDA_VISIBLE_DEVICES=1 python run_multi_cho.py --do_train --do_eval --do_lower_case --task_name dream --num_train_epochs 8 --learning_rate 3e-5 --train_batch_size 16 --model_type bert --load_model_path bert-base-uncased --output_dir dream_bert --fp16 --warmup_proportion 0.1
# PAWS-QQP 5e-5 16 3 0.06 128 should get 81.7
# CUDA_VISIBLE_DEVICES=0 python run_sent_clas.py --do_train --do_eval --do_lower_case --task_name paws-qqp --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 16 --model_type bert --load_model_path bert-base-uncased --output_dir paws-qqp_bert --fp16 --warmup_proportion 0.06

# hans
CUDA_VISIBLE_DEVICES=0 python run_sent_clas.py --do_train --do_eval --do_lower_case --task_name hans --num_train_epochs 3 --learning_rate 2e-5 --train_batch_size 32 --model_type bert --load_model_path bert-base-uncased --output_dir hans_bert --fp16 --warmup_proportion 0.06
# CUDA_VISIBLE_DEVICES=$cuda python run_sent_clas.py --do_train --do_eval --do_lower_case --task_name qqp --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 32 --model_type bert --load_model_path bert-base-uncased --output_dir qqp_bert --fp16 --warmup_proportion 0.06 && \ 
# CUDA_VISIBLE_DEVICES=$cuda python run_sent_clas.py --do_train --do_eval --do_lower_case --task_name qnli --num_train_epochs 3 --learning_rate 2e-5 --train_batch_size 32 --model_type bert --load_model_path bert-base-uncased --output_dir qnli_bert --fp16 --warmup_proportion 0.06 
# && \
# CUDA_VISIBLE_DEVICES=$cuda python run_sent_clas.py --do_train --do_eval --do_lower_case --task_name mnli --num_train_epochs 3 --learning_rate 3e-5 --train_batch_size 32 --model_type bert --load_model_path bert-base-uncased --output_dir mnli_bert --fp16 --warmup_proportion 0.06

# Roberta
# CUDA_VISIBLE_DEVICES=0 python run_sent_clas.py --do_train --do_eval --do_lower_case --task_name sst-2 --num_train_epochs 8 --learning_rate 2e-5 --train_batch_size 16 --model_type roberta --load_model_path roberta-base --output_dir sst-2_roberta --fp16 --warmup_proportion 0.06