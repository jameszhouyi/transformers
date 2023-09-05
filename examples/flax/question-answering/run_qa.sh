pip install -r ./requirements.txt
export ZE_AFFINITY_MASK=0.0
python run_qa.py   \
--model_name_or_path bert-large-uncased-whole-word-masking   \
--dataset_name squad   \
--do_train   \
--do_eval   \
--per_device_train_batch_size 6   \
--learning_rate 3e-5   \
--num_train_epochs 2   \
--max_seq_length 384   \
--doc_stride 128   \
--output_dir ./wwm_uncased_finetuned_squad/ \
--eval_steps 1000
