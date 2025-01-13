#!/bin/bash

source .venv/bin/activate

model_name="test-run"
checkpoint_location="checkpoints"

python3 train.py --id "${model_name}" \
	--caption_model "simple_transformer" \
	--input_json "/home/henry/Datasets/coco/cocotalk.json" \
	--input_fc_dir "/home/henry/Datasets/coco/superpixel_features/CLIP/whole_img/" \
	--input_att_dir "/home/henry/Datasets/coco/superpixel_features/CLIP/SLIC/m25/" \
	--input_box_dir "/home/henry/Datasets/coco/butd_box/" \
	--input_label_h5 "/home/henry/Datasets/coco/cocotalk_label.h5" \
	--checkpoint_path "${checkpoint_location}" \
	--fc_feat_size 512 \
	--att_feat_size 512 \
	--rnn_size 512 \
	--input_encoding_size 512 \
	--noamopt \
	--noamopt_warmup 10000 \
	--label_smoothing 0.0 \
	--batch_size 8 \
	--learning_rate 5e-4 \
	--num_layers 6 \
	--learning_rate_decay_start 0 \
	--scheduled_sampling_start 0 \
	--save_checkpoint_every 10000 \
	--language_eval 1 \
	--val_images_use 5000 \
	--max_epochs 30 \
	--seed -1 \
	--cached_tokens "/home/henry/Datasets/coco/coco-train-idxs" \

# Copy model for SCST
bash scripts/copy_model.sh "${checkpoint_location}" "${model_name}" "${model_name}_rl"

# Train SCST
python train.py --id "${model_name}_rl" \
	--caption_model "simple_transformer" \
	--input_json "/home/henry/Datasets/coco/cocotalk.json" \
	--input_fc_dir "/home/henry/Datasets/coco/superpixel_features/CLIP/whole_img/" \
	--input_att_dir "/home/henry/Datasets/coco/superpixel_features/CLIP/SLIC/m25/" \
	--input_box_dir "/home/henry/Datasets/coco/butd_box/" \
	--input_label_h5 "/home/henry/Datasets/coco/cocotalk_label.h5" \
	--checkpoint_path "${checkpoint_location}" \
	--fc_feat_size 512 \
	--att_feat_size 512 \
	--rnn_size 512 \
	--input_encoding_size 512 \
	--label_smoothing 0.0 \
	--batch_size 10 \
	--learning_rate 5e-4 \
	--num_layers 6 \
	--learning_rate_decay_start 0 \
	--scheduled_sampling_start 0 \
	--start_from "${checkpoint_location}" \
	--save_checkpoint_every 6000 \
	--language_eval 1 \
	--val_images_use 5000 \
	--self_critical_after 30 \
	--max_epochs 60 \
	--cached_tokens "/home/henry/Datasets/coco/coco-train-idxs" \


# Eval SCST
python eval.py --dump_images 0 \
	--num_images 5000 \
	--model "${checkpoint_location}/model-${model_name}_rl.pth" \
	--infos_path "${checkpoint_location}/infos_${model_name}_rl-best.pkl" \
	--image_root /home/henry/Datasets/coco/img/ \
	--input_json "/home/henry/Datasets/coco/cocotalk.json" \
	--input_fc_dir "/home/henry/Datasets/coco/superpixel_features/CLIP/whole_img/" \
	--input_att_dir "/home/henry/Datasets/coco/superpixel_features/CLIP/SLIC/m25/" \
	--input_box_dir "/home/henry/Datasets/coco/butd_box/" \
	--input_label_h5 "/home/henry/Datasets/coco/cocotalk_label.h5" \
	--language_eval 1 \
