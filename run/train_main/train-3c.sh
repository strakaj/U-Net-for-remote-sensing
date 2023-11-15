ROOT_DIR="../../"
CONFIG_FILE="flair-2-config.yml"

PROJECT_NAME="FLAIR-two"
export WANDB_API_KEY=
export CUDA_VISIBLE_DEVICES=0

NAME=03c
python "${ROOT_DIR}main.py"  \
	--config_file "${ROOT_DIR}${CONFIG_FILE}"  \
	--out_model_name $NAME \
	--name $NAME \
	--out_folder "${ROOT_DIR}out" \
	--project "PROJECT_NAME" \
	--num_epochs 6 \
	--batch_size 10 \
	--optimizer "adamw" \
	--average_month \
	--filter_clouds \
	--use_augmentation \
	--seed 2022 \
	--encoder_name "resnet34" \
	--loss_name "focalS2" \
	--scheduler "multi_0" \
	--fold 2 \
	--val_percent 0.9 \
	--unet_checkpoint "${ROOT_DIR}out/00c/checkpoints/last.ckpt" \
	--lr 0.0001 > "${ROOT_DIR}out/${NAME}.txt"
