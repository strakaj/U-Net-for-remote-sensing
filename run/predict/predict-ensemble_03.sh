ROOT_DIR=""
CONFIG_FILE="flair-2-config.yml"

export CUDA_VISIBLE_DEVICES=0

NAME=ensemble_03
python "${ROOT_DIR}predict_ensemble_lightning.py" \
	--config_file "${ROOT_DIR}${CONFIG_FILE}" \
	--num_workers 10 \
	--out_model_name $NAME \
	--average_month \
	--filter_clouds \
	--encoder_names  "resnet34" "mit_b2" "resnext50_32x4d"  \
	--output_from "logits" \
	--out_folder "${ROOT_DIR}out" \
	--checkpoint_paths  \
	"${ROOT_DIR}weights/main/fin-03c.ckpt" \
	"${ROOT_DIR}weights/main/fin-04b.ckpt" \
	"${ROOT_DIR}weights/main/fin-05c.ckpt" > "${ROOT_DIR}out/${NAME}.txt"
	