python "${WORK_DIR}"/export_saved_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_DIR}_saved" \
  --model_variant="${MODEL_NAME}" \
  --output_stride=16 \
  --num_classes=15 \
  --crop_size=513 \
  --crop_size=513 \
  --img_channels=1 \
  --inference_scales=1.0 \
  --inference_scales=0.75 \
  --inference_scales=1.25 \