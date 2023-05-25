python3 -m openpifpaf.predict \
  --checkpoint outputs/pose_overfit.epoch100 \
  --debug-indices cif:5 caf:5  --long-edge=385 --loader-workers=0 \
  --save-all --image-output all-images/overfitted.jpeg \
  /work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training/segment-15628918650068847391_8077_670_8097_670_with_camera_labels/150888933886058500.jpg
  