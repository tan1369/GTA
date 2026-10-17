# python ./run_avl_retrieval_only.py \
#   --dataset_root /mnt/d/xuqi/Benchmark/Data/AnyVisLoc \
#   --yaml ./config_selectable_matchers.yaml \
#   --save_dir ./Result/CAMP_rgb_all2_aerial_yp_u1652 \
#   --reference_mode aerial \
#   --retrieval_methods CAMP \
#   --retrieval_ks 1 3 5 \
#   --pose_priori yp \
#   --device cuda

  # --sample_ratio 0.1 \
  # --sample_seed 42 \

# python -u run_anyvisloc_retrieval.py \
#   --dataset_root /mnt/d/xuqi/Benchmark/Data/AnyVisLoc \
#   --model Sample4Geo \
#   --save_dir /mnt/d/xuqi/Benchmark/Result/Sample4Geo_smoke_test \
#   --reference_mode aerial \
#   --retrieval_ks 1 3 5 \
#   --pose_priori yp \
#   --scenes Scene_01 \
#   --limit 5 \
#   --device cuda \
#   --debug_raise


  # python -u run_anyvisloc_retrieval.py \
  # --dataset_root /mnt/d/xuqi/Benchmark/Data/AnyVisLoc \
  # --model DAC \
  # --save_dir /mnt/d/xuqi/Benchmark/Result/DAC_scene1_2_aerial_yp \
  # --reference_mode aerial \
  # --retrieval_ks 1 3 5 \
  # --pose_priori yp \
  # --scenes Scene_01 Scene_02 \
  # --device cuda \
  # --debug_raise

  # python -u run_anyvisloc_retrieval.py \
  # --dataset_root /mnt/d/xuqi/Benchmark/Data/AnyVisLoc \
  # --model Wave \
  # --save_dir /mnt/d/xuqi/Benchmark/Result/WAVE_scene1_smoke_test \
  # --reference_mode aerial \
  # --retrieval_ks 1 3 5 \
  # --pose_priori yp \
  # --scenes Scene_01 \
  # --device cuda \
  # --debug_raise

  # python -u run_anyvisloc_retrieval.py \
  # --dataset_root /mnt/d/xuqi/Benchmark/Data/AnyVisLoc \
  # --model Wave \
  # --save_dir ./Result/WAVE_scene1_same_aerial_yp \
  # --reference_mode aerial \
  # --retrieval_ks 1 3 5 \
  # --pose_priori yp \
  # --scenes Scene_01 \
  # --device cuda \
  # --debug_raise

    python -u run_anyvisloc_retrieval.py \
  --dataset_root /mnt/d/xuqi/Benchmark/Data/AnyVisLoc \
  --model GTA-UAV \
  --save_dir /mnt/d/xuqi/Benchmark/Result/GTA-UAV_10%_all_scene_yp \
  --reference_mode aerial \
  --retrieval_ks 1 3 5 \
  --pose_priori yp \
  --device cuda \
  --sample_ratio 0.1 \
  --sample_seed 42 \
  --debug_raise