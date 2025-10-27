export PYTHONPATH=${PWD}:$PYTHONPATH
export PROJECT_DIR=${PWD}
python ManiFlow/eval_robogen.py \
    --low_level_exp_dir  /data/robogen/sim_chenyuan/ManiFlow_Policy/ManiFlow/data/outputs/robogen_open_door-dit_pointcloud_policy_act3d-test_seed0 \
    --low_level_ckpt_name latest.ckpt \
    --eval_exp_name 1022_test \
    --exp_dir /data/robogen/sim_chenyuan/ManiFlow_Policy/data/foldingchair/100520 \
    --invert 