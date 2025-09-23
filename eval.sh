export PYTHONPATH=${PWD}:$PYTHONPATH
export PROJECT_DIR=${PWD}
python ManiFlow/eval_robogen.py \
    --low_level_exp_dir  ckpt/ManiFlow_open/ \
    --low_level_ckpt_name epoch-90.ckpt \
    --eval_exp_name 0920_test \
    --exp_dir ${1}${2} \
    --invert 