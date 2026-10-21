export TASK=bp

export NBS_WEAK_GRAPH_MANIFEST=/home/dataset-local/data_local/shaojiangyi/latence-project/outputs/latence_nbs/bp_weak_detr_v3_expert_prob_warmstart340_to400/epoch100/bp/weak_graph_predictions_full_task_top512/weak_graph_predictions_manifest.json
export NBS_EVAL_CANDIDATE_SELECTOR_SCOPE=full_task
export NBS_INPUT_CACHE_POLICY=refresh

export NBS_CHECKPOINT_DIR=outputs/latence_nbs_train/bp_nbs_v060_perfopt_tailfix_fullrestore_formal
# export NBS_EVAL_EPOCHS=1,2,3,4
# export NBS_EVAL_EPOCHS=5,6,7,8,9
export NBS_EVAL_EPOCHS=16,17,18,19,20
#export NBS_EVAL_SERIES_ROOT=outputs/latence_nbs_eval/bp_nbs_v060_perfopt_epoch1_4
# export NBS_EVAL_SERIES_ROOT=outputs/latence_nbs_eval/bp_nbs_v060_perfopt_epoch5_9
export NBS_EVAL_SERIES_ROOT=outputs/latence_nbs_eval/bp_nbs_v060_perfopt_epoch16_20

# 使用训练目录里保存的精确配置
export NBS_TRAIN_CONFIG="$NBS_CHECKPOINT_DIR/resolved_config.json"

# 必须换成第一阶段实际checkpoint路径
export STAGE1_CHECKPOINT=/home/dataset-local/data_local/shaojiangyi/latence-project/outputs/weak_exp_train_detr/bp_weak_detr_v3_expert_prob_warmstart340_to400/weak_detr_decoder_epoch100.pt

export METADATA_FILE=data/unidata_with_exp_train_pseudo.pkl
export STAGE1_MSA_INDEX=data/ind_MSA_bin/index.pkl

export NBS_EVAL_DEVICE=cuda:0
export STAGE1_EVAL_DEVICE=cuda:0

# 防止与仍在运行的训练争抢显存
export NBS_EVAL_MIN_FREE_GPU_GB=30
export STAGE1_EVAL_MIN_FREE_GPU_GB=30

# 正式归纳推理必须开启
export NBS_EVAL_USE_EXTERNAL_PP=1
export NBS_EVAL_USE_CANDIDATE_EVIDENCE=1
export NBS_SAVE_INFERENCE_DIAGNOSTICS=1

# 使用与第一阶段相同的正式指标实现
export NBS_METRIC_BACKEND=stage1

# 配对bootstrap与排序指标
export NBS_EVAL_BOOTSTRAP_REPLICATES=1000
export NBS_EVAL_BOOTSTRAP_SEED=6061
export NBS_EVAL_PRECISION_K=10,50,100

unset NBS_EVAL_LIMIT_PROTEINS
unset NBS_IND_TEST_PROB

python scripts/nbs/run_eval_nbs_checkpoint_series.py
