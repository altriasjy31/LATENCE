from os.path import join, dirname
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(root_dir)

settings_dict = {
    'root_dir': root_dir,
    'DATA_DIR': join(root_dir, 'data'),
    'ia_file': join(root_dir, "esm_models", 'IA.txt'),
    'ia_script': join(root_dir, "esm_models", 'ia.py'),
    'obo_file': join(root_dir, 'go-basic.obo'),
    'obo_pkl_file': join(root_dir, 'obo.pkl'),

    'diamond_path': join(root_dir, 'utils', 'diamond'),

    'esm3b_path': join(proj_dir, 'data', 'esm_models', 'esm2_t36_3B_UR50D.pt'),

    'MODEL_CHECKPOINT_DIR': join(root_dir, 'models', 'ZLPR_PTF1_GOF1'),
    'LOGS_DIR': join(root_dir, 'logs'),
    'SUBMISSION_DIR': join(root_dir, 'submissions')
}

training_config = {
    'activation':'gelu',
    'layer_list':[2048],
    'embed_dim':2560,
    'dropout':0.3,
    'epochs':200,
    'batch_size':512,
    'pred_batch_size':8124*4,
    'learning_rate':0.001,
    'num_models': 5,
    'patience':10,
    'min_epochs':20,
    'seed':12,
    'repr_layers': [34, 35, 36],
    'log_interval':1,
    'eval_interval':1,
    'monitor': 'both',
}

add_res_dict = {
    'BPO':True,
    'CCO':False,
    'MFO':False,
} # whether to add residual connections or not