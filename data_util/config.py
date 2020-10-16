import os

root_dir = os.path.expanduser("~")

#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "/home/kxiao/kangjia/works_ex3/pointer_train_data.txt")
eval_data_path = os.path.join(root_dir, "/home/kxiao/kangjia/works_ex3/pointer_test_data.txt")
decode_data_path = os.path.join(root_dir, "/home/kxiao/kangjia/works_ex3/pointer_test_data.txt")
vocab_path = os.path.join(root_dir, "/home/kxiao/kangjia/works_ex3/pointerVocab")
log_root = os.path.join(root_dir, "/home/kxiao/pointer_generator_pytorch/log")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 32
max_enc_steps=100
max_dec_steps=30
beam_size=4
min_dec_steps=1
vocab_size=2600

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15
