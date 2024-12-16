_gpu=0
export CUDA_VISIBLE_DEVICES=$_gpu
python main.py \
    --method ogc_gce \
    --method_params \
        q 0.7 \
        epsilon 20.0 \
        time_frame_s 32 \
        queue_size_q 4096 \
        binary_search_tol 1e-7 \
        binary_search_max_iter 10 \
    --model resnet18 \
    --train_dataset cifar10pdn_train \
    --train_dataset_params \
        root /path/to/dataset/cifar10 \
        noise_rate 0.4 \
        batch_size 128 \
        num_workers 4 \
    --eval_dataset cifar10_test \
    --eval_dataset_params \
        root /path/to/dataset/cifar10 \
        batch_size 256 \
        num_workers 4 \
    --num_classes 10 \
    --total_epoch 150 \
    --optimizer sgd \
    --optimizer_params \
        lr 0.1 \
        momentum 0.9 \
        weight_decay 0.0005 \
        nesterov no \
    --scheduler multisteplr \
    --scheduler_params \
        milestones 50,100 \
        gamma 0.1 \
    --seed 1 \
    --gpu $_gpu \
    --eval_freq 1 \
    --log_freq 100 \
    --write_freq 100 \
    --save_model
sleep 1