Mario Rüttgers, 08.07.2024

This file explains details about the parameters that can be changed by the user in the tutorial for hyperparameter optimization (HPO)

1. SLURM JOBSCRIPT

# general configuration of the job
> HDFML has 16 CPUs per node, this might be different for other systems 
> Each node launches one task with all GPUs ('--gpus-per-node=4', '--gres=gpu:4') and CPUs ('--cpus-per-task=16')
> Ray takes care to distribute the resources internally

# command
> '--num-samples 6' stands for the number of HPO configurations that will be investigated
> '--ngpus 1' is the number of GPUs that are used to train one configuration
> '--max-iterations 2' is the maximum number of epochs for each configuration
> '--scheduler ASHA' is one of three popular schedulers. 
     For more details about code for the other schedulers see: https://gitlab.jsc.fz-juelich.de/CoE-RAISE/FZJ/ai4hpc/ai4hpc/-/tree/master/HPO/Cases?ref_type=heads

2. FILE hpo.py

Unfortunately, in the following lines 'num_workers' is set manually. Ideally, this should be changed to '--cpus-per-task' / '--gpus-per-node'. 

   train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4)

   test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=4)

The same goes for the follwing line:

   scaling_config=ScalingConfig(num_workers=args.ngpus, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1}),

Here, I would automate:

   "CPU" --> '--cpus-per-task' / '--gpus-per-node'
   "GPU" --> '--ngpus'

3. HPO

Depending on the parameters explained above, the user should also define the type and range of hyperparameters, and the default configuration:

    # define the hyperparameter search space 
    config = {
        "batch_size": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(10e-5, 1),
        "data_dir": tune.choice([args.data_dir]),
    }


    # default hyperparameters for the function
    train_loop_config={"batch_size": 64, "lr": 0.1, "data_dir": "/"},

Furthermore, it is important to specify the metric which is used to evaluate a trained configuration. Here, the test accuracy is chosen as the metric:

   # define which metric to use for measuring the performance of the trials
   metric="test_acc",
   # if the metric should be maximized or minimized 
   mode="max",

If all parameters are set, Ray runs the optimization and, finally, the hyperparameters of the configuration with the best metric are printed.
