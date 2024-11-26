from itwinai.parser import ConfigParser

if __name__ == "__main__":

    parser = ConfigParser(
        config="config_raytrainer.yaml"
    )
    my_pipeline = parser.parse_pipeline(
        pipeline_nested_key="training_pipeline_small",
        verbose=False
    )
    my_pipeline.execute()

    # data_root = "./data"  # Set your data root
    # batch_size = 64  # Example: you can set it dynamically
    # learning_rate = 1e-4  # Example: can be tuned dynamically
    # strategy = "ddp"  # or "deepspeed", can be set dynamically as well

    # my_pipeline = Pipeline(
    #     steps=[
    #         TimeSeriesDatasetGenerator(data_root=data_root),
    #         TimeSeriesDatasetSplitterSmall(
    #             train_proportion=0.9, validation_proportion=0.1, rnd_seed=42),
    #         TimeSeriesProcessorSmall(),
    #         NoiseGeneratorTrainer(
    #             strategy=strategy,
    #             config={
    #                 "scaling_config": {
    #                     "num_workers": 2,
    #                     "use_gpu": True,
    #                     "resources_per_worker": {"CPU": 5, "GPU": 1}
    #                 },
    #                 "tune_config": {
    #                     "num_samples": 2,
    #                     "scheduler": {
    #                         "name": "asha",
    #                         "max_t": 10,
    #                         "grace_period": 5,
    #                         "reduction_factor": 4,
    #                         "brackets": 1
    #                     },
    #                 },
    #                 "run_config": {
    #                     "storage_path": "ray_checkpoints",
    #                     "name": "Virgo-HPO-Experiment"
    #                 },
    #                 "train_loop_config": {
    #                     "batch_size": {
    #                         "type": "choice",
    #                         "options": [32, 64, 128]
    #                     },
    #                     "learning_rate": {
    #                         "type": "uniform",
    #                         "min": 1e-5,
    #                         "max": 1e-3
    #                     },
    #                     "epochs": 10,
    #                     "generator": "simple",  # or 'unet'
    #                     "loss": "L1",
    #                     "save_best": False,
    #                     "shuffle_train": True,
    #                     "random_seed": 17
    #                 }
    #             }
    #         )
    #     ]
    # )
