from trainer import AtmoRepTrainer  # make sure this path is right

from itwinai.pipeline import Pipeline

my_pipeline = Pipeline(
    [
        AtmoRepTrainer(
            config={
                "load_model": None,  # ['dys79lgw', 82],
                "mode": "train",
                "path_models": "models",  # "/p/home/jusers/luise1/juwels/interTwin/itwinai_AtmoRep/itwinai/use-cases/atmorep/models/",
                "path_results": "results",  # "/p/home/jusers/luise1/juwels/interTwin/itwinai_AtmoRep/itwinai/use-cases/atmorep/results/",
                "path_data": "/p/scratch/intertwin/datasets/era5_y2021_res025.zarr/",  # "./data/era5_y1979_2021_res025_chunk8.zarr/",
                "file_path": "/p/scratch/intertwin/datasets/era5_y2021_res025.zarr",  # "/p/scratch/atmo-rep/data/era5_1deg/months/era5_y1979_2021_res025_chunk8.zarr",  # this will change in the new atmorep branch
                "fields": [
                    [
                        "velocity_u",
                        [1, 1024, [], 0],
                        [96, 105, 114, 123, 137],
                        [12, 3, 6],
                        [3, 18, 18],
                        [0.5, 0.9, 0.2, 0.05],
                    ]
                ],
                "fields_prediction": [["velocity_u", 1.0]],
                "fields_targets": [],
                # TODO: make a range
                "years_train": [2021],  # list( range( 1979, 2021))
                "years_val": [2021],
                "month": None,
                "geo_range_sampling": [[-90.0, 90.0], [0.0, 360.0]],
                "time_sampling": 1,  # sampling rate for time steps
                # random seeds
                "torch_seed": 0,
                # training params
                "batch_size_validation": 1,  # 64
                "batch_size": 96,
                "num_epochs": 128,
                "num_samples_per_epoch": 4096 * 12,
                "num_samples_validate": 128 * 12,
                "num_loader_workers": 6,
                # additional infos
                "size_token_info": 8,
                "size_token_info_net": 16,
                "grad_checkpointing": True,
                "with_cls": False,
                # network config
                "with_mixed_precision": True,
                "with_layernorm": True,
                "coupling_num_heads_per_field": 1,
                "dropout_rate": 0.05,
                "with_qk_lnorm": False,
                # encoder
                "encoder_num_layers": 6,
                "encoder_num_heads": 16,
                "encoder_num_mlp_layers": 2,
                "encoder_att_type": "dense",
                # decoder
                "decoder_num_layers": 6,
                "decoder_num_heads": 16,
                "decoder_num_mlp_layers": 2,
                "decoder_self_att": False,
                "decoder_cross_att_ratio": 0.5,
                "decoder_cross_att_rate": 1.0,
                "decoder_att_type": "dense",
                # tail net
                "net_tail_num_nets": 16,
                "net_tail_num_layers": 0,
                # loss: mse, mse_ensemble, stats, crps, weighted_mse
                "losses": ["mse_ensemble", "stats"],
                # training
                "optimizer_zero": False,
                "lr_start": 5.0 * 10e-7,
                "lr_max": 0.00005 * 3,
                "lr_min": 0.00004,  # 0.00002
                "weight_decay": 0.05,  # 0.1
                "lr_decay_rate": 1.025,
                "lr_start_epochs": 3,
                "model_log_frequency": 256,  # save checkpoint every X batches
                # BERT strategies: 'BERT', 'forecast', 'temporal_interpolation'
                "BERT_strategy": "BERT",
                "forecast_num_tokens": 2,  # only needed / used for BERT_strategy 'forecast
                "BERT_fields_synced": False,  # apply synchronized / identical masking to all fields
                # (fields need to have same BERT params for this to have effect)
                "BERT_mr_max": 2,  # maximum reduction rate for resolution
                # debug / output
                "log_test_num_ranks": 0,
                "save_grads": False,
                "profile": False,
                "attention": False,
                "rng_seed": None,
                # usually use %>wandb offline to switch to disable syncing with server
                "with_wandb": True,
                "n_size": 0,
            },
            strategy="ddp",
            epochs=128,
            random_seed=0,
            profiling_wait_epochs=0,
            profiling_warmup_epochs=0,
        )
    ]
)

my_pipeline.execute()
