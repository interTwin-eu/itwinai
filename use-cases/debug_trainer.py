from trainer import AtmoRepTrainer  # make sure this path is right

from itwinai.pipeline import Pipeline

my_pipeline = Pipeline(
    [
        AtmoRepTrainer(
            config={
                "path_models": "../../models/",
                "path_results": "../../results/",
                "path_plots": "../results/plots/",
                "path_data": "./data/era5_y1979_2021_res025_chunk8.zarr/",
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
