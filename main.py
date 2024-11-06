from itwinai.loggers import Prov4MLLogger, LoggersCollection, MLFlowLogger
import os
import torch
LOGS_DIR = 'logs_prov'

if __name__ == "__main__":
    run_name = 'test_run'
    logger = LoggersCollection([
        Prov4MLLogger(
            provenance_save_dir=os.path.join(LOGS_DIR, "ITWINAI", "provenance"),
            experiment_name=run_name,
            save_after_n_logs=1,
            create_graph=False,
            create_svg=False,
        ),
        MLFlowLogger()
    ])

    model = torch.nn.Linear(100, 100)

    with logger.start_logging():

        logger.save_hyperparameters(dict(batch_size=32, lr=1e-3))

        for current_epoch in range(10):
            context = 'validation'
            logger.log(item=current_epoch, identifier="epoch",
                       kind='metric', step=current_epoch, context=context)
            logger.log(item=model, identifier=f"model_version_{current_epoch}",
                       kind='model_version', step=current_epoch, context=context)
            logger.log(item=None, identifier=None, kind='system',
                       step=current_epoch, context=context)
            logger.log(item=None, identifier=None, kind='carbon',
                       step=current_epoch, context=context)
            logger.log(item=None, identifier="train_epoch_time",
                       kind='execution_time', step=current_epoch, context=context)

        logger.log(item=None, identifier=None, kind="prov_documents")
