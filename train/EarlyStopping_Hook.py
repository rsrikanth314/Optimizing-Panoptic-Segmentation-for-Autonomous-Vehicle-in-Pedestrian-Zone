from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Define a custom Trainer class that supports early stopping
class EarlyStoppingTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.early_stopping_patience = cfg.SOLVER.EARLY_STOPPING_PATIENCE
        self.best_metric = -float("inf")
        self.wait = 0  # Counter for how long to wait for improvement

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, self.EarlyStoppingHook(self.early_stopping_patience))
        return hooks

    class EarlyStoppingHook:
        def __init__(self, patience):
            self.patience = patience

        def before_train(self, trainer):
            pass

        def after_step(self, trainer):
            pass

        def before_validate(self, trainer):
            pass

        def after_epoch(self, trainer):
            # Perform validation and check the validation metric
            if trainer.epoch % trainer.cfg.TEST.EVAL_PERIOD == 0:
                metrics = inference_on_dataset(trainer.model, build_detection_test_loader(trainer.cfg, trainer.cfg.DATASETS.TEST[0]))
                validation_metric = metrics[trainer.cfg.TEST.VAL_METRIC]
                print(f"Validation {trainer.cfg.TEST.VAL_METRIC}: {validation_metric}")

                # Check if the validation metric has improved
                if validation_metric > trainer.best_metric:
                    trainer.best_metric = validation_metric
                    trainer.wait = 0
                    # Save the checkpoint when the metric improves
                    trainer.checkpointer.save("best_model")
                else:
                    trainer.wait += 1

                # Check if we've exceeded the patience limit
                if trainer.wait >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    trainer.max_iter = trainer.iter
                    trainer.stop = True
