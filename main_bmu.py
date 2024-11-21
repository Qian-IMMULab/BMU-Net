import os
from datetime import datetime
import evaluate
from mmeval import Accuracy
import pytz
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from objprint import objstr
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import optim_factory

from src import utils
from src.data.loader_bmu import get_dataloader
from src.model.bmunet import bmu_load_pretrained_model, BMUNet
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, MetricSaver


def train_one_epoch(
        model: torch.nn.Module,
        loss_function: torch.nn.modules.loss._Loss,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        accelerator: Accelerator,
        evaluator: evaluate.EvaluationModule,
        epoch: int,
        step: int,
):
    # Train
    model.train()
    for i, (mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info, label) in enumerate(
            train_loader
    ):

        logist = model(mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info=clinic_info)
        loss_category = loss_function(logist, label)
        evaluator.add(logist.argmax(dim=-1), label)
        accelerator.log(
            {
                "Train loss": float(loss_category),
            },
            step=step,
        )

        accelerator.backward(loss_category)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.bmu.num_epochs}] Training [{i}/{len(train_loader)}] Train Loss1: {loss_category:1.5f} "
        )
    evaluate_result = evaluator.compute()

    accelerator.log(
        {
            "Train acc": float(evaluate_result["top1"]),
        },
        step=epoch,
    )
    scheduler.step(epoch)
    return step


@torch.no_grad()
def val_one_epoch(
        model: torch.nn.Module,
        loss_function: torch.nn.modules.loss._Loss,
        val_loader: torch.utils.data.DataLoader,
        config: EasyDict,
        accelerator: Accelerator,
        evaluator: evaluate.EvaluationModule,
        epoch,
        step,
):
    # Eval
    model.eval()
    for i, (mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info, label) in enumerate(
            val_loader
    ):
        logist = model(mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info=clinic_info)
        loss = loss_function(logist, label)
        evaluator.add(logist.argmax(dim=-1), label)
        accelerator.log(
            {
                "Val Loss": float(loss),
            },
            step=step,
        )
        accelerator.log({"lr": float(scheduler.get_last_lr()[0])}, step=step)
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.bmu.num_epochs}] Validation [{i}/{len(val_loader)}] Val Loss: {loss:1.5f}"
        )
        step += 1

    evaluate_result = evaluator.compute()

    accelerator.log(
        {
            "Val acc": float(evaluate_result["top1"]),
        },
        step=epoch,
    )
    return float(evaluate_result["top1"]), step


if __name__ == "__main__":
    # Load config
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(42)
    logging_dir = (
        os.path.join(os.getcwd(), 'logs', 'bmu', str(
            datetime.now(tz=pytz.timezone("Asia/Shanghai")).strftime(
                "%Y-%m-%d-%H-%M-%S"
            ))
        )
    )
    accelerator = Accelerator(log_with=["tensorboard"], project_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers("bmu")
    accelerator.print(objstr(config))

    # Init model
    accelerator.print("Loading model...")
    model = BMUNet()

    # Load mg and us module pretrained weight
    model = bmu_load_pretrained_model(
        model, config.bmu.mg_model, config.bmu.us_model
    )
    model.finetune_conv()

    # Load dataset
    accelerator.print("Loading dataset...")
    train_loader, val_loader = get_dataloader(config)

    # Define train optimizer
    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=config.trainer.weight_decay,
        lr=config.bmu.lr,
        betas=(0.9, 0.95),
    )
    # Define learning rate update strategy
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.bmu.num_epochs,
    )
    # Define loss function
    loss_function = LabelSmoothingCrossEntropy()
    evaluator = Accuracy()

    step = 0
    val_step = 0
    starting_epoch = 0
    metric_saver = MetricSaver()

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )
    accelerator.register_for_checkpointing(metric_saver)
    metric_saver.to(accelerator.device)

    # Try to continue training
    if config.trainer.resume:
        starting_epoch, step, val_step = utils.resume_train_state(
            config.bmu.save_dir, train_loader, val_loader, accelerator
        )

    # Start training
    accelerator.print("Start training！")

    for epoch in range(starting_epoch, config.bmu.num_epochs):
        # Train one epoch
        step = train_one_epoch(
            model,
            loss_function,
            train_loader,
            optimizer,
            scheduler,
            accelerator,
            evaluator,
            epoch,
            step,
        )
        # Eval one epoch
        acc, val_step = val_one_epoch(
            model,
            loss_function,
            val_loader,
            config,
            accelerator,
            evaluator,
            epoch,
            val_step,
        )

        accelerator.print(
            f"Epoch [{epoch + 1}/{config.bmu.num_epochs}] lr = {scheduler.get_last_lr()},  acc = {acc} %"
        )

        metric_saver.current_acc.data = torch.Tensor([acc]).to(accelerator.device)
        metric_saver.epoch.data = torch.Tensor([epoch]).to(accelerator.device)
        # Save model
        if acc > metric_saver.best_acc:
            metric_saver.best_acc.data = torch.Tensor([acc]).to(accelerator.device)
            accelerator.save_state(
                output_dir=os.path.join(os.getcwd(), config.bmu.save_dir, 'best')
            )

        accelerator.save_state(
            output_dir=os.path.join(os.getcwd(), config.bmu.save_dir, f"epoch_{epoch}")
        )

    accelerator.print(f"Best acc: {metric_saver.best_acc}")
