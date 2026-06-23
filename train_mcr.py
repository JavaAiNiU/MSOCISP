import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from datasets.MCRDataset import MCRDataset
from Losses.color_loss import ColorHistogramKLLoss
from Losses.vgg_loss import VGGPerceptualLoss
from Losses.wavelet_loss import CombinedLoss
from models.model_MSO_CCA_RAWLoss import MGCC
from utils.metrics import get_psnr_torch, get_ssim_torch


def count_model_params(model):
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def init_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train model_MSO_CCA_RAWLoss.")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        local_rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = 0

    return distributed, local_rank, device, rank == 0


def barrier(distributed):
    if distributed:
        dist.barrier()


def reduce_epoch_values(values, count, device, distributed):
    packed = torch.tensor([*values, count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    total_count = packed[-1].item()
    if total_count == 0:
        raise RuntimeError("The data loader produced no batches.")
    return (packed[:-1] / total_count).tolist()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def remove_pth_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            os.remove(os.path.join(directory, filename))


def parse_checkpoint_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return float(parts[-2]), int(parts[-1])
    except ValueError:
        return None


def list_checkpoints(directory):
    checkpoints = []
    if not os.path.isdir(directory):
        return checkpoints

    for filename in os.listdir(directory):
        if not filename.endswith(".pth"):
            continue
        path = os.path.join(directory, filename)
        parsed = parse_checkpoint_name(path)
        if parsed is not None:
            value, epoch = parsed
            checkpoints.append((path, value, epoch))
    return checkpoints


def best_saved_value(directory, mode, default):
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        return default
    values = [item[1] for item in checkpoints]
    return min(values) if mode == "min" else max(values)


def load_latest_checkpoint(model, directory, device):
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        return 1, None

    path, _, epoch = max(checkpoints, key=lambda item: item[2])
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return epoch + 1, path


def save_single_model(model, directory, filename):
    remove_pth_files(directory)
    path = os.path.join(directory, filename)
    torch.save(unwrap_model(model).state_dict(), path)
    return path


def compute_losses(prediction, target, losses):
    criterion_l1, criterion_wave, criterion_color, criterion_vgg = losses

    loss_l1 = criterion_l1(prediction, target)
    loss_wave = criterion_wave(prediction, target)
    loss_color = criterion_color(prediction.float(), target.float())
    loss_vgg = criterion_vgg(prediction, target)
    if loss_vgg.ndim > 0:
        loss_vgg = loss_vgg.mean()

    # total_loss = loss_l1 + loss_wave + 2.0 * loss_color + loss_vgg
    total_loss = 0.53*loss_l1 + 0.28*loss_wave + 0.09*loss_color + 0.1*loss_vgg
    return total_loss, loss_l1, loss_wave, loss_color, loss_vgg


def create_output_directories(result_dir, is_main_process, distributed):
    directories = {
        "root": result_dir,
        "best_loss": os.path.join(result_dir, "best_loss_model"),
        "last": os.path.join(result_dir, "last_model"),
        "best_psnr": os.path.join(result_dir, "best_psnr_model"),
        "best_ssim": os.path.join(result_dir, "best_ssim_model"),
        "best_val_loss": os.path.join(result_dir, "best_val_loss_model"),
        "best_val_psnr": os.path.join(result_dir, "best_val_psnr_model"),
        "best_val_ssim": os.path.join(result_dir, "best_val_ssim_model"),
        "logs": os.path.join(result_dir, "logs"),
        "tensorboard": os.path.join(result_dir, "logs_tensorboard"),
    }

    if is_main_process:
        for directory in directories.values():
            os.makedirs(directory, exist_ok=True)
    barrier(distributed)
    return directories


def build_loaders(args, distributed):
    patch_size = args.patch_size if args.patch_size > 0 else False
    train_dataset = MCRDataset(
        data_dir=args.data_dir,
        image_list_file=args.train_list_file,
        patch_size=False,
        split="train",
    )
    test_dataset = MCRDataset(
        data_dir=args.data_dir,
        image_list_file=args.test_list_file,
        patch_size=False,
        split="test",
        transpose=False,
        h_flip=False,
        v_flip=False,
    )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if distributed else None
    )
    test_sampler = (
        DistributedSampler(test_dataset, shuffle=False) if distributed else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_sampler


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    losses,
    device,
    epoch,
    is_main_process,
    distributed,
):
    model.train()
    totals = [0.0] * 7
    count = 0
    progress = tqdm(
        loader,
        total=len(loader),
        dynamic_ncols=True,
        ascii=True,
        desc=f"Epoch {epoch:04d} [train]",
        disable=not is_main_process,
    )

    for batch in progress:
        input_raw = batch["input_raw"].to(device, non_blocking=True)
        gt_rgb = batch["gt_rgb"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            prediction, _ = model(input_raw)
            loss_values = compute_losses(prediction, gt_rgb, losses)
            total_loss, loss_l1, loss_wave, loss_color, loss_vgg = loss_values

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        prediction_metric = torch.clip(
            prediction.detach().float() * 255.0, 0.0, 255.0
        )
        target_metric = torch.clip(gt_rgb.detach().float() * 255.0, 0.0, 255.0)
        psnr = get_psnr_torch(prediction_metric, target_metric).mean().item()
        ssim = get_ssim_torch(prediction_metric, target_metric).mean().item()

        batch_values = [
            total_loss.item(),
            psnr,
            ssim,
            loss_l1.item(),
            loss_wave.item(),
            loss_color.item(),
            loss_vgg.item(),
        ]
        totals = [total + value for total, value in zip(totals, batch_values)]
        count += 1

        if is_main_process:
            progress.set_description(
                f"Epoch {epoch:04d} [Loss:{batch_values[0]:.3f} "
                f"L1:{batch_values[3]:.3f} Wave:{batch_values[4]:.3f} "
                f"Color:{batch_values[5]:.3f} VGG:{batch_values[6]:.3f}]"
            )

    return reduce_epoch_values(totals, count, device, distributed)


@torch.no_grad()
def validate(
    model,
    loader,
    losses,
    device,
    epoch,
    is_main_process,
    distributed,
):
    model.eval()
    totals = [0.0, 0.0, 0.0]
    count = 0
    progress = tqdm(
        loader,
        total=len(loader),
        dynamic_ncols=True,
        ascii=True,
        desc=f"Epoch {epoch:04d} [validation]",
        disable=not is_main_process,
    )

    for batch in progress:
        input_raw = batch["input_raw"].to(device, non_blocking=True)
        gt_rgb = batch["gt_rgb"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            prediction, _ = model(input_raw)
            total_loss, _, _, _, _ = compute_losses(prediction, gt_rgb, losses)

        prediction_metric = torch.clip(
            prediction.detach().float() * 255.0, 0.0, 255.0
        )
        target_metric = torch.clip(gt_rgb.detach().float() * 255.0, 0.0, 255.0)
        psnr = get_psnr_torch(prediction_metric, target_metric).mean().item()
        ssim = get_ssim_torch(prediction_metric, target_metric).mean().item()

        totals[0] += total_loss.item()
        totals[1] += psnr
        totals[2] += ssim
        count += 1

    return reduce_epoch_values(totals, count, device, distributed)


def train_and_evaluate(args):
    distributed, local_rank, device, is_main_process = init_runtime()
    directories = create_output_directories(
        args.result_dir, is_main_process, distributed
    )

    writer = None
    try:
        if is_main_process:
            logging.basicConfig(
                filename=os.path.join(directories["logs"], "train.log"),
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s: %(message)s",
            )

        train_loader, test_loader, train_sampler = build_loaders(args, distributed)
        model = MGCC().to(device)

        if is_main_process:
            parameter_count = count_model_params(model)
            print(f"Model parameters: {parameter_count:.2f} M")
            logging.info("Model parameters: %.2f M", parameter_count)

        start_epoch = 1
        loaded_checkpoint = None
        if args.resume:
            try:
                start_epoch, loaded_checkpoint = load_latest_checkpoint(
                    model, directories["last"], device
                )
            except (RuntimeError, OSError, ValueError) as error:
                if is_main_process:
                    print(f"Could not resume training: {error}")
                    logging.warning("Could not resume training: %s", error)
                start_epoch = 1

        if is_main_process:
            if loaded_checkpoint is not None:
                print(f"Resumed from {loaded_checkpoint}; next epoch: {start_epoch}")
                logging.info(
                    "Resumed from %s; next epoch: %d",
                    loaded_checkpoint,
                    start_epoch,
                )
            else:
                print("Starting training from scratch.")
            writer = SummaryWriter(directories["tensorboard"], purge_step=start_epoch)

        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        losses = (
            torch.nn.L1Loss().to(device),
            CombinedLoss().to(device),
            ColorHistogramKLLoss().to(device),
            VGGPerceptualLoss().to(device),
        )
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for param_group in optimizer.param_groups:
            param_group.setdefault("initial_lr", args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.cosine_t_max,
            eta_min=args.cosine_eta_min,
            last_epoch=start_epoch - 2,
        )
        scaler = torch.amp.GradScaler("cuda")

        min_loss = best_saved_value(directories["best_loss"], "min", float("inf"))
        max_psnr = best_saved_value(directories["best_psnr"], "max", 0.0)
        max_ssim = best_saved_value(directories["best_ssim"], "max", 0.0)
        val_min_loss = best_saved_value(
            directories["best_val_loss"], "min", float("inf")
        )
        val_max_psnr = best_saved_value(
            directories["best_val_psnr"], "max", 0.0
        )
        val_max_ssim = best_saved_value(
            directories["best_val_ssim"], "max", 0.0
        )

        for epoch in range(start_epoch, args.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                losses,
                device,
                epoch,
                is_main_process,
                distributed,
            )
            (
                train_loss,
                train_psnr,
                train_ssim,
                train_l1,
                train_wave,
                train_color,
                train_vgg,
            ) = train_metrics

            if is_main_process:
                elapsed = time.time() - epoch_start
                print(
                    f"\nEpoch {epoch}: Loss={train_loss:.4f} | "
                    f"L1={train_l1:.4f} | Wave={train_wave:.4f} | "
                    f"Color*2={2.0 * train_color:.4f} | VGG={train_vgg:.4f}"
                )
                print(
                    f"PSNR={train_psnr:.4f} | SSIM={train_ssim:.4f} | "
                    f"Time={elapsed:.2f}s | LR={optimizer.param_groups[0]['lr']:.3e}"
                )
                logging.info(
                    "Epoch %d - Loss: %.4f, L1: %.4f, Wave: %.4f, "
                    "Color: %.4f, VGG: %.4f, PSNR: %.4f, SSIM: %.4f, "
                    "Time: %.2fs, LR: %.3e",
                    epoch,
                    train_loss,
                    train_l1,
                    train_wave,
                    train_color,
                    train_vgg,
                    train_psnr,
                    train_ssim,
                    elapsed,
                    optimizer.param_groups[0]["lr"],
                )
                writer.add_scalar("train_loss/total", train_loss, epoch)
                writer.add_scalar("train_loss/L1", train_l1, epoch)
                writer.add_scalar("train_loss/wavelet", train_wave, epoch)
                writer.add_scalar("train_loss/color", train_color, epoch)
                writer.add_scalar("train_loss/vgg", train_vgg, epoch)
                writer.add_scalar("train_PSNR", train_psnr, epoch)
                writer.add_scalar("train_SSIM", train_ssim, epoch)
                writer.add_scalar(
                    "learning_rate", optimizer.param_groups[0]["lr"], epoch
                )

                if train_loss < min_loss:
                    min_loss = train_loss
                    save_single_model(
                        model,
                        directories["best_loss"],
                        f"bestmodel_{train_loss:.4f}_{epoch}.pth",
                    )
                if train_psnr > max_psnr:
                    max_psnr = train_psnr
                    save_single_model(
                        model,
                        directories["best_psnr"],
                        f"bestpsnrmodel_{train_psnr:.4f}_{epoch}.pth",
                    )
                if train_ssim > max_ssim:
                    max_ssim = train_ssim
                    save_single_model(
                        model,
                        directories["best_ssim"],
                        f"bestssimmodel_{train_ssim:.4f}_{epoch}.pth",
                    )
                save_single_model(
                    model,
                    directories["last"],
                    f"ModelSnapshot_{train_loss:.4f}_{epoch}.pth",
                )

            barrier(distributed)

            if epoch % args.val_freq == 0:
                val_loss, val_psnr, val_ssim = validate(
                    model,
                    test_loader,
                    losses,
                    device,
                    epoch,
                    is_main_process,
                    distributed,
                )

                if is_main_process:
                    print(
                        f"Validation: Loss={val_loss:.4f} | "
                        f"PSNR={val_psnr:.4f} | SSIM={val_ssim:.4f}"
                    )
                    logging.info(
                        "Epoch %d validation - Loss: %.4f, PSNR: %.4f, SSIM: %.4f",
                        epoch,
                        val_loss,
                        val_psnr,
                        val_ssim,
                    )
                    writer.add_scalar("val_loss/total", val_loss, epoch)
                    writer.add_scalar("val_PSNR", val_psnr, epoch)
                    writer.add_scalar("val_SSIM", val_ssim, epoch)

                    if val_loss < val_min_loss:
                        val_min_loss = val_loss
                        save_single_model(
                            model,
                            directories["best_val_loss"],
                            f"bestvalloss_{val_loss:.4f}_{epoch}.pth",
                        )
                    if val_psnr > val_max_psnr:
                        val_max_psnr = val_psnr
                        save_single_model(
                            model,
                            directories["best_val_psnr"],
                            f"bestvalpsnr_{val_psnr:.4f}_{epoch}.pth",
                        )
                    if val_ssim > val_max_ssim:
                        val_max_ssim = val_ssim
                        save_single_model(
                            model,
                            directories["best_val_ssim"],
                            f"bestvalssim_{val_ssim:.4f}_{epoch}.pth",
                        )
                barrier(distributed)

            scheduler.step()

        if is_main_process:
            print(
                f"Training complete. Min loss={min_loss:.4f}, "
                f"max PSNR={max_psnr:.4f}, max SSIM={max_ssim:.4f}"
            )
            logging.info(
                "Training complete - Min loss: %.4f, max PSNR: %.4f, max SSIM: %.4f",
                min_loss,
                max_psnr,
                max_ssim,
            )
    finally:
        if writer is not None:
            writer.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train model_MSO_CCA_RAWLoss on the MCR dataset."
    )
    parser.add_argument("--data_dir", default="/mnt/data/zpc/data/MCR")
    parser.add_argument("--train_list_file", default="MCR_train_list.txt")
    parser.add_argument("--test_list_file", default="MCR_test_list.txt")
    parser.add_argument(
        "--result_dir",
        default=(
            "/mnt/data/zpc/MSOCO/MSOCO/Results/result_MSO_CCA/MCR/"
            "model_MSO_CCA_RAWLoss_Perceptual"
        ),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Packed RAW patch size; 0 uses the full MCR image.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per GPU.",
    )
    parser.add_argument("--num_epoch", type=int, default=2001)
    parser.add_argument("--val_freq", type=int, default=3)
    parser.add_argument(
        "--cosine_t_max",
        "--t_max",
        dest="cosine_t_max",
        type=int,
        default=2000,
        help="Number of epochs for one cosine decay from lr to eta_min.",
    )
    parser.add_argument(
        "--cosine_eta_min",
        "--eta_min",
        dest="cosine_eta_min",
        type=float,
        default=1e-7,
        help="Minimum learning rate used by cosine annealing.",
    )
    parser.add_argument("--workers", type=int, default=32)

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.cosine_t_max <= 0:
        parser.error("--cosine_t_max/--t_max must be greater than 0")
    if not 0.0 <= args.cosine_eta_min <= args.lr:
        parser.error("--cosine_eta_min/--eta_min must be between 0 and --lr")
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_and_evaluate(parse_args())

#  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 /mnt/data/zpc/MSOCO/MSOCO/train_mcr.py --resume --lr 7e-05
