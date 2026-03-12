import copy
import math
import random

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import Adapter_merge
from utils.toolkit import tensor2numpy

num_workers = 8


def contrastive_loss(features: torch.Tensor,
                     labels: torch.Tensor,
                     tau: float = 0.07) -> torch.Tensor:
    """
    Simple supervised contrastive-style loss using cosine similarity.

    Args:
        features: [N, D] feature tensor.
        labels: [N] label tensor.
        tau: Margin for negative pairs.

    Returns:
        Scalar loss tensor.
    """
    f = F.normalize(features, dim=1)
    sim = torch.matmul(f, f.T)  # [N, N]

    mask = labels.unsqueeze(0) == labels.unsqueeze(1)

    # positive: same label but not self
    pos = sim[mask & ~torch.eye(len(labels), dtype=bool, device=features.device)]
    # negative: different label
    neg = sim[~mask]

    loss_pos = (1.0 - pos).mean()
    loss_neg = F.relu(neg - tau).mean()
    return loss_pos + loss_neg


def compute_lambda(task_size: int,
                   lamda_max: float = 0.1,
                   lamda_min: float = 0.01,
                   k: float = 2.3979) -> float:
    """
    Compute lambda for mixing CE and contrastive loss as a function of task size.

    Args:
        task_size: Number of classes in the current task.
        lamda_max: Maximum lambda.
        lamda_min: Minimum lambda.
        k: Decay factor.

    Returns:
        Scalar lambda value.
    """
    lam = lamda_min + (lamda_max - lamda_min) * math.exp(-k * (task_size - 1))
    return lam


def seed_all(s: int) -> None:
    """
    Set global random seeds for reproducibility.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """
    Create a worker_init_fn for DataLoader that sets per-worker seeds.
    """

    def _seed_worker(worker_id: int):
        worker_seed = (base_seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


class Learner(BaseLearner):
    """
    One-A learner with adapter-based incremental training and SVD-based
    adapter alignment / merging.
    """

    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = (
            args["weight_decay"] if args["weight_decay"] is not None else 5e-4
        )
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8

        self.init_cls = args["cls_per_task"][0]
        self.inc = args["cls_per_task"]

        self.use_diagonal = args["use_diagonal"]


        self._network = Adapter_merge(args, True)
        self.task_adapter_indices = [0] * len(self.inc)

        self.logger = args["logger"]


        self.rb_mode = args["rb_mode"]
        self.k = args["lamda_k"]
        self.rb_kappa = args["rb_kappa"]
        self.rb_tau_q = args["rb_tau_q"]

        self.use_flexe = args["use_flexe"]
        if self.use_flexe:
            self.config = {
                "epoch_base": args["epoch_base"],
                "epoch_min": args["epoch_min"],
                "epoch_max": args["epoch_max"],
                "epoch_ref_t0": args["epoch_ref_t0"],
                "epoch_beta": args["epoch_beta"],
            }

    def after_task(self) -> None:
        """
        Hook called after each task finishes.
        Freeze current network and register the adapter into the adapter list.
        """
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id: int):
        """
        Return class index range [start, end) for a given task.

        Args:
            task_id: Task index.

        Returns:
            (start_cls, end_cls)
        """
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + sum(self.inc[1:task_id])
            end_cls = start_cls + self.inc[task_id]
        return start_cls, end_cls

    def replace_fc(self, train_loader: DataLoader) -> None:
        """
        Replace classifier weights with class prototypes computed from features
        of the current task (ProtoNet-style FC initialization).
        """
        model = self._network
        model.eval()

        with torch.no_grad():
            start_idx = 0
            for index in range(start_idx, self._cur_task + 1):
                if self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                if index != -1:
                    embedding_list, label_list = [], []
                    for _, batch in enumerate(train_loader):
                        _, data, label = batch
                        data = data.to(self._device)
                        label = label.to(self._device)

                        embedding = model.backbone.forward_proto(
                            data, adapt_index=0
                        )
                        embedding_list.append(embedding.cpu())
                        label_list.append(label.cpu())

                    embedding_list = torch.cat(embedding_list, dim=0)
                    label_list = torch.cat(label_list, dim=0)

                    class_list = np.unique(
                        self.train_dataset_for_protonet.labels
                    )
                    for class_index in class_list:
                        data_index = (label_list == class_index).nonzero().squeeze(-1)
                        embedding = embedding_list[data_index]
                        proto = embedding.mean(0)
                        model.fc.weight.data[class_index, : self._network.out_dim] = proto
                    break

    def incremental_train(self, data_manager) -> None:
        """
        Main per-task training entry point:
        1) Update classifier size.
        2) Build train / test loaders.
        3) Train with CE + contrastive loss.
        4) SVD-align and merge current adapter into base adapter.
        5) Refresh FC weights with prototypes.
        """
        self._cur_task += 1
        self._cur_task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_task_size

        seed_all(self.args["seed"])
        g_train = torch.Generator(device="cpu").manual_seed(self.args["seed"])

        self._network.update_fc(self._total_classes)
        self.logger.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        self.data_manager = data_manager

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=g_train,
            worker_init_fn=make_worker_init_fn(self.args["seed"]),
        )

        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # 1) standard training
        self._train(self.train_loader, self.test_loader)

        # 2) SVD-based alignment + adaptive merge into base adapter
        try:
            merge_mode = self.args.get("merge_mode", "adaptive")
            merge_alpha = self.args.get("merge_alpha", 1.0)
            merge_beta = self.args.get("merge_beta", 1.0)
            merge_gamma = self.args.get("merge_gamma", 1.0)
            svd_dev = self.args.get("merge_svd_device", "auto")

            self.logger.info(f"[Adapter Merge] mode={merge_mode}")

            if self._cur_task != 0:
                self.svd_align_merge_adapter_into(
                    src_index=self._cur_task,
                    dst_index=0,
                    mode=merge_mode,
                    alpha=merge_alpha,
                    beta=merge_beta,
                    gamma=merge_gamma,
                    svd_device=svd_dev,
                )
            else:
                # First task: set base_adapter directly from current adapter
                self._network.backbone.base_adapter = (
                    copy.deepcopy(self._network.backbone.cur_adapter)
                    .requires_grad_(False)
                )

            # Route current task to merged adapter index (0)
            self.task_adapter_indices[self._cur_task] = 0

            self.logger.info(
                "[SVD-Align Merge] adapter {} -> 0 (mode={}, alpha={}, beta={}, gamma={})"
                .format(self._cur_task, merge_mode, merge_alpha, merge_beta, merge_gamma)
            )
        except Exception as e:
            self.logger.exception(f"[SVD-Align Merge] Failed: {e}")

        # 3) refresh classifier with prototypes of merged space
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """
        Wrapper to construct optimizer/scheduler and call the main training loop.
        """
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            # scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]
            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            # scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        # self._init_train(train_loader, test_loader, optimizer, scheduler)
        self._init_train(train_loader, test_loader, optimizer)

    def get_optimizer(self, lr: float):
        """
        Build optimizer over trainable parameters.
        """
        params = filter(lambda p: p.requires_grad, self._network.parameters())

        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(
                params,
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay,
            )
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(
                params,
                lr=lr,
                weight_decay=self.weight_decay,
            )
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args['optimizer']}")

        return optimizer

    def get_scheduler(self, optimizer, epoch: int):
        """
        Build LR scheduler.
        """
        if self.args["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=epoch,
                eta_min=self.min_lr,
            )
        elif self.args["scheduler"] == "steplr":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        elif self.args["scheduler"] == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.args['scheduler']}")

        return scheduler

    def _init_train(self,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    optimizer,
                    # scheduler
                    ) -> None:
        """
        Main epoch loop: CE + contrastive loss, with optional dynamic epochs.
        """
        if not self.use_flexe:
            if self._cur_task == 0 or self.init_cls == self.inc:
                epochs = self.args["init_epochs"]
            else:
                epochs = self.args["later_epochs"]
        else:
            epochs = self.compute_dynamic_epochs(
                task_size=self._cur_task_size,
                E0=self.config["epoch_base"],
                Emin=self.config["epoch_min"],
                Emax=self.config["epoch_max"],
                t0=self.config["epoch_ref_t0"],
                beta=self.config["epoch_beta"],
            )
            self.logger.info(
                "[Task {}] class_num={} -> epochs={} "
                "(E0={}, Emin={}, Emax={}, t0={}, beta={})".format(
                    self._cur_task,
                    self._cur_task_size,
                    epochs,
                    self.config["epoch_base"],
                    self.config["epoch_min"],
                    self.config["epoch_max"],
                    self.config["epoch_ref_t0"],
                    self.config["epoch_beta"],
                )
            )
        scheduler = self.get_scheduler(optimizer, epochs)
        prog_bar = tqdm(range(epochs))
        lamda = compute_lambda(self._cur_task_size, k=self.k)
        self.logger.info(f"lambda (CE/contrastive mixing) = {lamda}")

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )

                output = self._network(inputs, test=False)
                logits = output["logits"]
                features = output["features"]

                loss_ce = F.cross_entropy(logits, aux_targets)
                loss_ctr = contrastive_loss(features, aux_targets)
                loss = (1.0 - lamda) * loss_ce + lamda * loss_ctr
               

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(
                tensor2numpy(correct) * 100.0 / total, decimals=2
            )

            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            )
            prog_bar.set_description(info)

        self.logger.info(info)

    def _compute_accuracy(self, model, loader: DataLoader) -> float:
        """
        Compute top-1 accuracy of a given model on a loader.
        """
        model.eval()
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(
            tensor2numpy(correct) * 100.0 / total, decimals=2
        )

    def _eval_cnn(self, loader: DataLoader):
        """
        Evaluate CNN classifier: returns top-k predictions and ground truth.
        """
        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _as_2d(self, mat: torch.Tensor):
        """
        Reshape an arbitrary tensor into a 2D matrix [out, in_like] for SVD.

        Rules:
            - 0D/1D -> [1, N]
            - 2D    -> as is
            - >2D   -> flatten all but first dimension to columns
        """
        if mat.dim() == 0:
            m2d = mat.reshape(1, 1)
            info = ("scalar", mat.shape)
        elif mat.dim() == 1:
            m2d = mat.unsqueeze(0)
            info = ("1d", mat.shape)
        elif mat.dim() == 2:
            m2d = mat
            info = ("2d", mat.shape)
        else:
            out = mat.shape[0]
            m2d = mat.reshape(out, -1)
            info = ("nd", mat.shape)
        return m2d, info

    def _from_2d(self, m2d: torch.Tensor, info):
        """
        Inverse of _as_2d: reshape a 2D matrix back to the original shape.
        """
        kind, orig_shape = info
        if kind == "scalar":
            return m2d.reshape(orig_shape)
        elif kind == "1d":
            return m2d.squeeze(0).reshape(orig_shape)
        elif kind == "2d":
            return m2d.reshape(orig_shape)
        else:
            return m2d.reshape(orig_shape)

    @torch.no_grad()
    def merge_adapter_into(self,
                           src_index: int,
                           dst_index: int,
                           alpha: float = 1.0,
                           beta: float = 1.0) -> None:
        """
        Simple backup: linear merge without SVD
            dst = alpha * dst + beta * src
        """
        adapters = self._network.backbone.adapter_list
        src = adapters[src_index]
        dst = adapters[dst_index]

        src_params = dict(src.named_parameters())
        for name, p_dst in dst.named_parameters():
            p_src = src_params[name]
            p_dst.mul_(alpha).add_(p_src, alpha=beta)

    @torch.no_grad()
    def copy_adapter(self, to_index: int, from_index: int) -> None:
        """
        Copy parameters from one adapter to another.
        """
        adapters = self._network.backbone.adapter_list
        src = adapters[from_index]
        dst = adapters[to_index]

        src_params = dict(src.named_parameters())
        for name, p_dst in dst.named_parameters():
            p_dst.copy_(src_params[name])

    @torch.no_grad()
    def svd_align_merge_adapter_into(
        self,
        src_index: int,
        dst_index: int,
        mode: str = "adaptive",  # "adaptive" / "weighted_sum" / "info_adaptive" / "fixed"
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-8,
        svd_device: str = "auto",  # currently SVD runs on the tensor's own device
    ) -> None:
        """
        SVD-based adapter alignment and merge.

        For each matched parameter:
            1) reshape dst/src/target to 2D.
            2) compute SVD on dst: M_dst = U S V_dst^T.
            3) project src into dst's basis.
            4) blend in coefficient (V-space) with weights w_d, w_s.
            5) directional gating in singular directions.
            6) reconstruct and write back into the target adapter.

        The final merged adapter is stored in adapter_list[dst_index] and
        also used to update backbone.base_adapter outside this function.
        """
        # Shortcuts
        adapters = self._network.backbone.adapter_list
        cur = self._network.backbone.cur_adapter
        target_mod = adapters[dst_index]

        # Use accumulated class counts as a proxy for importance
        alpha_eff = float(sum(self.inc[: self._cur_task])) if hasattr(self, "inc") else 1.0
        beta_eff = float(self.inc[self._cur_task]) if hasattr(self, "inc") else 1.0

        # Decide which adapter is treated as "dst" in SVD computation
        if alpha_eff <= beta_eff:
            # dst = current adapter, src = previous merged
            dst_mod = cur
            src_mod = adapters[dst_index]
        else:
            # dst = previous merged, src = current adapter
            dst_mod = adapters[dst_index]
            src_mod = cur

        self._network.backbone.base_adapter = (
            copy.deepcopy(dst_mod).requires_grad_(False)
        )

        src_params = dict(src_mod.named_parameters())
        dst_params = dict(dst_mod.named_parameters())

        for name, p_tgt in target_mod.named_parameters():
            p_src = src_params[name]
            p_dst = dst_params[name]

            if p_tgt.shape != p_src.shape or p_tgt.shape != p_dst.shape:
                raise RuntimeError(
                    "[svd_align_merge_adapter_into] Shape mismatch at {}: "
                    "target {}, src {}, dst {}".format(
                        name, p_tgt.shape, p_src.shape, p_dst.shape
                    )
                )

            # Flatten to 2D matrices
            m_ref, info = self._as_2d(p_tgt.data)
            m_dst, _ = self._as_2d(p_dst.data)
            m_src, _ = self._as_2d(p_src.data)

            # SVD on dst
            U, S, Vh_dst = torch.linalg.svd(m_dst, full_matrices=False)

            S_inv = torch.where(S > eps, 1.0 / S, torch.zeros_like(S))
            coef_src_in_dst = (
                (S_inv.unsqueeze(1) * U.transpose(0, 1)) @ m_src
            ).contiguous()

            # Determine blending weights wd, ws
            if mode == "adaptive":
                nd = torch.linalg.norm(m_dst, ord="fro")
                ns = torch.linalg.norm(m_src, ord="fro")
                nd_g = nd.clamp_min(eps).pow(gamma)
                ns_g = ns.clamp_min(eps).pow(gamma)
                wd = (nd_g / (nd_g + ns_g + eps)).item()
                ws = 1.0 - wd
            elif mode == "weighted_sum":
                alpha_ws = sum(self.inc[: self._cur_task])
                beta_ws = self.inc[self._cur_task]
                s = max(alpha_ws + beta_ws, eps)
                wd = float(alpha_ws / s)
                ws = float(beta_ws / s)
            elif mode == "info_adaptive":
                # (1) Frobenius energy as information proxy
                U_d, S_d, _ = torch.linalg.svd(m_dst, full_matrices=False)
                U_s, S_s, _ = torch.linalg.svd(m_src, full_matrices=False)
                phi_old = torch.sum(S_d**2)
                phi_new = torch.sum(S_s**2)

                s = max(phi_old + phi_new, eps)
                wd = (phi_old / s).item()
                ws = (phi_new / s).item()
            else:
                # fixed weights
                s = max(alpha + beta, eps)
                wd = float(alpha / s)
                ws = float(beta / s)

            Vh_blend = wd * Vh_dst + ws * coef_src_in_dst
            delta = Vh_blend - Vh_dst

            # Directional gating on singular directions
            r = S.shape[0]
            rb_mode = getattr(self, "rb_mode", "sv")
            r_head_frac = getattr(self, "rb_r_head_frac", 0.3)
            rho_head = getattr(self, "rb_rho_head", 0.1)
            rho_tail = getattr(self, "rb_rho_tail", 0.9)

            
            # Smooth gating based on normalized singular values
            S_norm = S / (S[0] + eps)

            tau_mode = getattr(self, "rb_tau_mode", "quantile")
            kappa = self.rb_kappa

            if tau_mode == "quantile":
                tau_q = self.rb_tau_q
                tau = torch.quantile(
                    S_norm, torch.tensor(tau_q, device=S_norm.device)
                )
            else:
                tau = torch.tensor(
                    getattr(self, "rb_tau", 0.5),
                    device=S_norm.device,
                    dtype=S_norm.dtype,
                )

            mask_row = 1.0 / (1.0 + torch.exp(kappa * (S_norm - tau)))

            if rb_mode == "hybrid":
                # Map (0,1) to [rho_head, rho_tail]
                mask_row = rho_head + (rho_tail - rho_head) * mask_row

            mask = mask_row.unsqueeze(1)
            Vh_m = Vh_dst + mask * delta
            m_m = ((U * S.unsqueeze(0)) @ Vh_m).to(p_tgt.device)

            p_tgt.copy_(self._from_2d(m_m, info))

    def compute_dynamic_epochs(
        self,
        task_size: int,
        E0: int = 20,
        Emin: int = 10,
        Emax: int = 30,
        t0: int = 10,
        beta: float = 0.5,
    ) -> int:
        """
        Compute dynamic training epochs as a function of task size.

        epoch(t) = clamp(Emin, E0 * (t / t0)^beta, Emax)
        """
        t = max(1, int(task_size))
        raw = E0 * (t / float(t0)) ** beta
        e = int(round(raw))
        e = max(Emin, min(Emax, e))
        return e
