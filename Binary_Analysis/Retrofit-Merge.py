import os
import re
import json
import math
import copy
import sys
import torch
import random
import logging
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Mapping, Union
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import torch.nn.functional as F

try:
    from transformers import T5ForConditionalGeneration, RobertaTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
    from copy import deepcopy
    import safetensors.torch as st
    from safetensors.torch import load_file
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing Python dependency for Binary_Analysis. "
        "Install Binary_Analysis/requirements.txt inside the intended CodeT5/BinT5 environment."
    ) from exc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UTIL_DIR = os.path.join(THIS_DIR, "Util")
if UTIL_DIR not in sys.path:
    sys.path.insert(0, UTIL_DIR)

from retrofit_config import parse_args

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================
# Data pipeline
# =========================
@dataclass
class Example:
    idx: int
    source: str
    target: str
    url: str = None


@dataclass
class InputFeatures:
    example_index: int
    source_ids: list
    target_ids: list
    url: str = None


def read_summarize_examples(filename, data_num):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(Example(idx=idx, source=code, target=nl))
            if data_num != -1 and idx + 1 == data_num:
                break
    return examples


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item
    source_str = example.source.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length,
                                  padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length,
                                      padding='max_length', truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1
    return InputFeatures(example_index, source_ids, target_ids, url=example.url)


class Args:
    def __init__(self):
        self.model_type = 'codet5'
        self.max_source_length = 512
        self.max_target_length = 128
        self.add_task_prefix = False
        self.add_lang_ids = False
        self.task = 'summarize'
        self.sub_task = None
        self.local_rank = -1
        self.data_num = -1
        self.cache_path = './cache_tmp'


def load_and_cache_gen_data(args, filename, tokenizer, split_tag, sample_cap,
                            only_src: bool = False,
                            is_sample: bool = False,
                            ):
    examples = read_summarize_examples(filename, args.data_num)

    if sample_cap is not None and len(examples) > sample_cap:
        examples = random.sample(examples, sample_cap)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
    features = list(map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples))))
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    if split_tag == 'test' or only_src:
        data = TensorDataset(all_source_ids)
    else:
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_target_ids)
    return examples, data


# =========================
# Utilize
# =========================
def _align_pad_token_id(model: T5ForConditionalGeneration, tokenizer: RobertaTokenizer):
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def apply_adapter_and_merge(base_model, adapter_path, device):
    m = PeftModel.from_pretrained(base_model, adapter_path)
    m = m.merge_and_unload()  # merge to full
    m.to(device).eval()
    return m


# =========================
# LoRA key patterns
# =========================
_LORA_B_RE = re.compile(r"\.lora_B(\.[^.]+)?\.weight$")
_LORA_A_RE = re.compile(r"\.lora_A(\.[^.]+)?\.weight$")


def is_lora_B_key(name: str) -> bool:
    return bool(_LORA_B_RE.search(name)) and name.endswith("weight")


def is_lora_A_key(name: str) -> bool:
    return bool(_LORA_A_RE.search(name)) and name.endswith("weight")


def map_B_to_A_key(b_key: str) -> str:
    """
    keep adapter name unchanged: .lora_B.default.weight -> .lora_A.default.weight
    """
    if ".lora_B." in b_key:
        return b_key.replace(".lora_B.", ".lora_A.")
    return b_key.replace(".lora_B.weight", ".lora_A.weight")


def map_B_to_base_weight_key(b_key: str) -> str:
    """
    remove lora_B.*.weight segment, map to underlying Linear .weight (may be unreliable, kept for fallback)
    """
    return _LORA_B_RE.sub(".weight", b_key)


# =========================
# LoRA-B utilities
# =========================
def collect_lora_B_tensors(peft_model: PeftModel, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Collect LoRA B matrices (param names *.lora_B[.adapter]?.weight) as τ_B (since B_pre≈0).
    """
    tau_B = {}
    for n, p in peft_model.named_parameters():
        if is_lora_B_key(n) and p.dtype.is_floating_point:
            tau_B[n] = p.detach().to(device).clone()
    logger.info(f"[collect_lora_B_tensors] collected {len(tau_B)} B tensors")
    return tau_B


@torch.no_grad()
def zero_all_lora_B_(peft_model: PeftModel):
    """Zero all LoRA B matrices (simulate θ_pre)."""
    cnt = 0
    for n, p in peft_model.named_parameters():
        if is_lora_B_key(n) and p.dtype.is_floating_point:
            p.zero_();
            cnt += 1
    logger.info(f"[zero_all_lora_B_] zeroed {cnt} B tensors")


@torch.no_grad()
def apply_mask_to_lora_B_(peft_model: PeftModel, mask: Dict[str, torch.Tensor], tau_B: Dict[str, torch.Tensor]):
    """
    Apply mask gate (soft/binary) to LoRA-B: B ← gate ⊙ τ_B
    reset→0 first, then set to gate*tau_B
    """
    cnt = 0
    for n, p in peft_model.named_parameters():
        if is_lora_B_key(n) and p.dtype.is_floating_point:
            if n not in tau_B:
                p.zero_()
            else:
                g = mask.get(n, None)
                if g is None:
                    p.zero_()
                else:
                    tgt = (g.to(p.device) * tau_B[n].to(p.device))
                    p.copy_(tgt);
                    cnt += 1
    logger.info(f"[apply_mask_to_lora_B_] applied to {cnt} B tensors")


# =========================
# Retrofit (full weights) kept; added RetrofitB (LoRA-B)
# =========================
class Retrofit:
    """
    Common logic: train gate S optimization/diagnostic/quick-eval interface.
    RetrofitB overrides reset/apply/quick_eval to work on LoRA-B.
    """

    def __init__(
            self,
            base_model: T5ForConditionalGeneration,
            tokenizer: RobertaTokenizer,
            tau: Dict[str, torch.Tensor],
            device: torch.device,
            lr: float = 1e7,
            l1_strength: float = 10.0,
            l2_strength: float = 10.0,
            num_epochs: int = 10,
            max_batches: int = 8,
            sigmoid_bias: float = 5.0,
            sparsity: Optional[float] = 1e-5,
            restrict_to_linear: bool = True,  # kept for compatibility
            use_kg_loss: bool = False,
    ):
        self.model = base_model
        self.tok = tokenizer
        self.device = device
        self.lr = lr
        self.l1 = l1_strength
        self.l2 = l2_strength
        self.epochs = num_epochs
        self.max_batches = max_batches
        self.sigmoid_bias = sigmoid_bias
        self.sparsity = sparsity
        self.restrict_to_linear = restrict_to_linear
        self.use_kg_loss = use_kg_loss

        # move τ to device
        self.tau: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in tau.items()}

        # error if τ empty
        if len(self.tau) == 0:
            names = [n for n, _ in self.model.named_parameters()]
            sample = [n for n in names if ("lora" in n or "Lora" in n)][:50]
            raise ValueError(
                "No τ tensors were provided (empty). "
                "Likely LoRA-B keys were not matched. "
                "Please check adapter is loaded and key patterns. "
                f"Sample param names containing 'lora': {sample}"
            )

        # learnable gate S
        self.mask_params: Dict[str, torch.nn.Parameter] = {
            k: torch.nn.Parameter(torch.zeros_like(v, device=device), requires_grad=True)
            for k, v in self.tau.items()
        }

        self.model.to(self.device).eval()

        # init: global |τ|-top-k → S = ±sigmoid_bias
        if (self.sparsity is None) or (self.sparsity <= 0):
            init_rho = 0.01
        else:
            init_rho = float(self.sparsity)
        self._init_mask_by_tau_global_topk(init_rho)

    @staticmethod
    def _sigmoid(x):
        return torch.sigmoid(x)

    @torch.no_grad()
    def _init_mask_by_tau_global_topk(self, rho: float):
        """
        Global |τ| top-k init: >|thr| → +bias, else −bias
        """
        flat = torch.cat([v.abs().flatten() for v in self.tau.values()])
        K = max(1, int(math.floor(rho * flat.numel())))
        thr = torch.topk(flat, K, largest=True).values.min()
        for kname, v in self.tau.items():
            s = self.mask_params[kname]
            pos = (v.abs() >= thr)
            s.data.copy_(pos * self.sigmoid_bias + (~pos) * (-self.sigmoid_bias))

        with torch.no_grad():
            total, ones = 0, 0
            for kname, p in self.mask_params.items():
                m0 = (self._sigmoid(p) >= 0.5).float()
                ones += int(torch.count_nonzero(m0))
                total += m0.numel()
            logger.info(f"[retrofit] init sparsity ≈ {ones / total:.6f} ({ones}/{total})")

    # overridden in RetrofitB
    @torch.no_grad()
    def _reset_to_pretrained(self):
        ...

    @torch.no_grad()
    def _apply_soft_merge_inplace(self):
        ...

    def _get_soft_mask(self) -> Dict[str, torch.Tensor]:
        gamma = {}
        for k, p in self.mask_params.items():
            gamma[k] = self._sigmoid(p)  # sigmoid(p)
        return gamma

    @torch.no_grad()
    def _report_sparsity(self, gamma: Dict[str, torch.Tensor], tag="final"):
        total = sum(int(v.numel()) for v in gamma.values())
        ones = sum(int(torch.count_nonzero(v)) for v in gamma.values())
        logger.info(f"[retrofit] {tag} sparsity = {ones / total:.6f} ({ones}/{total})")

    def _clean_ids_for_decode(self, ids, tok):
        ids = ids.detach().cpu().tolist()
        ids = [tok.pad_token_id if x == -100 else x for x in ids]
        try:
            eos = ids.index(tok.eos_token_id)
            ids = ids[:eos + 1]
        except ValueError:
            pass
        ids = [x for x in ids if x != tok.pad_token_id]
        return ids

    def _decode_pair(self, input_ids, labels, tok):
        src_ids = self._clean_ids_for_decode(input_ids, tok)
        tgt_ids = self._clean_ids_for_decode(labels, tok)
        src_txt = tok.decode(src_ids, skip_special_tokens=True)
        tgt_txt = tok.decode(tgt_ids, skip_special_tokens=True)
        return src_txt, tgt_txt

    def train_graft(self, dataloader: DataLoader, log_prefix: str = "") -> Dict[str, torch.Tensor]:
        """
        Training with encoder-MSE KG loss (kg_loss_from_enc_mse_by_avg_conf)
        + CE loss from model
        + L1/L2 regularization on mask parameters.
        Unified, detailed logging for all losses.
        """
        import copy
        import torch
        import torch.nn.functional as F

        total_samples = len(dataloader.dataset)
        logger.info(f"[Training] Total samples in dataset: {total_samples}")
        device = self.device

        # -------- freeze old/new teacher (outside training loop) --------
        with torch.no_grad():
            # old: B=0 (reset pretrained)
            self._reset_to_pretrained()
            old_model = copy.deepcopy(self.model).to(device)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad_(False)

            # new: B = tau (gate=1)
            one_gate = {k: torch.ones_like(v) for k, v in self.tau.items()}
            apply_mask_to_lora_B_(self.model, one_gate, self.tau)
            new_model = copy.deepcopy(self.model).to(device)
            new_model.eval()
            for p in new_model.parameters():
                p.requires_grad_(False)

        # restore to soft merge state
        self._reset_to_pretrained()
        self._apply_soft_merge_inplace()

        # helper
        def _teacher_forward(_model, input_ids, labels_for_loss):
            with torch.no_grad():
                out = _model(input_ids=input_ids, labels=labels_for_loss, output_hidden_states=True)
                if hasattr(out, "encoder_last_hidden_state"):
                    enc = out.encoder_last_hidden_state
                elif hasattr(out, "encoder_hidden_states") and out.encoder_hidden_states:
                    enc = out.encoder_hidden_states[-1]
                else:
                    raise RuntimeError("Teacher forward: encoder hidden states not found")
                return out.logits, enc

        # training epochs
        for ep in range(self.epochs):
            logger.info(f"[retrofit] {log_prefix} epoch={ep + 1}/{self.epochs}")

            self._reset_to_pretrained()
            self._apply_soft_merge_inplace()

            acc_grads: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in self.tau.items()}
            steps = 0

            ce_losses, kg_losses, total_losses = [], [], []
            keep_losses, gain_losses, l1_losses, l2_losses = [], [], [], []
            epoch_old, epoch_new, epoch_none = 0, 0, 0

            for step, batch in enumerate(dataloader):
                if step >= self.max_batches:
                    break
                steps += 1

                self.model.zero_grad(set_to_none=True)

                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                labels_for_ce = labels.clone()
                labels_for_ce[labels_for_ce == self.tok.pad_token_id] = -100

                # forward (student)
                out = self.model(input_ids=input_ids, labels=labels_for_ce, output_hidden_states=True)
                ce_loss = out.loss
                student_logits = out.logits

                if hasattr(out, "encoder_last_hidden_state"):
                    enc_student = out.encoder_last_hidden_state
                elif hasattr(out, "encoder_hidden_states") and out.encoder_hidden_states:
                    enc_student = out.encoder_hidden_states[-1]
                else:
                    raise RuntimeError("Student forward: encoder hidden states not found")

                # teachers
                with torch.no_grad():
                    logits_old, enc_old = _teacher_forward(old_model, input_ids, labels_for_ce)
                    logits_new, enc_new = _teacher_forward(new_model, input_ids, labels_for_ce)

                enc_attention_mask = (input_ids != self.tok.pad_token_id).long()

                # compute KG loss
                kg_out = kg_loss_from_enc_mse_by_avg_conf(
                    labels=labels_for_ce,
                    student_logits=student_logits,
                    enc_old=enc_old,
                    enc_new=enc_new,
                    enc_student=enc_student,
                    enc_attention_mask=enc_attention_mask,
                    logits_old=logits_old,
                    ignore_index=-100,
                    conf_T=2.0,
                    conf_thresh=0.6,
                    gate_mode="soft",
                    old_weight_hi=0.8,
                    old_weight_lo=0.2,
                    beta=0.08,
                    feat_lambda=10.0,
                )

                epoch_old += kg_out.get("num_old", 0)
                epoch_new += kg_out.get("num_new", 0)
                epoch_none += kg_out.get("num_none", 0)

                kg_loss = kg_out["L_kg"]
                
                total_elem = sum(p.numel() for p in self.mask_params.values())
                l1_reg = sum(p.abs().sum() for p in self.mask_params.values())
                l2_reg = sum((p ** 2).sum() for p in self.mask_params.values())

                kg_loss = kg_out["L_kg"] if self.use_kg_loss else torch.tensor(0.0, device=ce_loss.device)
                total_loss = ce_loss + kg_loss + self.l1 * l1_reg + self.l2 * l2_reg
                total_loss.backward()

                # accumulate grads
                for n, p in self.model.named_parameters():
                    if (n in self.tau) and (p.grad is not None):
                        acc_grads[n] += (self.lr * p.grad.detach()).to(device)

                # record losses
                ce_losses.append(ce_loss.item())
                total_losses.append(total_loss.item())
                l1_losses.append(l1_reg.item())
                l2_losses.append(l2_reg.item())

                if self.use_kg_loss:
                    kg_losses.append(kg_loss.item() if isinstance(kg_loss, torch.Tensor) else float(kg_loss))
                else:
                    kg_losses.append(0.0)

                # per-step summary (every 50 steps)
                if (step % 50) == 0:
                    if self.use_kg_loss:
                        mean_conf = float(kg_out["mean_conf_old"].mean().item())
                    else:
                        mean_conf = torch.tensor(0.0, device=ce_loss.device)
                    logger.info(
                        f"[Step {step}] CE={ce_loss.item():.4f} | KG={kg_loss.item():.6f} | "
                        f"L1={self.l1*l1_reg.item()/total_elem:.6f} | L2={self.l2*l2_reg.item()/total_elem:.6f} | "
                    )

            # epoch summary
            print(f"[Epoch {ep + 1}] choose old: {epoch_old}, choose new: {epoch_new}, choose none: {epoch_none}")

            if steps > 0:
                for k in acc_grads:
                    acc_grads[k] /= float(steps)

            # update mask
            with torch.no_grad():
                for k in self.tau:
                    g_w = acc_grads.get(k, None)
                    if g_w is None:
                        continue
                    tv = self.tau[k]
                    p = self.mask_params[k]
                    gate = torch.sigmoid(p)
                    deriv = gate * (1.0 - gate)
                    grad_S = (g_w * tv) * deriv
                    reg_l1 = self.l1 * deriv
                    reg_l2 = self.l2 * 2.0 * gate * deriv
                    p -= (grad_S + reg_l1 + reg_l2)
                    p.clamp_(-8.0, 8.0)

            # log epoch averages
            logger.info(
                f"[Epoch {ep + 1:02d}] CE={sum(ce_losses) / len(ce_losses):.4f} | "
                f"KG={sum(kg_losses) / len(kg_losses):.6f} | "
                f"L1={self.l1 * sum(l1_losses) / len(l1_losses) / total_elem:.6f} | "
                f"L2={self.l2 * sum(l2_losses) / len(l2_losses) / total_elem:.6f} | "
                f"Total={sum(total_losses) / len(total_losses):.6f}"
            )

            # (keep quick-eval unchanged)
            if (ep + 1) == self.epochs or (ep + 1) % 2 == 0:
                gamma_tmp = self._get_soft_mask()
                try:
                    eval_loss = self._quick_eval_with_mask(gamma_tmp, dataloader, max_batches=2)
                    logger.info(f"[retrofit] {log_prefix} epoch={ep + 1} quick_eval_loss≈{eval_loss:.4f}")
                except Exception as e:
                    logger.warning(f"[retrofit] quick eval skipped: {e}")

        gamma = self._get_soft_mask()
        self._report_sparsity(gamma, tag="final")
        return gamma

    @torch.no_grad()
    def _quick_eval_with_mask(self, gamma: Dict[str, torch.Tensor], dataloader: DataLoader, max_batches=2) -> float:
        # RetrofitB impl
        raise NotImplementedError


class RetrofitB(Retrofit):
    """
    Work on LoRA-B:
      - τ := B_ft (B_pre≈0)
      - reset:    B ← 0
      - apply:    B ← σ(S) ⊙ τ
    Other training/logging consistent with parent.
    """

    def __init__(self, peft_model: PeftModel, tokenizer: RobertaTokenizer,
                 tau_B: Dict[str, torch.Tensor], device: torch.device,
                 lr=1e7, l1_strength=10.0, l2_strength=10.0, num_epochs=10, max_batches=8,
                 sigmoid_bias=5.0, sparsity: Optional[float] = 1e-5, use_kg_loss=False):
        super().__init__(
            base_model=peft_model, tokenizer=tokenizer, tau=tau_B, device=device,
            lr=lr, l1_strength=l1_strength, l2_strength=l2_strength, num_epochs=num_epochs, max_batches=max_batches,
            sigmoid_bias=sigmoid_bias, sparsity=sparsity, restrict_to_linear=True, use_kg_loss=use_kg_loss
        )
        # freeze non-B
        for n, p in self.model.named_parameters():
            if is_lora_B_key(n):
                p.requires_grad_(True)
            elif is_lora_A_key(n) or ("lora" not in n):
                p.requires_grad_(False)
            else:
                p.requires_grad_(False)

    @torch.no_grad()
    def _reset_to_pretrained(self):
        zero_all_lora_B_(self.model)

    @torch.no_grad()
    def _apply_soft_merge_inplace(self):
        gamma_soft = {k: torch.sigmoid(p) for k, p in self.mask_params.items()}
        apply_mask_to_lora_B_(self.model, gamma_soft, self.tau)

    @torch.no_grad()
    def _quick_eval_with_mask(self, gamma: Dict[str, torch.Tensor], dataloader: DataLoader, max_batches=2) -> float:
        self._reset_to_pretrained()
        apply_mask_to_lora_B_(self.model, gamma, self.tau)
        tot, cnt = 0.0, 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == self.tok.pad_token_id] = -100
            out = self.model(input_ids=input_ids, labels=labels_for_loss)
            tot += out.loss.item();
            cnt += 1
        return tot / max(1, cnt)


# =========================
# Merge (for B)
# =========================
def average_overlap_masks(masks_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    sum_masks: Dict[str, torch.Tensor] = {}
    for masks in masks_list:
        for n, m in masks.items():
            sum_masks[n] = sum_masks.get(n, torch.zeros_like(m)) + m
    processed = []
    for masks in masks_list:
        normed = {}
        for n, m in masks.items():
            denom = sum_masks[n].clamp_min(1.0)
            normed[n] = m / denom
        processed.append(normed)
    return processed


def merge_lora_B(tauB_list: List[Dict[str, torch.Tensor]],
                  processed_masks: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    B merge: B_merged[n] = Σ_i ( processed_masks[i][n] ⊙ tauB_list[i][n] )
    """
    merged: Dict[str, torch.Tensor] = {}
    for tauB_i, gamma_i in zip(tauB_list, processed_masks):
        for n, B in tauB_i.items():
            g = gamma_i.get(n, None)
            if g is None:
                continue
            add = (g.to(B.device) * B)
            if n not in merged:
                merged[n] = add.detach().clone()
            else:
                merged[n] += add
    return merged


# =========================
# Helpers: get module path from B param key; safe module getter
# =========================
def bkey_to_module_path(b_key: str) -> str:
    """
    '...<module>.lora_B[.adapter].weight' -> '<module>' path
    Example:
      encoder.block.0.layer.0.SelfAttention.q.lora_B.default.weight
    ->  encoder.block.0.layer.0.SelfAttention.q
    """
    return b_key.split(".lora_B", 1)[0]


def safe_get_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    if hasattr(root, "get_submodule"):
        return root.get_submodule(path)
    mod = root
    for p in path.split('.'):
        mod = getattr(mod, p)
    return mod


# =========================
# Export: merged_full_state_dict (.pt), recover ΔW from merged B and reference A and write back by module addressing
# =========================
def load_adapter_sd(adapter_path: str) -> Tuple[dict, dict]:
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    st_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.exists(st_path):
        sd = safetensors_load(st_path, device="cpu")
    elif os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError("adapter model not found in safetensors or bin at " + adapter_path)
    return cfg, sd


def strip_known_prefixes(path: str) -> str:
    """
    Strip common prefixes added by PeftModel wrapper to match bare T5 module paths.
    """
    for pref in ("base_model.model.", "base_model.", "model."):
        if path.startswith(pref):
            return path[len(pref):]
    return path


def bkey_to_module_paths(b_key: str) -> Tuple[str, str]:
    """
    From B param key generate two module paths:
      - probe_path: used on peft_probe (with wrapper prefix)
      - base_path:  used on bare T5 (prefix stripped)
    Example:
      base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_B.default.weight
      -> probe_path = base_model.model.encoder.block.0.layer.0.SelfAttention.q
         base_path  = encoder.block.0.layer.0.SelfAttention.q
    """
    core = _LORA_B_RE.sub("", b_key)
    if core.endswith("."):
        core = core[:-1]
    probe_path = core
    base_path = strip_known_prefixes(core)
    return probe_path, base_path


def safe_get_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Compatibility wrapper: prefer official get_submodule; fallback to manual traversal.
    """
    if hasattr(root, "get_submodule"):
        return root.get_submodule(path)
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def save_merged_full_state_dict_from_loraB(
        base_model: T5ForConditionalGeneration,
        reference_adapter_path: str,
        merged_B: Dict[str, torch.Tensor],
        out_pt_path: str,
        *,
        do_verify: bool = True,
        base_model_path: Optional[str] = None,
        verify_max_layers: int = 20,
        out_adapter_dir: Optional[str] = None,
):
    """
    θ_hat = θ_pre + ΔW; read actual fan_in_fan_out and scaling per layer:
      - fan_in_fan_out=False: Δ = (B @ A) * scale
      - fan_in_fan_out=True : Δ = (A @ B).T * scale
    """

    # 0) writable base (CPU)
    model_for_write = copy.deepcopy(base_model).eval()
    model_for_write.to("cpu")

    # 1) build peft_probe (not merged) to read real lora layer attributes
    if base_model_path is not None:
        tmp_base = T5ForConditionalGeneration.from_pretrained(base_model_path)
    else:
        tmp_base = copy.deepcopy(base_model)
    peft_probe = PeftModel.from_pretrained(tmp_base, reference_adapter_path, is_trainable=False).eval().cpu()

    # 2) per-B compute ΔW and write back
    def _parse_adapter_name(b_key: str) -> str:
        m = re.search(r"\.lora_B\.([^.]+)\.weight$", b_key)
        return m.group(1) if m else "default"

    updated, skipped = 0, []

    for b_key, B in merged_B.items():
        adapter_name = _parse_adapter_name(b_key)
        probe_path, base_path = bkey_to_module_paths(b_key)

        # find lora.Linear on peft_probe to read attributes
        try:
            lora_linear = safe_get_submodule(peft_probe, probe_path)
        except Exception:
            skipped.append((b_key, f"probe path not found: {probe_path}"))
            continue

        if not (hasattr(lora_linear, "lora_A") and adapter_name in lora_linear.lora_A):
            skipped.append((b_key, f"A[{adapter_name}] not found at {probe_path}"))
            continue
        if not (hasattr(lora_linear, "lora_B") and adapter_name in lora_linear.lora_B):
            skipped.append((b_key, f"B slot[{adapter_name}] not found at {probe_path}"))
            continue

        A = lora_linear.lora_A[adapter_name].weight.detach().cpu()  # [r, in]
        fan = bool(getattr(lora_linear, "fan_in_fan_out", False))

        # scaling: prefer module-level; fallback to global config
        scale = None
        if hasattr(lora_linear, "scaling") and isinstance(lora_linear.scaling,
                                                          dict) and adapter_name in lora_linear.scaling:
            scale = float(lora_linear.scaling[adapter_name])
        if scale is None:
            cfg, _ = load_adapter_sd(reference_adapter_path)
            r = int(cfg.get("r", A.shape[0]))
            lora_alpha = int(cfg.get("lora_alpha", r))
            scale = float(lora_alpha) / float(r)

        B_cpu = B.detach().cpu()  # [out, r]
        A_cpu = A.detach().cpu()  # [r, in]
        if fan:
            Delta = torch.matmul(A_cpu, B_cpu).t() * scale  # [out, in]
        else:
            Delta = torch.matmul(B_cpu, A_cpu) * scale  # [out, in]

        # find corresponding Linear.weight on bare T5 and write back
        try:
            linear = safe_get_submodule(model_for_write, base_path)
        except Exception:
            skipped.append((b_key, f"base path not found: {base_path}"))
            continue

        if not hasattr(linear, "weight"):
            skipped.append((b_key, f"module has no .weight: {base_path}"))
            continue

        W = linear.weight.detach()
        if W.shape != Delta.shape:
            skipped.append((b_key, f"shape mismatch W={tuple(W.shape)} vs Δ={tuple(Delta.shape)} at {base_path}"))
            continue

        with torch.no_grad():
            linear.weight.copy_(W + Delta)
        updated += 1

    if skipped:
        for (k, why) in skipped[:10]:
            logger.warning(f"[FullExport] skip {k}: {why}")
        if len(skipped) > 10:
            logger.warning(f"[FullExport] ... and {len(skipped) - 10} more skipped.")

    # 3) T5 weight tying
    if hasattr(model_for_write, "tie_weights"):
        try:
            model_for_write.tie_weights()
        except Exception as e:
            logger.warning(f"[FullExport] tie_weights failed: {e}")

    # 4) save CPU state_dict
    final_sd = model_for_write.state_dict()
    for k in list(final_sd.keys()):
        v = final_sd[k]
        if hasattr(v, "is_cuda") and v.is_cuda:
            final_sd[k] = v.cpu()
    os.makedirs(os.path.dirname(out_pt_path), exist_ok=True)
    torch.save(final_sd, out_pt_path)
    logger.info(f"[OK] merged_full_state_dict saved ({updated} modules updated) -> {out_pt_path}")

    # ===== write adapter: first place new A/B, then fill old buffers =====
    adapter_out_sd = {}
    # 1. place「zeroed merged_B」+ captured A
    for b_key, B in merged_B.items():
        adapter_name = _parse_adapter_name(b_key)
        probe_path, _ = bkey_to_module_paths(b_key)
        lora_linear = safe_get_submodule(peft_probe, probe_path)
        A = lora_linear.lora_A[adapter_name].weight.detach().cpu()
        # write adapter: align key names (remove .default.)
        b_key = b_key.replace(".lora_B.default.", ".lora_B.")
        a_key = b_key.replace(".lora_B.", ".lora_A.")
        adapter_out_sd[a_key] = A
        adapter_out_sd[b_key] = B.detach().cpu()  # use「zeroed B」

    # 2. fill non-A/B buffers from old file (scaling/dropout etc.)
    cfg, ref_sd = load_adapter_sd(reference_adapter_path)
    for k, v in ref_sd.items():
        # skip B matrix, only fill non-B keys
        if "lora_B" not in k and k not in adapter_out_sd:
            adapter_out_sd[k] = v
        elif "lora_B" in k:
            continue

    # 3. save
    os.makedirs(out_adapter_dir, exist_ok=True)
    safetensors_save(adapter_out_sd, os.path.join(out_adapter_dir, "adapter_model.safetensors"))

    # 4. copy old adapter_config.json
    import shutil
    ref_adapter_dir = os.path.dirname(reference_adapter_path)
    old_config_path = os.path.join(ref_adapter_dir, "lora_adapter/adapter_config.json")
    new_config_path = os.path.join(out_adapter_dir, "adapter_config.json")

    if os.path.exists(old_config_path):
        shutil.copy2(old_config_path, new_config_path)
        print(f"Copied adapter_config.json from {old_config_path} to {new_config_path}")
    else:
        print(f"Warning: adapter_config.json not found at {old_config_path}")


# =========================
# End2End（LoRA-B：Retrofit & Merge & Export）
# =========================
def retrofit_and_merge_loraB_and_export_full(
        base_model_path: str,
        adapter_paths: List[str],
        val_files: List[str],
        out_adapter_dir: str,
        out_full_pt_path: str,  # .pt path for soft-gate merged full weights
        *,
        # Retrofit hyperparams
        loc_epochs: int = 10,
        loc_lr: float = 5e6,
        loc_l1: float = 10.0,
        loc_l2: float = 10.0,
        loc_sigmoid_bias: float = 5.0,
        loc_sparsity: float = 0.01,
        loc_max_batches: int = 512,
        use_kg_loss: bool = False,
        # data
        loc_sample_cap: int = 512,
        batch_size: int = 4,
):
    """
    Multi-task (based on LoRA-B):
      1) train soft gate (σ(S)) per task
      2) normalize soft gates with overlap
      3) weighted sum on B with soft gates to get merged_B_soft
      4) export:
         - soft-gate merged LoRA adapter (adapter_config.json + adapter_model.safetensors)
         - write soft-gate merged ΔW(B@A)*scale back to base full .pt with numerical verification
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
    base_for_export = T5ForConditionalGeneration.from_pretrained(base_model_path)  # for final full export
    _align_pad_token_id(base_for_export, tokenizer)

    args = Args()
    # 1) validation DataLoader
    dls_loc = []
    for vf in val_files:
        _, data_loc = load_and_cache_gen_data(args, vf, tokenizer, 'dev', sample_cap=loc_sample_cap)
        dls_loc.append(DataLoader(data_loc, sampler=SequentialSampler(data_loc),
                                  batch_size=batch_size, num_workers=2, pin_memory=True))

    # 2) per adapter: build peft_model (not merged), collect τ_B, and retrofit (get soft gate)
    tauB_list: List[Dict[str, torch.Tensor]] = []
    masks_soft: List[Dict[str, torch.Tensor]] = []

    os.makedirs(out_adapter_dir, exist_ok=True)
    # also backup per-task soft gate (optional)
    per_task_gate_dir = os.path.join(out_adapter_dir, "per_task_soft_gates")
    os.makedirs(per_task_gate_dir, exist_ok=True)

    for i, ap in enumerate(adapter_paths):
        logger.info(f"=== Task {i + 1}/{len(adapter_paths)} ===")
        base = T5ForConditionalGeneration.from_pretrained(base_model_path).to(device).eval()
        _align_pad_token_id(base, tokenizer)
        peft = PeftModel.from_pretrained(base, ap, is_trainable=True)  # keep adapter unmerged
        peft.eval()

        # ensure B requires grad (double check)
        for n, p in peft.named_parameters():
            if is_lora_B_key(n):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        tau_B = collect_lora_B_tensors(peft, device)  # on device (for training & quick_eval)
        tauB_list.append({k: v.detach().cpu() for k, v in tau_B.items()})   # CPU version (for merge)

        retrofit = RetrofitB(
            peft_model=peft,
            tokenizer=tokenizer,
            tau_B=tau_B,
            device=device,
            lr=loc_lr,
            l1_strength=loc_l1,
            l2_strength=loc_l2,
            num_epochs=loc_epochs,
            max_batches=loc_max_batches,
            sigmoid_bias=loc_sigmoid_bias,
            sparsity=loc_sparsity,
            use_kg_loss=use_kg_loss,
        )
        _ = retrofit.train_graft(dls_loc[i], log_prefix=f"task#{i + 1}")

        # get soft gate (σ(S)), move to CPU
        gate_soft_i = {k: torch.sigmoid(p).detach().cpu() for k, p in retrofit.mask_params.items()}
        masks_soft.append(gate_soft_i)

        # backup per-task soft gate
        torch.save(gate_soft_i, os.path.join(per_task_gate_dir, f"task{i + 1}_soft_gate.pt"))

        # optional quick sanity: soft / full-τ / all-one
        try:
            retrofit._reset_to_pretrained()
            loss_soft = retrofit._quick_eval_with_mask(gate_soft_i, dls_loc[i], max_batches=16)
            gate_one = {k: torch.ones_like(v) for k, v in gate_soft_i.items()}
            loss_full = retrofit._quick_eval_with_mask(gate_one, dls_loc[i], max_batches=16)
            logger.info(f"[Sanity(B)] task#{i + 1}: loss_soft={loss_soft:.4f} | loss_fullτ={loss_full:.4f}")
        except Exception as e:
            logger.warning(f"[Sanity(B)] task#{i + 1} skipped: {e}")

        del peft, base
        torch.cuda.empty_cache()

    # 3) Merging (soft): overlap normalization → weighted sum on B
    logger.info("[Merging/Soft] average-on-overlaps & accumulate B ...")
    processed_soft = average_overlap_masks(masks_soft)  # normalized per-task soft gates
    # merged_B_soft[n] = Σ_i processed_soft[i][n] ⊙ tauB_list[i][n]
    merged_B_soft = merge_lora_B(
        [{k: v.cpu() for k, v in tauB.items()} for tauB in tauB_list],
        [{k: v.cpu() for k, v in g.items()} for g in processed_soft]
    )

    # 4a) export soft-gate merged LoRA adapter (safetensors)
    reference_adapter = adapter_paths[0]
    soft_adapter_dir = os.path.join(out_adapter_dir, "merged_soft_adapter")

    # 4b) export soft-gate merged full weights (.pt): θ_pre + ΔW(B@A)*scale (module addressing + numerical check)
    # export adapter together in this function, no need for write_lora_adapter_from_AB
    save_merged_full_state_dict_from_loraB(
        base_model=base_for_export,
        reference_adapter_path=reference_adapter,
        merged_B=merged_B_soft,
        out_pt_path=out_full_pt_path,
        do_verify=True,
        base_model_path=base_model_path,
        out_adapter_dir=soft_adapter_dir,
    )
    logger.info(f"[Export] soft-merged full state_dict saved to: {out_full_pt_path}")
    logger.info("Done.")


def _masked_mean_pool_seq(enc_hidden: torch.Tensor, enc_attn_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool encoder hidden states using the encoder attention mask.
    enc_hidden: [B, S, H], enc_attn_mask: [B, S] (1 real, 0 pad) -> returns [B, H]
    """
    w = enc_attn_mask.unsqueeze(-1).type_as(enc_hidden)  # [B,S,1]
    num = (enc_hidden * w).sum(dim=1)  # [B,H]
    den = w.sum(dim=1).clamp_min(1e-6)  # [B,1]
    return num / den


# --------- mse loss (seq2seq variant) ----------
def kg_loss_from_enc_mse_by_avg_conf(
        *,
        labels: torch.Tensor,  # [B,T] with -100 as ignore
        student_logits: torch.Tensor,  # [B,T,V]
        enc_old: torch.Tensor,  # [B,S,H]
        enc_new: torch.Tensor,  # [B,S,H]
        enc_student: torch.Tensor,  # [B,S,H]
        enc_attention_mask: torch.Tensor,  # [B,S] (1 for real tokens, 0 pad)
        logits_old: torch.Tensor,  # [B,T,V] (only to compute OLD confidence)
        ignore_index: int = -100,
        conf_T: float = 2.0,
        conf_thresh: float = 0.6,
        gate_mode: str = "soft",
        old_weight_hi: float = 0.8,
        old_weight_lo: float = 0.2,
        beta: float = 0.08,
        feat_lambda: float = 0.2,
) -> dict:
    """
    Returns a dict similar to previous kg_out, but **kg loss excludes CE**.
    This function is adapted for seq2seq:
      - CE is computed externally
      - compute mean_conf_old per sequence using logits_old and labels (ignore_index used)
      - compute pooled encoder MSE between pooled_student and g_seq * pooled_old + (1-g_seq) * pooled_new
      - L_kg: encoder distillation term to be added to CE
    """
    device = student_logits.device
    B, T, V = student_logits.shape

    # ----- compute mask of valid tokens -----
    valid = (labels != ignore_index)  # [B,T]
    m = valid.view(-1)  # [B*T]

    # ----- OLD average true-label confidence per sequence -----
    if m.any():
        lo = logits_old.view(-1, V)[m]  # [N,V]
        log_q_old = F.log_softmax(lo / conf_T, dim=-1)  # [N,V]
        y_flat = labels.view(-1)[m]
        p_old_true = torch.exp(log_q_old.gather(1, y_flat.view(-1, 1)).squeeze(1))  # [N]
        # back to [B,T] and average over valid positions
        conf_full = torch.zeros(B * T, dtype=torch.float32, device=device)
        conf_full[m] = p_old_true.float()
        conf_bt = conf_full.view(B, T)
        mean_conf_old = (conf_bt * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)  # [B]
    else:
        mean_conf_old = torch.full((B,), 0.0, device=device)

    # ----- Build g_seq (OLD weight per sequence) -----
    if gate_mode == "hard":
        g_seq = (mean_conf_old > conf_thresh).float().unsqueeze(1)  # [B,1]
    elif gate_mode == "piecewise":
        hi = (mean_conf_old > conf_thresh).float().unsqueeze(1)  # [B,1]
        g_seq = hi * old_weight_hi + (1.0 - hi) * old_weight_lo  # [B,1]
    elif gate_mode == "soft":
        s = torch.sigmoid((mean_conf_old - conf_thresh) / beta).unsqueeze(1)  # [B,1]
        g_seq = old_weight_lo + s * (old_weight_hi - old_weight_lo)  # [B,1]
    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")

    # ----- Encoder alignment (pooled MSE) -----
    pooled_old = _masked_mean_pool_seq(enc_old, enc_attention_mask)  # [B,H]
    pooled_new = _masked_mean_pool_seq(enc_new, enc_attention_mask)  # [B,H]
    pooled_stu = _masked_mean_pool_seq(enc_student, enc_attention_mask)  # [B,H]

    enc_mix = g_seq * pooled_old + (1.0 - g_seq) * pooled_new  # [B,H]
    L_feat = F.mse_loss(pooled_stu, enc_mix, reduction="mean")  # scalar

    # kg loss w/o CE
    L_kg = feat_lambda * L_feat

    # For compatibility with original code expectation of some keys:
    # - num_old/num_new/num_none : approximate sequence-level counts
    with torch.no_grad():
        if gate_mode == "hard":
            num_old = int((g_seq.squeeze(1) == 1.0).sum().item())
            num_new = int((g_seq.squeeze(1) == 0.0).sum().item())
            num_none = 0
        else:
            # soft or piecewise -> treat g_seq > conf_thresh as "old" selection for counting
            num_old = int((mean_conf_old > conf_thresh).sum().item())
            num_new = int((mean_conf_old <= conf_thresh).sum().item())
            num_none = 0

    return {
        "L_kg": L_kg,
        "L_feat": L_feat,  # raw mse (not scaled)
        "g_seq_old_weight": g_seq.squeeze(1),  # [B]
        "mean_conf_old": mean_conf_old.detach(),  # [B]
        # compatibility stats
        "num_old": num_old,
        "num_new": num_new,
        "num_none": num_none,
        # keep placeholders that original code might try to fetch
    }


# =========================
# __main__
# =========================
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_adapter_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_full_pt_path), exist_ok=True)

    retrofit_and_merge_loraB_and_export_full(
        base_model_path=args.base_model_path,
        adapter_paths=args.adapter_paths,
        val_files=args.val_files,
        out_adapter_dir=args.out_adapter_dir,
        out_full_pt_path=args.out_full_pt_path,
        loc_epochs=args.loc_epochs,
        loc_lr=args.loc_lr, 
        loc_l1=args.loc_l1,
        loc_l2=args.loc_l2,
        loc_sigmoid_bias=args.loc_sigmoid_bias,
        loc_sparsity=args.loc_sparsity,
        loc_max_batches=args.loc_max_batches,
        loc_sample_cap=args.loc_sample_cap,
        batch_size=args.batch_size,
        use_kg_loss=args.use_kg_loss,
    )
