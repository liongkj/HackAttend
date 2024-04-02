import argparse
import copy
import csv
import logging
import math
import os
import random
import sys
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

# from . import attack
import attack
import wandb
from run_hackattend import (
    convert_examples_to_features_for_multiple_choice,
    convert_examples_to_features_for_sequence_classification,
)
from modelling.bert import BertForMultipleChoice, BertForSequenceClassification
from victim.HackAttend.run_multi_cho import (
    AlphaNliProcessor,
    DreamProcessor,
    HellaswagProcessor,
    Metrics,
    ReclorProcessor,
    SwagProcessor,
)
from victim.HackAttend.run_sent_clas import (
    HansProcessor,
    MnliProcessor,
    PawsqqpProcessor,
    QqpProcessor,
    SstProcessor,
)
from utils import generate_pertubed_attention_mask, query_model, rank_layers_and_heads

# from torch.cuda.amp import autocast


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

attack_args = SimpleNamespace(
    **{
        "top_percentage_in_layer": 0.01,
        "head_tuple_size": (12, 12),
        "layer_tuple_size": (12, 12),
        "best_epoch": "best",
        "aadv": True,
        "grad_strat": "reverse_grad",
        "attn_layer_mask": None,
        "attn_head_mask": None,
        "debug": False,
        "chosen_metrics": "hamming_distance",
    }
)


class HackAttendDataset(Dataset):
    def __init__(
        self,
        task: str,
        asa_model_name: str,
        dataset,
        # train_dataloader: DataLoader,
        asa_data_dir: str = None,
    ):
        # self.data = data
        self.attack_args = attack_args
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     args.load_model_path,
        #     do_lower_case=args.do_lower_case,
        #     # cache_dir=cache_dir,
        #     use_fast=not args.use_slow_tokenizer,
        # )
        self.dataset = dataset
        if not asa_data_dir:
            asa_data_dir = f"data/asa/{task}_{asa_model_name}"
        self.asa_data_dir = asa_data_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        b = self.dataset[idx]
        b = tuple(a.unsqueeze(0) for a in b)
        # push data to gpu
        # adv_examples = hackattend(
        #     batch_i=0,
        #     batch=b,
        #     attack_args=attack_args,
        #     victim=self.asa_model,
        #     device=self.device,
        #     tokenizer=self.tokenizer,
        # )
        # # transfer back to cpu
        # if adv_examples is not None:
        c = tuple(a.squeeze(0) for a in b)
        return tuple([*c] + [c[1]])
        # return None


class TextAttackDataset(Dataset):
    def __init__(
        self,
        task: str,
        args,
        # train_dataloader: DataLoader,
        asa_data_dir: str = None,
        tokenizer: AutoTokenizer = None,
        recipe="bertattack",
        is_clean=False,
    ):
        # self.data = data
        self.args = args
        self.attack_args = attack_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.load_model_path,
            do_lower_case=args.do_lower_case,
            # cache_dir=cache_dir,
            use_fast=not args.use_slow_tokenizer,
        )
        self.is_clean = is_clean
        if not asa_data_dir:
            asa_data_dir = (
                f"data/asa/{recipe}_{task}_log.csv"
                if args.adv_split.find("test") != -1
                else f"data/asa/train_{recipe}_{task}_log.csv"
            )
        # with open(normal_data,"r") as file:
        #     self.normal_data = list()
        with open(asa_data_dir, "r") as file:
            self.data = list(
                csv.reader(
                    file,
                )
            )[1:]
        # remove data that is skipped by the attack
        self.data = [d for d in self.data if d[8].lower() != "skipped"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        "original_text","perturbed_text","original_score","perturbed_score","original_output","perturbed_output","ground_truth_output","num_queries","result_type", choice1-3

        """
        b = self.data[idx]
        if self.is_clean:
            pertubed_text, label = b[0], int(b[6])
        else:
            pertubed_text, label = b[1], int(b[6])
        pertubed_text = pertubed_text.replace("[[", "").replace("]]", "")
        mcq3 = len(b) == 12
        ismcq = len(b) > 9
        tup_list = []
        if ismcq:
            if mcq3:
                for i in range(3, 0, -1):
                    tup_list.append((pertubed_text, b[-i]))
            else:
                for i in range(4, 0, -1):
                    tup_list.append((pertubed_text, b[-i]))
            inputs_dict = self.tokenizer(
                tup_list,
                max_length=self.args.max_seq_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            return (
                inputs_dict["input_ids"],
                inputs_dict["attention_mask"],
                inputs_dict["token_type_ids"],
                torch.tensor(label),
            )
        else:
            inputs_dict = self.tokenizer(
                pertubed_text,
                max_length=self.args.max_seq_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            return (
                inputs_dict["input_ids"][0],
                inputs_dict["attention_mask"][0],
                inputs_dict["token_type_ids"][0],
                torch.tensor(label),
            )

            # return pertubed_text, label


class EvalDataset(Dataset):
    def __init__(
        self,
        task: str,
        args,
        # train_dataloader: DataLoader,
        asa_data_dir: str = None,
        device: str = "cuda",
        tokenizer: AutoTokenizer = None,
        recipe="bertattack",
        is_clean=False,
        normal_data=None,
        skip_skipped=False,
    ):
        # self.data = data
        self.args = args
        self.attack_args = attack_args
        self.tokenizer = tokenizer
        self.is_clean = is_clean
        if not asa_data_dir:
            asa_data_dir = f"data/asa/{recipe}_{task}_log.csv"
        # self.normal_data = normal_data
        # with open(normal_data,"r") as file:
        #     self.normal_data = list()
        with open(asa_data_dir, "r") as file:
            self.data = list(
                csv.reader(
                    file,
                )
            )[1:]

        if skip_skipped:
            self.data = [i for i in self.data if i[8] != "Skipped"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        "original_text","perturbed_text","original_score","perturbed_score","original_output","perturbed_output","ground_truth_output","num_queries","result_type", choice1-3

        """
        b = self.data[idx]
        if self.is_clean:
            pertubed_text, label = b[0], int(b[6])
        else:
            pertubed_text, label = b[1], int(b[6])
        pertubed_text = pertubed_text.replace("[[", "").replace("]]", "")
        mcq3 = len(b) == 12
        ismcq = len(b) > 9
        isnli = len(b) == 10
        tup_list = []
        if ismcq:
            if mcq3:
                for i in range(3, 0, -1):
                    tup_list.append((pertubed_text, b[-i]))
            elif isnli:
                tup_list.append((pertubed_text, b[-1]))
            else:
                for i in range(4, 0, -1):
                    tup_list.append((pertubed_text, b[-i]))
            inputs_dict = self.tokenizer(
                tup_list,
                max_length=self.args.max_seq_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            return (
                inputs_dict["input_ids"],
                inputs_dict["attention_mask"],
                inputs_dict["token_type_ids"],
                torch.tensor(label),
            )
        else:
            return pertubed_text, label


def main(args):
    def select_field(features, field):
        return [
            [choice[field] for choice in feature.choices_features]
            for feature in features
        ]

    def generate_normal_mask(attention_mask):
        attention_mask = attention_mask.clone().detach().unsqueeze(0)
        attention_mask_temp = attention_mask.clone().detach()
        original_attention_mask = attention_mask.clone().detach()
        if attention_mask_temp.dim() == 3:  # for mcq
            # attention_mask_temp[:, :, 0] = 0
            # attention_mask_temp[:, 0, 0] = 0
            # attention_mask_temp[:, 0, :] = 0
            seq_len = attention_mask_temp.shape[-1]

            new_original_attention_mask = (
                (
                    original_attention_mask[:, :, :, np.newaxis]
                    * torch.ones(
                        (1, 1, seq_len, seq_len), device=original_attention_mask.device
                    )
                )
                .squeeze()
                .unsqueeze(1)
                .repeat(1, 12, 1, 1)
            )

            #  mcq [4,12,128,128]
        else:
            # seq 1,12,128,128
            # attention_mask_temp[:, 0] = 0
            new_original_attention_mask = (
                (
                    original_attention_mask[:, np.newaxis, :]
                    * original_attention_mask[:, :, np.newaxis]
                )
                .type(torch.float)[:, None, :, :]
                .repeat(1, 12, 1, 1)
            )

        return new_original_attention_mask[None, :, :, :, :].repeat(12, 1, 1, 1, 1)

    def generate_normal_mask_old(attention_mask):
        attention_mask = attention_mask.clone().detach().unsqueeze(0)
        attention_mask_temp = attention_mask.clone().detach()
        ref_attention_mask_temp = attention_mask.clone().detach()
        original_attention_mask = attention_mask.clone().detach()
        if attention_mask_temp.dim() == 3:  # for mcq
            attention_mask_temp[:, :, 0] = 0
            attention_mask_temp[:, 0, 0] = 0
            ref_attention_mask_temp[:, :, 0] = -1
            ref_attention_mask_temp[:, 0, 0] = -1
            # attention_mask_temp[:, 0, :] = 0
            seq_len = attention_mask_temp.shape[-1]
            new_attention_mask = (
                attention_mask_temp[:, :, :, np.newaxis]
                * torch.ones(
                    (1, 1, seq_len, seq_len), device=attention_mask_temp.device
                )
            ).squeeze()

            ref_new_attention_mask = (
                (
                    ref_attention_mask_temp[:, :, :, np.newaxis]
                    * torch.ones(
                        (1, 1, seq_len, seq_len), device=ref_attention_mask_temp.device
                    )
                )
                .squeeze()
                .unsqueeze(1)
                .repeat(1, 12, 1, 1)
            )
            new_original_attention_mask = (
                (
                    original_attention_mask[:, :, :, np.newaxis]
                    * torch.ones(
                        (1, 1, seq_len, seq_len), device=original_attention_mask.device
                    )
                )
                .squeeze()
                .unsqueeze(1)
                .repeat(1, 12, 1, 1)
            )

            #  mcq [4,12,128,128]
        else:
            # seq 1,12,128,128
            attention_mask_temp[:, 0] = 0
            ref_attention_mask_temp[:, 0] = -1
            # perturbed
            new_attention_mask = (
                (
                    attention_mask_temp[:, np.newaxis, :]
                    * attention_mask_temp[:, :, np.newaxis]
                )
                .type(torch.float)
                .flatten()
            )
            ref_new_attention_mask = (
                (
                    ref_attention_mask_temp[:, np.newaxis, :]
                    * ref_attention_mask_temp[:, :, np.newaxis]
                )
                .type(torch.float)[:, None, :, :]
                .repeat(1, 12, 1, 1)
            )
            new_original_attention_mask = (
                (
                    original_attention_mask[:, np.newaxis, :]
                    * original_attention_mask[:, :, np.newaxis]
                )
                .type(torch.float)[:, None, :, :]
                .repeat(1, 12, 1, 1)
            )

        return (
            new_original_attention_mask[None, :, :, :, :].repeat(12, 1, 1, 1, 1),
            ref_new_attention_mask[None, :, :, :, :].repeat(12, 1, 1, 1, 1),
        )
        # attn_mask_by_layer_list.append(new_attention_mask)

    def collate_fn(batch, p=args.mask_rate):
        # check if batch[-1] is None, if so, it is normal data
        new_batch = []
        for exm in batch:
            if exm is not None:
                inputs = exm[0]
                # normal_mask, ref_mask = generate_normal_mask(exm[1])
                # normal_mask = generate_normal_mask(exm[1])
                normal_mask = exm[1]
                if exm[-1].shape == exm[-2].shape and p > 0:  # normal example
                    # if exm[1].shape == exm[-1].shape:
                    # normal data
                    ori_shape = None
                    if inputs.dim() == 2:
                        ori_shape = inputs.shape
                        inputs = inputs.view(-1)
                    pertubed_mask = torch.clone(normal_mask).detach()
                    probability_matrix = torch.full(normal_mask.shape, p)
                    special_tokens_mask = tokenizer.get_special_tokens_mask(
                        inputs, already_has_special_tokens=True
                    )
                    special_tokens_mask = torch.tensor(
                        special_tokens_mask, dtype=torch.bool
                    )

                    if ori_shape is not None:
                        special_tokens_mask = special_tokens_mask.reshape(ori_shape)
                        # special_tokens_mask = generate_normal_mask(
                        #     special_tokens_mask
                        # )
                    neq_tokens_mask = (normal_mask == 0).cpu()

                    probability_matrix.masked_fill_(
                        special_tokens_mask,
                        value=0.0,
                    )
                    probability_matrix.masked_fill_(
                        neq_tokens_mask,
                        value=0.0,
                    )
                    masked_indices = torch.bernoulli(probability_matrix).bool()

                    # probability_matrix = torch.full(labels.shape, p)
                    pertubed_mask[masked_indices] = torch.tensor(0)
                    exm = tuple(list(exm[:-1] + (pertubed_mask,)))
                else:
                    exm = tuple(list(exm[:-1] + (normal_mask,)))
                new_batch.append(exm)
        # collate function here

        transposed_batch = list(zip(*new_batch))

        # Stack the elements into tensors
        stacked_batch = []
        for item in transposed_batch:
            if torch.is_tensor(item[0]):
                # tack expects each tensor to be equal size, but got [1, 128] at entry 0 and [128] at entry 1 TODO
                stacked_item = torch.stack(item, dim=0)
            else:
                stacked_item = item
            stacked_batch.append(stacked_item)

        return stacked_batch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        device2 = torch.device("cuda:1")
        device = torch.device("cuda:0")
    else:
        device2 = device
    n_gpu = torch.cuda.device_count()
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device,
            n_gpu,
            "Unsupported",
            "On" if args.fp16 else "Off",
        )
    )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    task_name = args.task_name.lower()
    processors = {
        "swag": SwagProcessor,
        "dream": DreamProcessor,
        "hellaswag": HellaswagProcessor,
        "alphanli": AlphaNliProcessor,
        "sst-2": SstProcessor,
        "qqp": QqpProcessor,
        "mnli": MnliProcessor,
        "reclor": ReclorProcessor,
        "paws-qqp": PawsqqpProcessor,
        "hans": HansProcessor,
    }
    mcq = ["alphanli", "dream", "hellaswag", "reclor"]
    if task_name not in processors:
        raise ValueError(f"Task not found: {task_name}")
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.load_model_path, num_labels=num_labels)
    model: AutoModel = None
    converter = None
    if args.task_name in mcq:
        # if args.model_type == "roberta":
        # model = RobertaForMultipleChoice
        # elif args.model_type == "bert":
        model = BertForMultipleChoice
        # model = AutoModelForMultipleChoice.from_pretrained(
        #     args.load_model_path,        # )
        converter = convert_examples_to_features_for_multiple_choice
    elif args.task_name in ["sst-2", "mnli", "paws-qqp", "hans"]:
        # if args.model_type == "roberta":
        # model = RobertaForSequenceClassification
        # else:
        model = BertForSequenceClassification
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     args.load_model_path,        # )
        converter = convert_examples_to_features_for_sequence_classification

    assert model is not None and converter is not None
    tokenizer = AutoTokenizer.from_pretrained(
        args.load_model_path,
        do_lower_case=args.do_lower_case,
        # cache_dir=cache_dir,
        use_fast=not args.use_slow_tokenizer,
    )
    # model = model.from_pretrained(
    #     args.load_model_path,
    #     num_labels=num_labels if args.task_name not in ["hans"] else 3,
    # ) #load fine-tuned model
    model = model.from_pretrained("bert-base-uncased")
    model.config = config
    # model = model(config=config) # load pretrained model
    #
    if args.do_train:
        train_examples = processor.get_train_examples(
            os.path.join(args.data_dir, task_name)
        )
        train_features = converter(
            train_examples, label_list, args.max_seq_length, tokenizer
        )
        if args.task_name in mcq:
            all_input_ids = torch.tensor(
                select_field(train_features, "input_ids"), dtype=torch.long
            )
            all_input_mask = torch.tensor(
                select_field(train_features, "input_mask"), dtype=torch.long
            )
            all_segment_ids = torch.tensor(
                select_field(train_features, "segment_ids"), dtype=torch.long
            )
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.long
            )
        elif args.task_name in ["sst-2", "mnli", "paws-qqp"]:
            all_input_ids = torch.tensor(
                [f.input_ids for f in train_features], dtype=torch.long
            )
            all_input_mask = torch.tensor(
                [f.input_mask for f in train_features], dtype=torch.long
            )
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in train_features], dtype=torch.long
            )
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features],
                dtype=torch.long if task_name != "sts-b" else torch.float,
            )
        elif args.task_name in ["hans"]:
            # hans means train in mnli and test in hans
            mnli_train_examples = processors["mnli"]().get_train_examples(
                os.path.join(args.data_dir, "mnli")
            )
            mnli_label_list = processors["mnli"]().get_labels()
            mnli_train_features = converter(
                mnli_train_examples, mnli_label_list, args.max_seq_length, tokenizer
            )
            all_input_ids = torch.tensor(
                [f.input_ids for f in mnli_train_features], dtype=torch.long
            )
            all_input_mask = torch.tensor(
                [f.input_mask for f in mnli_train_features], dtype=torch.long
            )
            all_segment_ids = torch.tensor(
                [f.segment_ids for f in mnli_train_features], dtype=torch.long
            )
            all_label_ids = torch.tensor(
                [f.label_id for f in mnli_train_features], dtype=torch.long
            )
            # normal_data = TensorDataset(
            #     all_input_ids,
            #     all_input_mask,
            #     all_segment_ids,
            #     all_label_ids,
            #     torch.ones_like(all_label_ids),
            # )
        else:
            raise ValueError(f"Task not found: {task_name}")

        normal_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
        )
        # normal_data = torch.utils.data.Subset(normal_data, list(range(0, 16)))
        if args.adversarial:
            # normal_data_split_1,normal_data_split_2 = torch.utils.data.Subset(normal_data,range(0,int(len(normal_data)/2))), torch.utils.data.Subset(normal_data,range(int(len(normal_data)/2),len(normal_data)))
            if args.adv_split.find("test") != -1:
                # S-attend
                if args.mask_rate > 0:
                    adversarial_data2 = HackAttendDataset(
                        task=args.task_name,
                        asa_model_name=args.model_type,
                        dataset=normal_data,
                    )
                    del normal_data
                    normal_data = TensorDataset(
                        all_input_ids,
                        all_input_mask,
                        all_segment_ids,
                        all_label_ids,
                        torch.ones_like(all_label_ids),
                    )
                    # train_data = ConcatDataset([normal_data, adversarial_data2])
                    train_data = ConcatDataset([normal_data])
                else:
                    normal_data = TensorDataset(
                        all_input_ids,
                        all_input_mask,
                        all_segment_ids,
                        all_label_ids,
                    )
                    train_data = ConcatDataset([normal_data])
                # if args.task_name in ["hans"]:
                #     train_data = ConcatDataset([mnli_data])
            else:
                # ADA training
                reci = args.adv_split.split("_")[-1]
                adversarial_data2 = TextAttackDataset(
                    task=args.task_name,
                    args=args,
                    # tokenizer=tokenizer,
                    recipe=reci,
                )

                # train_data = ConcatDataset([adversarial_data2])
                train_data = ConcatDataset([normal_data, adversarial_data2])
            if args.test_run:
                train_data = torch.utils.data.Subset(train_data, list(range(0, 16)))
            # train_data = ConcatDataset([normal_data])
            # train_data = ConcatDataset([normal_data, adversarial_data2,adversarial_data1])

        else:
            train_data = normal_data

        # eval_data = torch.utils.data.Subset(eval_data, range(43, 44))
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=(
                collate_fn
                if (args.adv_split.find("test") != -1 and args.mask_rate > 0)
                else None
            ),
            num_workers=1,
            pin_memory=True,
        )
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = int(
                args.num_train_epochs * num_update_steps_per_epoch
            )
        else:
            args.num_train_epochs = math.ceil(
                args.max_train_steps / num_update_steps_per_epoch
            )

        if args.adversarial:
            logger.info("***** Running asa training *****")
        else:
            logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %f", args.max_train_steps)
        model.to(device)

        # Prepare optimizer
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate
        )
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.max_train_steps * args.warmup_proportion,
            num_training_steps=args.max_train_steps,
        )

        scaler = GradScaler() if args.fp16 else None
        if args.do_eval:

            def eval_collate_fn(batch):
                if isinstance(batch[0][0], torch.Tensor):
                    transposed_batch = list(zip(*batch))

                    # Stack the elements into tensors
                    stacked_batch = []
                    for item in transposed_batch:
                        if torch.is_tensor(item[0]):
                            # tack expects each tensor to be equal size, but got [1, 128] at entry 0 and [128] at entry 1 TODO
                            stacked_item = torch.stack(item, dim=0)
                        else:
                            stacked_item = item
                        stacked_batch.append(
                            stacked_item.squeeze(1)
                            if stacked_item.dim() == 3
                            else stacked_item
                        )

                    return stacked_batch
                texts = [item[0] for item in batch]

                # Initialize the tokenizer
                # if already tensor

                # Tokenize the texts using batch_encode_plus
                tokenized_batch = tokenizer.batch_encode_plus(
                    texts, padding=True, truncation=True, return_tensors="pt"
                )

                # Extract the input IDs, attention masks, and labels
                input_ids = tokenized_batch["input_ids"]
                attention_masks = tokenized_batch["attention_mask"]
                segment_ids = tokenized_batch["token_type_ids"]
                labels = torch.tensor([item[1] for item in batch])

                # Return the tokenized inputs, attention masks, and labels as a dictionary
                return (input_ids, attention_masks, segment_ids, labels)
                # return {
                #     'input_ids': input_ids,
                #     'attention_masks': attention_masks,
                #     'labels': labels
                # }

            eval_examples = processor.get_test_examples(
                os.path.join(args.data_dir, task_name)
            )

            eval_features = converter(
                eval_examples, label_list, args.max_seq_length, tokenizer
            )
            if args.task_name in mcq:
                all_input_ids = torch.tensor(
                    select_field(eval_features, "input_ids"), dtype=torch.long
                )
                all_input_mask = torch.tensor(
                    select_field(eval_features, "input_mask"), dtype=torch.long
                )
                all_segment_ids = torch.tensor(
                    select_field(eval_features, "segment_ids"), dtype=torch.long
                )
                all_label_ids = torch.tensor(
                    [f.label_id for f in eval_features], dtype=torch.long
                )
            elif args.task_name in ["sst-2", "mnli", "paws-qqp"]:
                all_input_ids = torch.tensor(
                    [f.input_ids for f in eval_features], dtype=torch.long
                )
                all_input_mask = torch.tensor(
                    [f.input_mask for f in eval_features], dtype=torch.long
                )
                all_segment_ids = torch.tensor(
                    [f.segment_ids for f in eval_features], dtype=torch.long
                )
                all_label_ids = torch.tensor(
                    [f.label_id for f in eval_features],
                    dtype=torch.long if task_name != "sts-b" else torch.float,
                )

            # eval_data = TensorDataset(
            #     all_input_ids, all_input_mask, all_segment_ids, all_label_ids
            # )
            # eval_sampler = SequentialSampler(eval_data)
            # eval_dataloader = DataLoader(
            #     eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
            # )

            ##################
            # clean
            ###############
            eval_data = EvalDataset(
                recipe="textfooler",
                task=args.task_name,
                tokenizer=tokenizer,
                args=args,
                is_clean=True,
                normal_data=normal_data if (args.task_name in mcq) else None,
            )
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn if (args.task_name not in mcq) else None,
            )

            ############
            # BertAttack
            #############
            eval_data_bertattack = EvalDataset(
                recipe="bertattack",
                task=args.task_name,
                tokenizer=tokenizer,
                args=args,
                # normal_data = normal_data if (args.task_name in mcq) else None
                normal_data=train_examples,
            )
            eval_sampler_bertattack = SequentialSampler(eval_data_bertattack)
            eval_dataloader_bertattack = DataLoader(
                eval_data_bertattack,
                sampler=eval_sampler_bertattack,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn if (args.task_name not in mcq) else None,
            )

            # ###################
            # textfooler
            #######################
            eval_data_textfooler = EvalDataset(
                recipe="textfooler",
                task=args.task_name,
                tokenizer=tokenizer,
                normal_data=normal_data if (args.task_name in mcq) else None,
                args=args,
            )
            eval_sampler_textfooler = SequentialSampler(eval_data_textfooler)
            eval_dataloader_textfooler = DataLoader(
                eval_data_textfooler,
                sampler=eval_sampler_textfooler,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn if (args.task_name not in mcq) else None,
            )

        progress_bar = tqdm(range(args.max_train_steps))
        global_step = 0
        best_result_bertattack = 0.0
        best_result = 0.0
        best_result_textfooler = 0.0

        for epoch in range(int(args.num_train_epochs)):
            model.train()
            train_loss = 0
            num_train_examples = 0
            train_steps = 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                if len(batch) == 5:
                    (
                        input_ids,
                        input_mask,
                        segment_ids,
                        label_ids,
                        attention_mask,
                    ) = batch
                else:
                    (
                        input_ids,
                        attention_mask,
                        segment_ids,
                        label_ids,
                    ) = batch

                if args.fp16:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=segment_ids,
                            labels=label_ids,
                            args=args,
                        )
                        # adv_outputs = model()

                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=segment_ids,
                        labels=label_ids,
                        args=args,
                    )
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss += loss.item()
                num_train_examples += input_ids.size(0)
                train_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    progress_bar.update(1)
                wandb.log({"train_loss": loss.item()})
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

                if global_step >= args.max_train_steps:
                    break

            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(
                args.output_dir, "{}_pytorch_model.bin".format(epoch)
            )
            if os.path.exists(args.output_dir) is False:
                os.makedirs(args.output_dir)

            torch.save(model_to_save.state_dict(), output_model_file)
            loss = train_loss / train_steps if args.do_train else None
            wandb.log({"loss": loss})
            if args.do_eval:
                logger.info("***** Running evaluation *****")
                # logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                model.eval()
                evaluate_dataloader(
                    args,
                    device,
                    model,
                    eval_dataloader,
                    loss=loss,
                    global_step=global_step,
                    epoch=epoch,
                    model_to_save=model_to_save,
                    best_result=best_result,
                    recipe="clean",
                )  # look at clean acc
                evaluate_dataloader(
                    args,
                    device,
                    model,
                    eval_dataloader_bertattack,
                    loss=loss,
                    global_step=global_step,
                    epoch=epoch,
                    model_to_save=model_to_save,
                    best_result=best_result_bertattack,
                    recipe="bertattack",
                )  # this shows robust acc
                evaluate_dataloader(
                    args,
                    device,
                    model,
                    eval_dataloader_textfooler,
                    loss=loss,
                    global_step=global_step,
                    epoch=epoch,
                    model_to_save=model_to_save,
                    best_result=best_result_textfooler,
                    recipe="textfooler",
                )  # this shows robust acc


def evaluate_dataloader(
    args,
    device,
    model,
    eval_dataloader_bertattack,
    loss,
    global_step,
    epoch,
    model_to_save,
    recipe,
    best_result,
):
    eval_loss = 0
    num_eval_examples = 0
    eval_steps = 0
    all_predictions, all_labels = [], []
    for batch in tqdm(eval_dataloader_bertattack, desc="Evaluation"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=label_ids,
                args=args,
            )
            tmp_eval_loss = outputs[0]
            logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        eval_loss += tmp_eval_loss.mean().item()
        all_predictions.extend(np.argmax(logits, axis=1).squeeze().tolist())
        all_labels.extend(label_ids.squeeze().tolist())
        num_eval_examples += input_ids.size(0)
        eval_steps += 1
    if args.task_name in ["hans"]:
        # change neutral (2) to non-entailment (0)
        print(sum(all_predictions))
        all_predictions = [0 if x == 2 else x for x in all_predictions]
        print(sum(all_predictions))
    eval_loss = eval_loss / eval_steps
    eval_acc = Metrics.acc(all_predictions, all_labels) * 100
    result = {
        f"global_step": global_step,
        f"{recipe}/eval_loss": eval_loss,
        f"{recipe}/eval_acc": eval_acc,
    }
    wandb.log(result)
    if result[f"{recipe}/eval_acc"] > best_result:
        best_epoch = epoch
        best_result = result[f"{recipe}/eval_acc"]
        output_model_file = os.path.join(
            args.output_dir, f"{recipe}_best_pytorch_model.bin"
        )
        torch.save(model_to_save.state_dict(), output_model_file)

    output_eval_file = os.path.join(args.output_dir, f"{recipe}_eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write(
            "Epoch %s: global step = %s | loss = %.3f | eval score = %.2f | eval loss = %.3f\n"
            % (
                str(epoch),
                str(result["global_step"]),
                loss,
                result[f"{recipe}/eval_acc"],
                result[f"{recipe}/eval_loss"],
            )
        )
        for key in sorted(result.keys()):
            logger.info("Epoch: %s,  %s = %s", str(epoch), key, str(result[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/glue/",
        help="Directory to contain the input data for all tasks.",
    )
    parser.add_argument("--best_epoch", default=0, help="Victim best epoch")

    parser.add_argument(
        "--task_name", type=str, default="SST-2", help="Name of the training task."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="Type of BERT-like architecture, e.g. BERT, ALBERT.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default="bert-base-uncased",
        help="Specific model path to load, e.g. bert-base-uncased.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="asa_model/",
        help="Directory to output predictions and checkpoints.",
    )
    # Training config.
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to evaluate on the dev set."
    )
    parser.add_argument(
        "--eval_on",
        type=str,
        default="test",
        help="Whether to evaluate on the test set.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="A slow tokenizer will be used if passed.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum total input sequence length after word-piece tokenization.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Total batch size for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides training epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="Scheduler type for learning rate warmup.",
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.06,
        help="Proportion of training to perform learning rate warmup for.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="L2 weight decay for training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward pass.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Whether to use mixed precision."
    )

    parser.add_argument(
        "--test_run", action="store_true", help="Whether to use mixed precision."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument("--aadv", default=None)

    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument(
        "--adv_split",
        type=str,
        default="None",
        choices=["test", "train_bertattack", "train_textfooler"],
    )
    parser.add_argument("--mask_rate", type=float, default=0.2)
    args = parser.parse_args()
    if args.load_model_path is None:
        args.load_model_path = f"model/{args.task_name}_{args.model_type}"

    if args.adversarial:
        assert (
            args.adv_split != "None"
        ), "Please select a split for adversarial training"

    if args.output_dir.find(args.task_name) == -1:
        args.output_dir = (
            f"{args.output_dir}{args.task_name}_{args.model_type}_{args.mask_rate}/"
        )
    tags = [args.task_name, args.model_type]
    if args.adversarial:
        tags.append("adv_train")
    wandb.init(project="asa_train", config=args, tags=tags)
    main(args)
