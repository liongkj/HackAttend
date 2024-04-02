from __future__ import absolute_import, division, print_function

import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.metrics as mtc
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm

# Import required libraries
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

import attack
import wandb
from modelling.bert import BertForMultipleChoice, BertForSequenceClassification

# from transformers.models.bert.modeling_bert import BertForMultipleChoice
from modelling.utils import divergence
from victim.HackAttend.run_multi_cho import DreamProcessor  # InputFeatures,
from victim.HackAttend.run_multi_cho import (
    AlphaNliProcessor,
    HellaswagProcessor,
    Metrics,
    ReclorProcessor,
    SwagProcessor,
)
from victim.HackAttend.run_sent_clas import MnliProcessor, QqpProcessor, SstProcessor
from utils import query_model, rank_layers_and_heads, save_results

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
)


def convert_examples_to_features_for_multiple_choice(
    examples, label_list, max_seq_length, tokenizer=None
):
    class InputFeatures(object):
        def __init__(self, choices_features, label_id):
            if tokenizer:
                self.choices_features = [
                    {
                        "input_ids": input_ids,
                        "input_mask": input_mask,
                        "segment_ids": segment_ids,
                    }
                    for input_ids, input_mask, segment_ids in choices_features
                ]
            else:
                self.choices_features = choices_features
            self.label_id = label_id

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in tqdm(enumerate(examples), desc="Converting examples to features"):
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
            zip(example.contexts, example.endings)
        ):
            text_a = context
            if example.question.find("_") != -1:
                # This is for cloze questions.
                text_b = example.question.replace("_", ending)
            else:
                text_b = f"{example.question} {ending}"
            if tokenizer:
                encoded_inputs = tokenizer(
                    text_a,
                    text_b,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )

                choices_features.append(
                    (
                        encoded_inputs["input_ids"],
                        encoded_inputs["attention_mask"],
                        encoded_inputs["token_type_ids"],
                    )
                )
            else:
                choices_features.append((text_a, text_b))

        label_id = label_map[example.label]

        # if i < 2:
        #     logger.info("*** Example ***")
        #     logger.info("guid: {}".format(example.guid))
        #     for choice_idx, (
        #         input_ids,
        #         attention_mask,
        #         token_type_ids,
        #     ) in enumerate(choices_features):
        #         tokens = tokenizer.convert_ids_to_tokens(input_ids)
        #         logger.info(f'tokens: {" ".join(tokens)}')
        #         logger.info("choice: {}".format(choice_idx))
        #         logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
        #         logger.info(
        #             "attention_mask: {}".format(" ".join(map(str, attention_mask)))
        #         )
        #         logger.info(
        #             "token_type_ids: {}".format(" ".join(map(str, token_type_ids)))
        #         )
        #         logger.info("label: {}".format(label_id))

        features.append(
            InputFeatures(choices_features=choices_features, label_id=label_id)
        )

    return features


def convert_examples_to_features_for_sequence_classification(
    examples, label_list, max_seq_length, tokenizer=None
):
    class InputFeatures(object):
        def __init__(self, data, label_id):
            if tokenizer is None:
                self.features = data
            else:
                self.input_ids = input_ids
                self.input_mask = input_mask
                self.segment_ids = segment_ids
            self.label_id = label_id

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in enumerate(examples):
        if tokenizer is not None:
            if example.text_b:
                encoded_inputs = tokenizer(
                    example.text_a,
                    example.text_b,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )
                input_ids = encoded_inputs["input_ids"]
                input_mask = encoded_inputs["attention_mask"]
                segment_ids = encoded_inputs["token_type_ids"]
                # tokens = tokenizer.convert_ids_to_tokens(input_ids)
            else:
                encoded_inputs = tokenizer(
                    example.text_a,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )
                input_ids = encoded_inputs["input_ids"]
                input_mask = encoded_inputs["attention_mask"]
                segment_ids = encoded_inputs["token_type_ids"]
                # tokens = tokenizer.convert_ids_to_tokens(input_ids)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            data = encoded_inputs
        else:
            if example.text_b:
                data = (example.text_a, example.text_b)
            else:
                data = (example.text_a,)

        if len(label_list) == 1:
            label_id = example.label
        else:
            label_id = label_map[example.label]
        # if i < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % example.guid)
        #     logger.info("tokens: %s" % " ".join(tokens))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
            InputFeatures(
                data=data,
                label_id=label_id,
            )
        )

    return features


# def tokenize_and_convert_to_input_ids(example) -> DataLoader:
from utils import mcq


def main(args):
    cache_dir = args.cache_dir
    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
    }

    if task_name not in processors:
        raise ValueError(f"Task not found: {task_name}")
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.load_model_path, num_labels=num_labels)
    dataset = processor.get_test_examples(os.path.join(args.data_dir, task_name))

    if args.task_name in mcq:
        # if args.model_type == "roberta":
        # victim = RobertaForMultipleChoice(config)
        # elif args.model_type == "bert":
        victim = BertForMultipleChoice(config)
        converter = convert_examples_to_features_for_multiple_choice
    elif args.task_name in ["sst-2", "mnli"]:
        # if args.model_type == "roberta":
        # victim = RobertaForSequenceClassification(config)
        # else:
        victim = BertForSequenceClassification(config)
        converter = convert_examples_to_features_for_sequence_classification
    else:
        victim = None
        converter = None
    assert victim is not None and converter is not None
    victim.load_state_dict(
        torch.load(
            os.path.join(args.load_model_path, f"{args.best_epoch}_pytorch_model.bin")
        ),
        strict=False,
    )

    def print_example(example, ori, adv):
        input_ids, input_mask, segment_ids, label_ids = example
        # decode input_ids
        tokens = tokenizer.batch_decode(input_ids[0].cpu())
        tokens = [a.replace("[PAD]", "") for a in tokens]
        logger.info(tokens[ori] + "\n" + tokens[adv] + "\n\n")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.load_model_path.find("roberta") != -1:
        args.load_model_path = "roberta-base"
    elif args.load_model_path.find("bert") != -1:
        args.load_model_path = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        args.load_model_path,
        do_lower_case=args.do_lower_case,
        cache_dir=cache_dir,
        use_fast=not args.use_slow_tokenizer,
    )

    eval_features = converter(dataset, label_list, args.max_seq_length, tokenizer)
    if args.task_name in mcq:

        def select_field(features, field):
            return [
                [choice[field] for choice in feature.choices_features]
                for feature in features
            ]

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
    elif args.task_name in ["sst-2", "mnli"]:
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
    else:
        raise ValueError(f"Task not found: {task_name}")

    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )

    # eval_data = torch.utils.data.Subset(eval_data, range(43, 44))
    eval_sampler = SequentialSampler(eval_data)
    dataset = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    # load weights from trained model

    # apply the function to the dataset to tokenize and convert to input IDs
    logger.info("***** Running attack *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    victim.eval()
    victim.to(device)

    num_eval_examples = len(dataset)
    original_correct_predictions = 0
    success_attack = 0
    failed_attack = 0
    skipped_attacks = 0
    total_num_query = defaultdict(int)
    mask_percentage_list = []
    evaluate_score_list = []
    attack_layer_dict = defaultdict()
    attack_head_dict = defaultdict()
    start_time = time.time()
    success_query = 0
    attack_idx_list = []
    is_random_mask = args.random_mask  # randomly choose heads and layers
    random_units_mask = args.random_unit  # random mask units (non-gair)
    print(f"{is_random_mask=}")
    print(f"{random_units_mask=}")
    stat_list = []

    pd_headers = [
        "sa_{}".format(i)
        for i in range(
            victim.config.num_hidden_layers * victim.config.num_attention_heads
        )
    ]
    sa_matrix_attacked = pd.DataFrame(columns=pd_headers)
    for batch_i, batch in tqdm(
        enumerate(dataset),
        desc="attack",
        total=len(dataset),
    ):
        (
            original_pred_is_correct,
            original_conf_score,
            ori_prediction,
            last_hidden_states,
            original_attn_scores,
        ) = query_model(
            model=victim,
            batch=batch,
            attn_mask_by_layer_list=None,
            args=args,
            device=device,
        )

        if args.statistics:
            input_ids, input_mask, segment_ids, label_ids = batch
            # get sum of input mask
            if input_mask.dim() == 3:
                input_mask_sum = input_mask.sum(dim=2).sum() / input_mask.size(1)
            else:
                input_mask_sum = input_mask.sum()
            stat_list.append(input_mask_sum.item())
            # get sum of attention mask
        else:
            if not original_pred_is_correct:  # skip wrong predictions
                skipped_attacks += 1
                continue
            original_correct_predictions += 1

            # Choose the top n filters for each category, useless when is_random_mask=True
            chosen_layer, chosen_head = rank_layers_and_heads(
                args, victim, batch, is_random_mask, device=device
            )

            # Ensure that the chosen layers and heads are not None
            assert chosen_layer is not None and chosen_head is not None

            # Generate all combinations of chosen layers and heads
            if args.attack_strat.find("greedy") != -1:
                has_succeed = attack.greedy_attack(
                    chosen_layer=chosen_layer,
                    args=args,
                    victim=victim,
                    batch=batch,
                    device=device,
                    batch_i=batch_i,
                    sa_matrix_pd=sa_matrix_attacked,
                    original_attn_scores=original_attn_scores,
                    tokenizer=tokenizer,
                    success_attack=success_attack,
                    total_num_query=total_num_query,
                    attack_idx_list=attack_idx_list,
                    evaluate_score_list=evaluate_score_list,
                    mask_percentage_list=mask_percentage_list,
                    attack_head_dict=attack_head_dict,
                    attack_layer_dict=attack_layer_dict,
                    failed_attack=failed_attack,
                    random_units_mask=random_units_mask,
                    is_random_mask=is_random_mask,
                )

            else:
                has_succeed = attack.score_attack(
                    args,
                    victim=victim,
                    success_attack=success_attack,
                    failed_attack=failed_attack,
                    total_num_query=total_num_query,
                    mask_percentage_list=mask_percentage_list,
                    evaluate_score_list=evaluate_score_list,
                    attack_layer_dict=attack_layer_dict,
                    attack_head_dict=attack_head_dict,
                    attack_idx_list=attack_idx_list,
                    is_random_mask=is_random_mask,
                    batch_i=batch_i,
                    batch=batch,
                    original_attn_scores=original_attn_scores,
                    chosen_layer=chosen_layer,
                    chosen_head=chosen_head,
                    device=device,
                    tokenizer=tokenizer,
                )
            if has_succeed:
                success_attack += 1
                success_query += total_num_query[batch_i]
                if success_attack == 100:
                    end_time = time.time()
                    wandb.log(
                        {
                            "time_100_samples": round(end_time - start_time, 1),
                            "query_100_samples": success_query,
                        }
                    )
            else:
                failed_attack += 1
            wandb.log({"run_asr": success_attack / (batch_i - skipped_attacks + 1)})
    if args.statistics:
        with open("stat_list.txt", "a+") as f:
            # write the task name, average length, max length and min length
            f.write(
                f"{task_name}, avg: {sum(stat_list) / len(stat_list)}, max:{max(stat_list)}, min:{min(stat_list)}\n"
            )
            # f.write(str(sum(stat_list) / len(stat_list)))
            return
    end_time = time.time()
    duration = end_time - start_time
    save_results(
        num_eval_examples,
        original_correct_predictions,
        success_attack,
        failed_attack,
        skipped_attacks,
        total_num_query,
        mask_percentage_list,
        evaluate_score_list,
        attack_layer_dict,
        attack_head_dict,
        attack_idx_list,
        duration=duration,
    )
    # sa_matrix_attacked.to_csv(f"results/{args.task_name}_attacked_sa")
    # wandb.log({"sa_attacked_table":wandb.Table(dataframe=sa_matrix_attacked)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data config.
    parser.add_argument("--statistics", action="store_true", default=False)
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
        default=None,
        help="Specific model path to load, e.g. bert-base-uncased.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../cache/",
        help="Directory to store the pre-trained language models downloaded from s3.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model/",
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
        default="dev",
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
        default=128,
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
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--layer_mask_type",
        type=str,
        default=None,
        help="mask SA layer strategy. put 1 at the i location to mask the i-th layer. each place holder is one layer, so bert base is 100000000000",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--top_n_in_layer",
        default=0,
        type=int,
        help="mask top n in each layer",
    )

    parser.add_argument(
        "--random_mask",
        default=False,
        type=int,
        help="mask random heads and layers",
    )

    parser.add_argument(
        "--random_unit",
        default=False,
        type=int,
        help="mask units randomly",
    )

    parser.add_argument(
        "--random_strat",
        default="importance_gair",
        type=str,
        help="random strategy used. importance_gair is used in the paper",
    )

    parser.add_argument("--top_layer_n", default=None, help="placeholder")

    # parser.add_argument("--top_head_n", default=None, help="placeholder")
    parser.add_argument("--attn_head_mask", type=str, default=None)
    parser.add_argument("--attn_layer_mask", type=str, default=None)

    parser.add_argument("--notes", type=str, default=None)

    parser.add_argument("--aadv", type=int, default=1)

    parser.add_argument("--num_tries", type=int, default=1)

    # parser.add_argument("--top_n_index_target", type=int, default=2)
    parser.add_argument("--run_i", type=int, default=1)
    parser.add_argument(
        "--head_tuple_size",
        type=str,
        default="12,12",
        help="(min tuple size, max tuple size)",
    )

    parser.add_argument(
        "--layer_tuple_size",
        type=str,
        default="12,12",
        help="(min tuple size, max tuple size)",
    )

    parser.add_argument(
        "--chosen_metrics",
        type=str,
        default="hamming_distance",
        choices=["frobenius_norm", "hamming_distance"],
    )

    parser.add_argument(
        "--max_combinations",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--grad_strat",
        type=str,
        default="reverse_grad",
        choices=["grad", "reverse_grad", "magnitude"],
    )

    parser.add_argument(
        "--attack_strat",
        type=str,
        default="agg_sorted",
    )
    parser.add_argument("--top_percentage_in_layer", type=float, default=0.01)
    args = parser.parse_args()
    # if args.task_name in ["reclor"]:
    # args.max_seq_length = 256
    if args.random_strat == "importance_gair":
        # pass
        args.random_mask = False
        args.random_unit = False
    else:
        ran_strat = args.random_strat.split("_")
        layer_strat = ran_strat[0]
        unit_strat = ran_strat[1]
        args.random_unit = unit_strat == "random"
        args.random_mask = layer_strat == "random"
    if args.load_model_path is None:
        args.load_model_path = f"model/{args.task_name}_{args.model_type}"
    if args.aadv in [0, False, "0", "false", "False"]:
        args.aadv = False
    if args.aadv in [1, True, "1", "true", "True"]:
        args.aadv = True
        if args.grad_strat is None:
            args.grad_strat = "reverse_grad"
    args.head_tuple_size = tuple(int(a) for a in args.head_tuple_size.split(","))
    args.layer_tuple_size = tuple(int(a) for a in args.layer_tuple_size.split(","))
    assert args.top_percentage_in_layer >= 0.0 and args.top_percentage_in_layer <= 1.0
    # check if not both top_n in layer and top_percentage_in_layer are set
    assert not (
        args.top_n_in_layer > 0 and args.top_percentage_in_layer > 0.0
    ), "top_n_in_layer and top_percentage_in_layer cannot be set at the same time"
    assert (
        args.head_tuple_size[1] >= args.head_tuple_size[0]
    ), "max_tuple_size must be >= min_tuple_size"
    assert (
        args.layer_tuple_size[1] >= args.layer_tuple_size[0]
    ), "max_tuple_size must be >= min_tuple_size"

    assert args.head_tuple_size[0] >= 1, "min_tuple_size must be >= 1"
    assert args.layer_tuple_size[0] >= 1, "min_tuple_size must be >= 1"
    args.attn_head_mask = None if args.attn_head_mask == "None" else args.attn_head_mask
    args.attn_layer_mask = (
        None if args.attn_layer_mask == "None" else args.attn_layer_mask
    )
    my_tag = []
    if args.attn_layer_mask:
        my_tag.append("auto_layer")
    if args.attn_head_mask:
        my_tag.append("auto_head")
    if args.aadv:
        my_tag.append("aadv")
    if args.head_tuple_size[0] == args.head_tuple_size[1]:
        my_tag.append("only_one_size")
    if args.attn_layer_mask is not None and args.attn_head_mask is not None:
        head_mask_count = [*args.attn_head_mask].count("1")
        layer_mask_count = [*args.attn_layer_mask].count("1")
        args.head_tuple_size = (
            min(head_mask_count, layer_mask_count),
            max(head_mask_count, layer_mask_count),
        )
    if args.random_mask == 1:
        my_tag.append("baseline")
    # assert args.aadv == (1 if args.grad_strat != None else 0)
    wandb.init(
        "asa",
        notes=args.notes,
        config={
            "top_layer_n": args.top_layer_n,
            "attn_head_mask": args.attn_head_mask or "auto",
            "attn_layer_mask": args.attn_layer_mask or "auto",
            "use_gradient": args.aadv,
            "head_tuple_size": args.head_tuple_size,
            "layer_tuple_size": args.layer_tuple_size,
        },
        tags=my_tag,
    )
    main(args=args)
