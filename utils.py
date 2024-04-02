from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tabulate
import torch
from matplotlib.patches import Rectangle

import wandb
from wandb import plot as wandb_plot
import logging

# Attention scores
import torch

from tabulate import tabulate

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

mcq = ["alphanli", "dream", "hellaswag", "reclor"]
import copy


def visualize_attention(
    attention_mask,
    source_sentence,
    layer,
    head,
    bs=1,
    seq_len=128,
    target_sentence=None,
    wandb=None,
    type="",
    args=None,
):
    if target_sentence is None:
        target_sentence = source_sentence
    if bs > 1:
        fig, axs = plt.subplots(nrows=1, ncols=bs, figsize=(5 * bs, 5))
    else:
        # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig, axs = plt.subplots(figsize=(5, 5))
    num_masked = 0
    total = 0
    for i in range(bs):
        try:
            pad_idx = (
                target_sentence[i * seq_len : i * seq_len + seq_len].index("[PAD]") - 1
            )
        except Exception as e:
            pad_idx = seq_len * i
        attention_mask_specific = (
            attention_mask[i, :, :][:pad_idx, :pad_idx].detach().cpu()
        )
        total += attention_mask_specific.shape[0]

        source_sentence_specific = source_sentence[i * seq_len : i * seq_len + seq_len][
            :pad_idx
        ]
        target_sentence_specific = target_sentence[i * seq_len : i * seq_len + seq_len][
            :pad_idx
        ]
        attn_matrix = np.zeros(
            (len(source_sentence_specific), len(target_sentence_specific))
        )

        # Fill in the matrix with attention weights and mark slots red if the attention mask is 0
        for x in range(len(source_sentence_specific)):
            for j in range(len(target_sentence_specific)):
                attn_matrix[x][j] = attention_mask_specific[x][j]
                # Check if the attention mask is 0 at this position
                if attention_mask_specific[x][j] == 0 or type in [
                    "attn_weights",
                    "attn_scores",
                ]:
                    num_masked += 1
                    # If the attention mask is 0, mark the corresponding slots in the heatmap with a light color
                    if bs > 1:
                        axs[i].add_patch(
                            Rectangle(
                                (j - 0.5, x - 0.5),
                                1,
                                1,
                                fill=True,
                                alpha=0.3,
                                color="red",
                            )
                        )
                        # Add a label above the mask to show the corresponding word
                        axs[i].text(
                            j,
                            x - 0.3,
                            f"{target_sentence_specific[j]} {source_sentence_specific[x]}",
                            fontsize=8,
                            horizontalalignment="center",
                        )
                    else:
                        axs.add_patch(
                            Rectangle(
                                (j - 0.5, x - 0.5),
                                1,
                                1,
                                fill=True,
                                alpha=0.3,
                                color="red",
                            )
                        )
                        # Add a label above the mask to show the corresponding word
                        axs.text(
                            j,
                            x - 0.3,
                            f"{target_sentence_specific[j]} {source_sentence_specific[x]}",
                            fontsize=8,
                            horizontalalignment="center",
                        )

        # Create a heatmap to visualize the attention matrix
        if bs > 1:
            axs[i].imshow(attn_matrix, cmap="ocean", aspect="auto")
            axs[i].set_xticks(range(len(target_sentence_specific)))
            axs[i].set_xticklabels(target_sentence_specific, rotation=90, fontsize=8)
            axs[i].set_yticks(range(len(source_sentence_specific)))
            axs[i].set_yticklabels(source_sentence_specific, fontsize=8)
            axs[i].set_title(f"Layer {layer} Head {head} Batch {i}")
        else:
            axs.imshow(attn_matrix, cmap="ocean", aspect="auto")
            axs.set_xticks(range(len(target_sentence_specific)))
            axs.set_xticklabels(target_sentence_specific, rotation=90, fontsize=8)
            axs.set_yticks(range(len(source_sentence_specific)))
            axs.set_yticklabels(source_sentence_specific, fontsize=8)
            axs.set_title(f"Layer {layer} Head {head} Batch {i}")
    if args.debug:
        logger.info(
            f"Number of masked positions: {num_masked} out of {total**2}. {num_masked/total**2} percent in Head {head} in layer {layer}."
        )
    plt.savefig(f"{type}_attention.png")
    plt.tight_layout()
    wandb.log({"attention": plt})
    plt.close()


import random


def query_model(
    model,
    batch,
    args=None,
    attn_mask_by_layer_list=None,
    head_mask=None,
    no_grad=False,
    device=torch.device("cpu"),
    visualize=False,
):
    """ """
    if len(batch) == 4:
        input_ids, input_mask, segment_ids, label_ids = [t.to(device) for t in batch]
    else:
        input_ids, input_mask, segment_ids, label_ids, _ = [t.to(device) for t in batch]
    # if attn_mask_by_layer_list:
    #     args.attn_mask_by_layer_list =
    conf_score = None
    # if args.aadv and attn_mask_by_layer_list is None and not no_grad:
    if visualize or (args.aadv and attn_mask_by_layer_list is None and not no_grad):
        outputs = model(
            input_ids=input_ids,
            attention_mask=input_mask
            if attn_mask_by_layer_list is None
            else attn_mask_by_layer_list,
            token_type_ids=segment_ids if segment_ids is not None else None,
            labels=label_ids if label_ids is not None else None,
            # layer_mask=attn_msk,
            args=args,
            output_hidden_states=True,
            output_attentions=True,
            # head_mask=head_mask,
            return_dict=False,
        )
        logits = outputs[1]
        loss = outputs[0]
        loss.backward()
        # prediction = torch.argmax(logits[0].softmax(-1).cpu(), axis=0)
        hidden_states = outputs[2]
        # attn_score = outputs[3]
        if args.grad_strat == "magnitude":
            attn_score = [torch.abs(a.grad) for a in outputs[4]]
        else:
            attn_score = [
                a.grad * (-1 if args.grad_strat == "reverse_grad" else 1)
                for a in outputs[4]
            ]  # list [3,12,128,128]
        attn_score = (attn_score, outputs[3])
        model.zero_grad()
    else:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_mask
                if attn_mask_by_layer_list is None
                else attn_mask_by_layer_list,
                token_type_ids=segment_ids if segment_ids is not None else None,
                labels=label_ids if label_ids is not None else None,
                # layer_mask=attn_msk,
                head_mask=head_mask,
                args=args,
                output_hidden_states=True,
                output_attentions=True,
            )
            logits = outputs[1]
            hidden_states = outputs[2]
            attn_score = outputs[3]
    logits_softmax = logits.softmax(-1).cpu()
    prediction = torch.argmax(logits_softmax, axis=-1)
    conf_score = logits_softmax[torch.arange(len(prediction)), prediction]
    assert label_ids.shape == conf_score.shape
    label_ids = label_ids.cpu()
    del outputs
    del model
    return (
        prediction.eq(label_ids),
        conf_score if conf_score is not None else None,
        prediction,
        hidden_states,
        attn_score,
    )


def generate_atten_mask(
    mask_type: Union[int, str],
    num_hidden_layers: int,
) -> List[torch.Tensor]:
    """Generate attention mask for each layer."""
    # prepare layer mask
    atten_mask_list = []
    if mask_type is not None:
        # logger.info("  Layer mask type = %s", mask_type)
        if mask_type in ["all", "-1"]:
            # 0 to close attention
            atten_mask_list.extend(torch.tensor(1) for _ in range(num_hidden_layers))
        else:
            try:
                if isinstance(mask_type, tuple):
                    mask_list = list(mask_type)
                    atten_mask_list.extend(
                        torch.tensor(int(a in mask_list))
                        for a in range(num_hidden_layers)
                    )
                    return atten_mask_list
                elif isinstance(int(mask_type), int):
                    # change to string
                    mask_list = [*str(mask_type)]
                    if len(mask_list) == num_hidden_layers:
                        # split the string to list
                        mask_list = [int(i) for i in mask_list]
                        assert (
                            len(mask_list) == num_hidden_layers
                        ), "number of layers not equal to models' no. of layers"
                        atten_mask_list.extend(torch.tensor(i) for i in mask_list)
                    else:
                        atten_mask_list.extend(
                            torch.tensor(int(a == mask_type))
                            for a in range(num_hidden_layers)
                        )
            except Exception as e:
                raise e
    assert atten_mask_list
    return atten_mask_list


def save_results(
    num_eval_examples,
    original_correct_predictions,
    success_attack,
    failed_attack,
    skipped_attacks,
    total_num_query,
    mask_percentage_list,
    pertube_score_list,
    attack_layer_dict,
    attack_head_dict,
    attack_idx_list,
    duration: float,
):
    """
    save results to wandb and print table result to console
    """
    attack_layer_table = wandb.Table(
        columns=["Layer", "Count"],
        data=[[b, attack_layer_dict[b]] for b in attack_layer_dict],
    )
    attack_head_table = wandb.Table(
        columns=["Head", "Count"],
        data=[[b, attack_head_dict[b]] for b in attack_head_dict],
    )

    num_queries_table = wandb.Table(
        columns=["Sample", "Num Queries"],
        data=[[b, total_num_query[b]] for b in total_num_query],
    )
    logger.info(pertube_score_list)

    attack_success_rate = success_attack / original_correct_predictions
    average_num_queries = sum(total_num_query.values()) / original_correct_predictions
    average_pertubation_score = (
        sum(pertube_score_list) / len(pertube_score_list) if pertube_score_list else 0
    )
    masked_percentange = [a[0] for a in mask_percentage_list]
    total_attacked_attn_seq = [a[1] for a in mask_percentage_list]
    total_attn_seq = [a[2] for a in mask_percentage_list]
    total_non_pad_tokens = [a[3] for a in mask_percentage_list]
    total_block_count = [a[4] for a in mask_percentage_list]

    masked_precentage_table = wandb.Table(
        columns=[
            "sample_id",
            "masked_percentage",
            "masked_count",
            "total_count",
            "total_non_pad_tokens",
            "total_block_count",
        ],
        data=[
            [
                attack_idx_list[i],
                masked_percentange[i],
                total_attacked_attn_seq[i],
                total_attn_seq[i],
                total_non_pad_tokens[i],
                total_block_count[i],
            ]
            for i in range(len(masked_percentange))
        ],
    )
    wandb.log(
        {
            "attacked_layer": wandb_plot.bar(
                attack_layer_table, "Layer", "Count", title="Attacked Layer"
            ),
            "attacked_head": wandb_plot.bar(
                attack_head_table, "Head", "Count", title="Attacked Head"
            ),
            "num_queries": wandb_plot.bar(
                num_queries_table, "Sample", "Num Queries", title="Num Queries"
            ),
            "masked_percentage": wandb_plot.histogram(
                masked_precentage_table,
                "masked_percentage",
                title="Masked Percentage Histogram",
            ),
        }
    )
    avg_mask_percent = (
        sum(masked_percentange) / len(masked_percentange) if masked_percentange else 0
    )

    # print(f"Total number of queries: {sum(total_num_query.values())}")
    # print(f"Original score: {original_correct_predictions / num_eval_examples}")
    print(f"Total number of examples: {num_eval_examples}")
    print(total_num_query)
    print(f"Avg. Mask percentage: {avg_mask_percent}")
    print(f"Attack success rate: {attack_success_rate * 100:.2f}%")
    print(f"Average number of queries: {average_num_queries:.2f}")
    print(f"Total time taken: {duration:2f} seconds")

    attack_layer_table = [
        ["attack_result", ""],
        ["number of successful attacks:", success_attack],
        ["Number of failed attacks", failed_attack],
        ["Number of skipped attacks", skipped_attacks],
        [
            "original_accuracy (%)",
            (original_correct_predictions / num_eval_examples) * 100,
        ],
        [
            "Robust Accuracy (%)",
            (failed_attack / num_eval_examples) * 100,
        ],
        ["Average Masked Percentage", float(f"{avg_mask_percent}")],
        [
            "Average Attacked Mask on Non-Padding Tokens",
            sum(total_attacked_attn_seq) / len(total_attacked_attn_seq)
            if total_attacked_attn_seq
            else 0,
        ],
        [
            "Average Total Attacked Mask of Selected Head Layers",
            sum(total_attn_seq) / len(total_attn_seq) if total_attn_seq else 0,
        ],
        ["Average Perturbation Score of Attention Mask", average_pertubation_score],
        [
            "Average Non-Padding Token Count of All Head Layers",
            sum(total_non_pad_tokens) / len(total_non_pad_tokens)
            if total_non_pad_tokens
            else 0,
        ],
        ["Attack Success rate (%)", attack_success_rate * 100],
        ["Average number of queries", average_num_queries],
        ["Total time taken (seconds)", duration],
        [
            "Average time taken per query (seconds)",
            (duration) / sum(total_num_query.values()),
        ],
    ]

    result_dict = {}
    for row in attack_layer_table[1:]:
        key = row[0]
        value = row[1]
        result_dict[key] = value
    # wandb.log(total_num_query)
    wandb.log(result_dict)
    print(tabulate(attack_layer_table, headers="firstrow", tablefmt="fancy_grid"))
    log_results(result_dict=result_dict)


def get_masked_percentage(args):
    masked_percentage = []
    # a = 0.001
    # while a < args.top_percentage_in_layer:
    #     masked_percentage.append(a)
    #     a *= 2
    masked_percentage.append(args.top_percentage_in_layer)
    return masked_percentage

def generate_pertubed_attention_mask_old(
    attn_scores,
    atten_layer_mask_list: List,
    atten_head_mask_list: List,
    random_units_mask: bool = False,
    args=None,
    batch=None,
    batch_i=None,
    device=None,
    tokenizer=None,
) -> List[torch.Tensor]:
    """
    return list of attn_mask, len of attn_mask should be equal to the models num. of layers
    """
    attn_mask_by_layer_list = []
    ori_attn_mask_by_layer_list = []
    total_attacked_attn_seq = 0
    total_attn_seq = 0
    attention_mask_metrics = {
        "frobenius_norm": lambda a, b: torch.norm(a - b, p="fro"),
        "hamming_distance": lambda a, b: torch.sum(
            (~torch.eq(a, b)).clone().detach(), dtype=torch.int32
        ).item()
        / (len(atten_layer_mask_list) * len(atten_head_mask_list)),
    }
    # get number of attention masks (batch_size * num_heads * seq_len * seq_len)
    top_k_mask = 0
    num_attention_seq = 0
    # print(atten_layer_mask_list)
    block_count = 0
    for layer_i, attack_layer_n in enumerate(atten_layer_mask_list):
        attention_mask = batch[1].to(device)  # [1, 3,128]
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
                (
                    attention_mask_temp[:, :, :, np.newaxis]
                    * torch.ones(
                        (1, 1, seq_len, seq_len), device=attention_mask_temp.device
                    )
                )
                .squeeze()
                .flatten()
            )

            ref_new_attention_mask = (
                (
                    ref_attention_mask_temp[:, :, :, np.newaxis]
                    * torch.ones(
                        (1, 1, seq_len, seq_len), device=ref_attention_mask_temp.device
                    )
                )
                .squeeze()
                .flatten()
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
                .repeat(1, len(atten_head_mask_list), 1, 1)
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
                .type(torch.float)
                .flatten()
            )

            # unpertubed
            new_original_attention_mask = (
                (
                    original_attention_mask[:, np.newaxis, :]
                    * original_attention_mask[:, :, np.newaxis]
                )
                .type(torch.float)[:, None, :, :]
                .repeat(1, len(atten_head_mask_list), 1, 1)
            )

        # ].flatten()
        # (
        #     original_attention_mask.squeeze(0)
        #     .unsqueeze(-1)
        #     .repeat(1, 1, attention_mask.shape[-1])
        #     .flatten()
        # )
        num_attention_seq = attention_mask_temp.sum() ** 2
        if args.debug:
            logger.info(f"layer {layer_i} has {num_attention_seq} attention seq")
        # attention_mask shape [1,3,128] -> [3,128,128]

        if args.aadv:
            attn_scores_layer_grad = attn_scores[0][layer_i]  # gradient
            attn_scores_layer_score = attn_scores[1][layer_i]  # attention_score
        else:
            attn_scores_layer_grad = attn_scores[layer_i]  # attention_score
            attn_scores_layer_score = None
        if not top_k_mask and args.top_percentage_in_layer > 0:
            # attention_mask_temp = attention_mask.type(
            #     torch.float32
            # ).clone()  # [1,3,128]
            # mask out [CLS] token
            top_k_mask = (
                int(
                    (num_attention_seq * args.top_percentage_in_layer)
                    / attention_mask_temp.shape[-2]
                )
                or 1
            )

        elif not top_k_mask and args.top_n_in_layer > 0:
            top_k_mask = args.top_n_in_layer
        assert top_k_mask > 0
        if batch_i == 0 and args.debug:
            logger.info(
                f"top_k_mask: {args.top_percentage_in_layer} of {num_attention_seq} is {top_k_mask}"
            )
        if attention_mask.dim() == 3:
            is_not_padding_mask = (attention_mask == 1).squeeze(0)
            # [3,1,1,128]
        else:
            # [1,128]
            is_not_padding_mask = attention_mask == 1
        is_not_padding_mask = is_not_padding_mask[:, None, None, :]

        if attack_layer_n == 1:  # choose which layer to attack
            if atten_head_mask_list in [
                "-1",
                "all",
            ]:  # mean across head, not running now
                attn_score_grad = attn_scores_layer_grad.mean(1)  # [3,128,128]
                # mask out [CLS] token
                attn_score_grad[:, :, 0] = 0
                attn_score_grad[:, 0, :] = 0
                attn_score_grad[:, 0, 0] = 0
                temp_overall_attn_score = attn_score_grad.view(-1)  # [3*128*128]
                (
                    _,
                    max_attention_positions,
                ) = temp_overall_attn_score.topk(top_k_mask, -1)

                attacked_attention_mask = torch.ones_like(temp_overall_attn_score)
                attacked_attention_mask[max_attention_positions] = 0
                total_attacked_attn_seq += len(max_attention_positions.flatten())
                attacked_attention_mask = attacked_attention_mask.view_as(
                    attn_score_grad
                )  # [3,128,128]
                ori_attention_mask = torch.ones_like(attacked_attention_mask)
                # attention_mask shape [1,3,128] -> [3,1,128]
                nonzero_padding = (attention_mask == 1).squeeze(0)[:, None, :]
                attacked_attention_mask = attacked_attention_mask * nonzero_padding
                ori_attention_mask = ori_attention_mask * nonzero_padding
                # torch.save(attacked_attention_mask,"mask.pt")
            else:  # select head
                head_attn_mask = []
                ori_head_attn_mask = []

                for h_i, attack_head_i in enumerate(atten_head_mask_list):
                    # for h_i in range(victim.config.num_attention_heads):
                    attn_score_grad = attn_scores_layer_grad[
                        :, h_i, :, :
                    ]  # [3,128,128]
                    attn_score_score = None
                    if attn_scores_layer_score != None:
                        attn_score_score = attn_scores_layer_score[
                            :, h_i, :, :
                        ]  # [3,128,128]
                        attn_score_score_flatten = (
                            attn_score_score.flatten() * new_attention_mask
                        )
                        attn_score_score = attn_score_score_flatten.view(
                            attn_score_score.shape[0], -1
                        )

                    temp_overall_attn_score = (
                        attn_score_grad.flatten() * new_attention_mask
                    )  # [3*128*128]
                    temp_overall_attn_score = temp_overall_attn_score.view(
                        attn_score_grad.shape[0], -1
                    )
                    # attention_mask shape [1,3,128] -> [3,128,128]
                    is_not_padding = (new_attention_mask == 1).nonzero().flatten()

                    ori_attention_mask_flatten = torch.zeros_like(
                        attn_score_grad
                    ).flatten()
                    ori_attention_mask_flatten[is_not_padding] = 1
                    ori_attention_mask = ori_attention_mask_flatten.view_as(
                        attn_score_grad
                    )
                    ori_attention_mask[:, : attention_mask.sum(), 0] = 1
                    ori_attention_mask[:, 0, 0] = 1
                    # ori_attention_mask[:, 0, : attention_mask.sum()] = 1

                    if attack_head_i == 1:
                        # print(f"layer:{layer_i} head:{h_i}")
                        block_count += 1
                        if random_units_mask:
                            # if not using GAIR
                            # randomly chooses SA units
                            max_attention_positions = torch.randint(
                                temp_overall_attn_score.shape[-1],
                                (temp_overall_attn_score.shape[0], top_k_mask),
                            )
                        else: #GAIR
                            if torch.all(
                                temp_overall_attn_score == 0
                            ):  # not all heads has gradient, use score or skip
                                max_attention_positions = []
                                # get non padding positions from temp_overall_attn_score
                                # attention_mask [1,3,128] -> [3x128x128]
                                if attn_score_score != None:
                                    (
                                        max_attention_val,
                                        max_attention_positions,
                                    ) = attn_score_score.topk(top_k_mask, -1)
                                else:
                                    max_attention_positions = []
                                    max_attention_val = []
                                # max_attention_positions = []
                            else:
                                (
                                    max_attention_val,
                                    max_attention_positions,
                                ) = temp_overall_attn_score.topk(top_k_mask, -1)
                                # only get non zero positions
                        attacked_attention_mask = new_attention_mask.clone()
                        if args.debug:
                            assert torch.all(
                                torch.isin(max_attention_positions, is_not_padding)
                            ), "max_attention_positions should be in is_not_padding"
                            assert torch.all(
                                attacked_attention_mask[max_attention_positions] == 1
                            )
                        total_attacked_attn_seq += len(
                            max_attention_positions.flatten()
                        )
                        # attacked_attention_mask = attacked_attention_mask.view_as(
                        #     attn_score_grad
                        # )  # [3,128,128]
                        attacked_attention_mask = attacked_attention_mask.view(
                            max_attention_positions.shape[0], -1
                        )
                        attacked_attention_mask[:, max_attention_positions] = 0
                        attacked_attention_mask = attacked_attention_mask.view_as(
                            attn_score_grad
                        )
                        # restore cls tokens attn
                        ref_new_attention_mask = ref_new_attention_mask.view_as(
                            attacked_attention_mask
                        )
                        attacked_attention_mask[ref_new_attention_mask == -1] = 1
                        if batch_i < 1 and args.debug:
                            logger.info(max_attention_positions[:10])
                            logger.info(
                                f"current masked count: {total_attacked_attn_seq}"
                            )
                            # logger.info(f"total masked count: {total_attn_seq}")
                            # logger.info(
                            #     f"masked percentage: {total_attacked_attn_seq/total_attn_seq}"
                            # )
                            logger.info(
                                f"masking {top_k_mask} out of {len(is_not_padding)}"
                            )
                            logger.info(f"Layer {layer_i} Head {h_i}")
                    else:
                        ori_attention_mask[0][0][
                            : attention_mask.sum()
                        ] = 1  # restore cls token attn
                        attacked_attention_mask = ori_attention_mask

                    ori_head_attn_mask.append(ori_attention_mask)
                    head_attn_mask.append(attacked_attention_mask)
                    # print(diff)
                # head attention mask for all heads in ith layer
                attacked_attention_mask = torch.stack(
                    head_attn_mask, 1
                )  # [3,12,128,128]
                ori_attention_mask = torch.stack(
                    ori_head_attn_mask, 1
                )  # [3,12,128,128]
            attention_mask = attacked_attention_mask
            attn_mask_by_layer_list.append(
                attacked_attention_mask
            )  # seq 1,12,128,128 mcq [16,12,4,12,128,128]
            ori_attn_mask_by_layer_list.append(ori_attention_mask)
        else:  # not attack layer, use original attention mask
            # new_attention_mask = torch.ones_like(
            #     attn_scores_layer_grad
            # )  # [3,12,128,128]
            # # attention_mask shape [1,3,128] -> [3,1,1,128]
            # is_not_padding = is_not_padding_mask.expand_as(
            #     attn_scores_layer_grad
            # )  # [3,12,128,128]
            # new_attention_mask = new_attention_mask * is_not_padding

            attn_mask_by_layer_list.append(new_original_attention_mask)
            ori_attn_mask_by_layer_list.append(new_original_attention_mask)
    # temp_attacked = torch.stack(attn_mask_by_layer_list).view(-1)
    # temp_ori = torch.stack(ori_attn_mask_by_layer_list).view(-1)
    # mask_metrics = attention_mask_metrics[args.chosen_metrics](
    #     temp_attacked,
    #     temp_ori,
    # )
    # mask_metrics = 0
    mask_metrics = total_attacked_attn_seq
    total_attn_seq = block_count * num_attention_seq
    masked_percentange = total_attacked_attn_seq / total_attn_seq
    # print(f"hamming {mask_metrics}")
    # print(f"masked unit {total_attacked_attn_seq}")
    # print(f"non_padding_seq {total_attn_seq}")
    # print(f"block_count {block_count}")
    total_non_pad_tokens = (
        num_attention_seq * len(atten_layer_mask_list) * len(atten_head_mask_list)
    )
    if batch_i < 1 and args.debug:
        logger.info(f"total_attn_seq {total_attn_seq}")
        logger.info(f"total_attacked_attn_seq {total_attacked_attn_seq}")
        logger.info(f"masked_percentage {masked_percentange}")
        logger.info(f"total_non_pad_tokens {total_non_pad_tokens}")
        logger.info(f"masked_metric {mask_metrics}")
    # mask_metrics1 = attention_mask_metrics["frobenius_norm"](
    #     torch.stack(attn_mask_by_layer_list),
    #     torch.stack(ori_attn_mask_by_layer_list),
    # )
    if batch_i < 10 and args.debug:
        count = 0
        try:
            bs = batch[0].shape[-2]
            seq_len = batch[0].shape[-1]
            # while count < 1:
            for i, ii in enumerate(atten_layer_mask_list):
                for j, jj in enumerate(atten_head_mask_list):
                    if ii == 1 and jj == 1:
                        source_sentence = tokenizer.convert_ids_to_tokens(
                            batch[0].flatten().tolist()
                        )  # [128]
                        # visualize_attention(
                        #     attn_scores[0][i][:, j, :, :],
                        #     bs=bs,
                        #     seq_len=seq_len,
                        #     source_sentence=source_sentence,
                        #     head=j,
                        #     layer=i,
                        #     wandb=wandb,
                        #     type="attn_weights",
                        # )
                        visualize_attention(
                            attn_mask_by_layer_list[i][:, j, :, :],
                            bs=bs,
                            seq_len=seq_len,
                            source_sentence=source_sentence,
                            head=j,
                            layer=i,
                            wandb=wandb,
                            type="masked",
                            args=args,
                        )
                        # visualize_attention(
                        #     attn_scores[1][i][:, j, :, :],
                        #     bs=bs,
                        #     seq_len=seq_len,
                        #     source_sentence=source_sentence,
                        #     head=j,
                        #     layer=i,
                        #     wandb=wandb,
                        #     type="attn_scores",
                        # )
                        if not args.debug:
                            count += 1
                            break
                    if not args.debug and count > 0:
                        break
                if not args.debug and count > 0:
                    break
        except Exception as e:
            logger.info(e)
            logger.info("visualize attention failed")
    return (
        [a.detach() for a in attn_mask_by_layer_list],
        (
            masked_percentange,
            total_attacked_attn_seq,
            total_attn_seq,
            total_non_pad_tokens,
            block_count,
        ),
        mask_metrics,
    )



def generate_pertubed_attention_mask(
    attn_scores,
    to_mask_sa_matrix: torch.Tensor,
    random_units_mask: bool = False,
    args=None,
    batch=None,
    batch_i=None,
    device=None,
    tokenizer=None,
) -> List[torch.Tensor]:
    """
    return list of attn_mask, len of attn_mask should be equal to the models num. of layers
    """
    attn_mask_by_layer_list = []
    ori_attn_mask_by_layer_list = []
    total_attacked_attn_seq = 0
    total_attn_seq = 0
    # get number of attention masks (batch_size * num_heads * seq_len * seq_len)
    top_k_mask = 0
    num_attention_seq = 0
    # print(atten_layer_mask_list)
    block_count = torch.sum(to_mask_sa_matrix)
    for layer_i, attack_layer_n in enumerate(to_mask_sa_matrix):
        attention_mask = batch[1].to(device)  # [1, 3,128]
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
                (
                    attention_mask_temp[:, :, :, np.newaxis]
                    * torch.ones(
                        (1, 1, seq_len, seq_len), device=attention_mask_temp.device
                    )
                )
                .squeeze()
                .flatten()
            )

            ref_new_attention_mask = (
                (
                    ref_attention_mask_temp[:, :, :, np.newaxis]
                    * torch.ones(
                        (1, 1, seq_len, seq_len), device=ref_attention_mask_temp.device
                    )
                )
                .squeeze()
                .flatten()
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
                .repeat(1, to_mask_sa_matrix.shape[1], 1, 1)
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
                .type(torch.float)
                .flatten()
            )

            # unpertubed
            new_original_attention_mask = (
                (
                    original_attention_mask[:, np.newaxis, :]
                    * original_attention_mask[:, :, np.newaxis]
                )
                .type(torch.float)[:, None, :, :]
                .repeat(1, to_mask_sa_matrix.shape[1], 1, 1)
            )

        # ].flatten()
        # (
        #     original_attention_mask.squeeze(0)
        #     .unsqueeze(-1)
        #     .repeat(1, 1, attention_mask.shape[-1])
        #     .flatten()
        # )
        num_attention_seq = attention_mask_temp.sum() ** 2
        if args.debug:
            logger.info(f"layer {layer_i} has {num_attention_seq} attention seq")
        # attention_mask shape [1,3,128] -> [3,128,128]

        if args.aadv:
            attn_scores_layer_grad = attn_scores[0][layer_i]  # gradient
            attn_scores_layer_score = attn_scores[1][layer_i]  # attention_score
        else:
            attn_scores_layer_grad = attn_scores[layer_i]  # attention_score
            attn_scores_layer_score = None
        if not top_k_mask and args.top_percentage_in_layer > 0:
            # attention_mask_temp = attention_mask.type(
            #     torch.float32
            # ).clone()  # [1,3,128]
            # mask out [CLS] token
            top_k_mask = (
                int(
                    (num_attention_seq * args.top_percentage_in_layer)
                    / attention_mask_temp.shape[-2]
                )
                or 1
            )

        elif not top_k_mask and args.top_n_in_layer > 0:
            top_k_mask = args.top_n_in_layer
        assert top_k_mask > 0
        if batch_i == 0 and args.debug:
            logger.info(
                f"top_k_mask: {args.top_percentage_in_layer} of {num_attention_seq} is {top_k_mask}"
            )
        if attention_mask.dim() == 3:
            is_not_padding_mask = (attention_mask == 1).squeeze(0)
            # [3,1,1,128]
        else:
            # [1,128]
            is_not_padding_mask = attention_mask == 1
        is_not_padding_mask = is_not_padding_mask[:, None, None, :]
        if not torch.allclose(attack_layer_n,torch.tensor(0.0)):
        # if attack_layer_n == 1:  # choose which layer to attack
            # if atten_head_mask_list in [
            #     "-1",
            #     "all",
            # ]:  # mean across head, not running now
            #     attn_score_grad = attn_scores_layer_grad.mean(1)  # [3,128,128]
            #     # mask out [CLS] token
            #     attn_score_grad[:, :, 0] = 0
            #     attn_score_grad[:, 0, :] = 0
            #     attn_score_grad[:, 0, 0] = 0
            #     temp_overall_attn_score = attn_score_grad.view(-1)  # [3*128*128]
            #     (
            #         _,
            #         max_attention_positions,
            #     ) = temp_overall_attn_score.topk(top_k_mask, -1)

            #     attacked_attention_mask = torch.ones_like(temp_overall_attn_score)
            #     attacked_attention_mask[max_attention_positions] = 0
            #     total_attacked_attn_seq += len(max_attention_positions.flatten())
            #     attacked_attention_mask = attacked_attention_mask.view_as(
            #         attn_score_grad
            #     )  # [3,128,128]
            #     ori_attention_mask = torch.ones_like(attacked_attention_mask)
            #     # attention_mask shape [1,3,128] -> [3,1,128]
            #     nonzero_padding = (attention_mask == 1).squeeze(0)[:, None, :]
            #     attacked_attention_mask = attacked_attention_mask * nonzero_padding
            #     ori_attention_mask = ori_attention_mask * nonzero_padding
            #     # torch.save(attacked_attention_mask,"mask.pt")
            # else:  # select head
            head_attn_mask = []
            ori_head_attn_mask = []

            for h_i, attack_head_i in enumerate(attack_layer_n):
                # for h_i in range(victim.config.num_attention_heads):
                attn_score_grad = attn_scores_layer_grad[
                    :, h_i, :, :
                ]  # [3,128,128]
                attn_score_score = None
                if attn_scores_layer_score != None:
                    attn_score_score = attn_scores_layer_score[
                        :, h_i, :, :
                    ]  # [3,128,128]
                    attn_score_score_flatten = (
                        attn_score_score.flatten() * new_attention_mask
                    )
                    attn_score_score = attn_score_score_flatten.view(
                        attn_score_score.shape[0], -1
                    )

                temp_overall_attn_score = (
                    attn_score_grad.flatten() * new_attention_mask
                )  # [3*128*128]
                temp_overall_attn_score = temp_overall_attn_score.view(
                    attn_score_grad.shape[0], -1
                )
                # attention_mask shape [1,3,128] -> [3,128,128]
                is_not_padding = (new_attention_mask == 1).nonzero().flatten()

                ori_attention_mask_flatten = torch.zeros_like(
                    attn_score_grad
                ).flatten()
                ori_attention_mask_flatten[is_not_padding] = 1
                ori_attention_mask = ori_attention_mask_flatten.view_as(
                    attn_score_grad
                )
                ori_attention_mask[:, : attention_mask.sum(), 0] = 1
                ori_attention_mask[:, 0, 0] = 1
                # ori_attention_mask[:, 0, : attention_mask.sum()] = 1

                if attack_head_i == 1:
                    # print(f"layer:{layer_i} head:{h_i}")
                    # block_count += 1
                    if random_units_mask:
                        # if not using GAIR
                        # randomly chooses SA units
                        max_attention_positions = torch.randint(
                            temp_overall_attn_score.shape[-1],
                            (temp_overall_attn_score.shape[0], top_k_mask),
                        )
                    else: #GAIR
                        if torch.all(
                            temp_overall_attn_score == 0
                        ):  # not all heads has gradient, use score or skip
                            max_attention_positions = []
                            # get non padding positions from temp_overall_attn_score
                            # attention_mask [1,3,128] -> [3x128x128]
                            if attn_score_score != None:
                                (
                                    max_attention_val,
                                    max_attention_positions,
                                ) = attn_score_score.topk(top_k_mask, -1)
                            else:
                                max_attention_positions = []
                                max_attention_val = []
                            # max_attention_positions = []
                        else:
                            (
                                max_attention_val,
                                max_attention_positions,
                            ) = temp_overall_attn_score.topk(top_k_mask, -1)
                            # only get non zero positions
                    attacked_attention_mask = new_attention_mask.clone()
                    if args.debug:
                        assert torch.all(
                            torch.isin(max_attention_positions, is_not_padding)
                        ), "max_attention_positions should be in is_not_padding"
                        assert torch.all(
                            attacked_attention_mask[max_attention_positions] == 1
                        )
                    total_attacked_attn_seq += len(
                        max_attention_positions.flatten()
                    )
                    # attacked_attention_mask = attacked_attention_mask.view_as(
                    #     attn_score_grad
                    # )  # [3,128,128]
                    attacked_attention_mask = attacked_attention_mask.view(
                        max_attention_positions.shape[0], -1
                    )
                    attacked_attention_mask[:, max_attention_positions] = 0
                    attacked_attention_mask = attacked_attention_mask.view_as(
                        attn_score_grad
                    )
                    # restore cls tokens attn
                    ref_new_attention_mask = ref_new_attention_mask.view_as(
                        attacked_attention_mask
                    )
                    attacked_attention_mask[ref_new_attention_mask == -1] = 1
                    if batch_i < 1 and args.debug:
                        logger.info(max_attention_positions[:10])
                        logger.info(
                            f"current masked count: {total_attacked_attn_seq}"
                        )
                        # logger.info(f"total masked count: {total_attn_seq}")
                        # logger.info(
                        #     f"masked percentage: {total_attacked_attn_seq/total_attn_seq}"
                        # )
                        logger.info(
                            f"masking {top_k_mask} out of {len(is_not_padding)}"
                        )
                        logger.info(f"Layer {layer_i} Head {h_i}")
                else:
                    ori_attention_mask[0][0][
                        : attention_mask.sum()
                    ] = 1  # restore cls token attn
                    attacked_attention_mask = ori_attention_mask

                ori_head_attn_mask.append(ori_attention_mask)
                head_attn_mask.append(attacked_attention_mask)
                # print(diff)
            # head attention mask for all heads in ith layer
            attacked_attention_mask = torch.stack(
                head_attn_mask, 1
            )  # [3,12,128,128]
            ori_attention_mask = torch.stack(
                ori_head_attn_mask, 1
            )  # [3,12,128,128]
            attention_mask = attacked_attention_mask
            attn_mask_by_layer_list.append(
                attacked_attention_mask
            )  # seq 1,12,128,128 mcq [16,12,4,12,128,128]
            ori_attn_mask_by_layer_list.append(ori_attention_mask)
        else:  # not attack layer, use original attention mask
            # new_attention_mask = torch.ones_like(
            #     attn_scores_layer_grad
            # )  # [3,12,128,128]
            # # attention_mask shape [1,3,128] -> [3,1,1,128]
            # is_not_padding = is_not_padding_mask.expand_as(
            #     attn_scores_layer_grad
            # )  # [3,12,128,128]
            # new_attention_mask = new_attention_mask * is_not_padding

            attn_mask_by_layer_list.append(new_original_attention_mask)
            ori_attn_mask_by_layer_list.append(new_original_attention_mask)
    # temp_attacked = torch.stack(attn_mask_by_layer_list).view(-1)
    # temp_ori = torch.stack(ori_attn_mask_by_layer_list).view(-1)
    # mask_metrics = attention_mask_metrics[args.chosen_metrics](
    #     temp_attacked,
    #     temp_ori,
    # )
    # mask_metrics = 0
    mask_metrics = total_attacked_attn_seq / block_count
    total_attn_seq = block_count * num_attention_seq
    masked_percentange = total_attacked_attn_seq / total_attn_seq
    # print(f"hamming {mask_metrics}")
    # print(f"masked unit {total_attacked_attn_seq}")
    # print(f"non_padding_seq {total_attn_seq}")
    # print(f"block_count {block_count}")
    total_non_pad_tokens = (
        num_attention_seq * to_mask_sa_matrix.view(-1).shape[0]
    )
    if batch_i < 1 and args.debug:
        logger.info(f"total_attn_seq {total_attn_seq}")
        logger.info(f"total_attacked_attn_seq {total_attacked_attn_seq}")
        logger.info(f"masked_percentage {masked_percentange}")
        logger.info(f"total_non_pad_tokens {total_non_pad_tokens}")
        logger.info(f"masked_metric {mask_metrics}")
    # mask_metrics1 = attention_mask_metrics["frobenius_norm"](
    #     torch.stack(attn_mask_by_layer_list),
    #     torch.stack(ori_attn_mask_by_layer_list),
    # )
    if batch_i < 10 and args.debug:
        count = 0
        try:
            bs = batch[0].shape[-2]
            seq_len = batch[0].shape[-1]
            # while count < 1:
            for i, ii in enumerate(to_mask_sa_matrix):
                for j, jj in enumerate(ii):
                    if ii == 1 and jj == 1:
                        source_sentence = tokenizer.convert_ids_to_tokens(
                            batch[0].flatten().tolist()
                        )  # [128]
                        # visualize_attention(
                        #     attn_scores[0][i][:, j, :, :],
                        #     bs=bs,
                        #     seq_len=seq_len,
                        #     source_sentence=source_sentence,
                        #     head=j,
                        #     layer=i,
                        #     wandb=wandb,
                        #     type="attn_weights",
                        # )
                        visualize_attention(
                            attn_mask_by_layer_list[i][:, j, :, :],
                            bs=bs,
                            seq_len=seq_len,
                            source_sentence=source_sentence,
                            head=j,
                            layer=i,
                            wandb=wandb,
                            type="masked",
                            args=args,
                        )
                        # visualize_attention(
                        #     attn_scores[1][i][:, j, :, :],
                        #     bs=bs,
                        #     seq_len=seq_len,
                        #     source_sentence=source_sentence,
                        #     head=j,
                        #     layer=i,
                        #     wandb=wandb,
                        #     type="attn_scores",
                        # )
                        if not args.debug:
                            count += 1
                            break
                    if not args.debug and count > 0:
                        break
                if not args.debug and count > 0:
                    break
        except Exception as e:
            logger.info(e)
            logger.info("visualize attention failed")
    return (
        [a.detach() for a in attn_mask_by_layer_list],
        (
            masked_percentange,
            total_attacked_attn_seq,
            total_attn_seq,
            total_non_pad_tokens,
            block_count,
        ),
        mask_metrics,
    )


def log_results(result_dict, table_name="Summary"):
    # Define the HTML table string
    html = "<table>\n"
    for key, value in result_dict.items():
        html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
    html += "</table>"

    wandb.log({table_name: wandb.Html(html)})


def get_confidence_score(
    args,
    victim,
    batch,
    by_layer: bool = True,
    device=torch.device("cpu"),
    target_layer: List[int] = None,
) -> Tuple[List, List]:
    """
    get the confidence score for each layer or head masked

    args:
        args: arguments
        victim: victim model
        batch: batch of data
        by_layer: if True, get the confidence score for each layer, else get the confidence score for each head
        target_layer: if has target_layer, use target_layer to choose head
    """
    layer_query = []
    head_mask = []

    if target_layer is None:
        for i in range(
            victim.config.num_hidden_layers
            if by_layer
            else victim.config.num_attention_heads
        ):
            temp_batch_attention_mask = batch[1].clone()  # [1,1,3,128]
            # make ith layer to all 0
            temp_batch_attention_mask = temp_batch_attention_mask.repeat(
                victim.config.num_hidden_layers, 1, 1
            )
            if by_layer:
                temp_batch_attention_mask[i] = temp_batch_attention_mask[i] * 0
            else:
                # (1.0 for keep, 0.0 for discard)
                head_mask_i = torch.tensor(
                    [int(a != i) for a in range(victim.config.num_attention_heads)],
                    device=device,
                )
                head_mask.append(head_mask_i)

            temp_batch = (
                batch[0],
                temp_batch_attention_mask,
                batch[2],
                batch[3],
            )
            layer_query.append(temp_batch)

        head_mask = torch.stack(head_mask) if head_mask else None
    else:
        for i in range(victim.config.num_attention_heads):
            temp_batch_attention_mask = batch[1].clone()  # [1,1,3,128]
            # make ith layer to all 0
            temp_batch_attention_mask = temp_batch_attention_mask.repeat(
                victim.config.num_hidden_layers, 1, 1
            )  # [12,1,3,128]
            head_mask_i = torch.tensor(
                [int(a != i) for a in range(victim.config.num_attention_heads)],
                device=device,
            )
            # else:
            #     head_mask_i = torch.ones(
            #         victim.config.num_attention_heads,
            #         device=device,
            #     )
            head_mask.append(head_mask_i)
            temp_batch = (
                batch[0],
                temp_batch_attention_mask,
                batch[2],
                batch[3],
            )
            layer_query.append(temp_batch)
        attacked_head_mask = torch.stack(head_mask) if head_mask else None
        n_batch, n_head = attacked_head_mask.shape
        healthy_head_mask = torch.ones(
            n_batch, victim.config.num_hidden_layers, n_head, device=device
        )
        healthy_head_mask[target_layer, :, :] = attacked_head_mask.type_as(
            healthy_head_mask
        )
        head_mask = healthy_head_mask

    # layer_query.append(temp_batch)
    # collate layer query
    input_ids = torch.stack([a[0] for a in layer_query]).squeeze()
    attention_mask = torch.stack([a[1] for a in layer_query])
    # if target_layer:
    #     attention_mask = attention_mask.repeat(
    #         1, 1, victim.config.num_attention_heads, 1
    #     )
    #     for j in range(victim.config.num_attention_heads):
    #         layer = attention_mask[j]
    #         for i in target_layer:
    #             layer[i][j] = layer[i][j] * 0
    #         attention_mask[j] = layer
    token_type_ids = torch.stack([a[2] for a in layer_query]).squeeze()
    label_ids = torch.stack([a[3] for a in layer_query]).squeeze()
    temp_batch = (input_ids, attention_mask, token_type_ids, label_ids)
    (
        original_pred,
        original_conf_score,
        ori_prediction,
        last_hidden_states,
        attn_scores,
    ) = query_model(
        model=victim,
        batch=temp_batch,
        head_mask=head_mask,
        no_grad=True,
        args=args,
        device=device,
    )

    return original_pred, original_conf_score


def get_filter_strategy(original_pred: List, original_conf_score: List) -> List[int]:
    """
    Returns the list of indices of the lowest score to the highest of false predictions.
    If all predictions are correct, returns the list of indices of lowest score to highest prediction.

    Args:
    original_pred: A list of predictions.
    original_conf_score: A list of confidence scores corresponding to the predictions.

    Returns:
    A list of indices sorted by the confidence score. If all predictions are correct, the indices are sorted
    in ascending order based on the confidence score.
    """
    indices = sorted(
        range(len(original_conf_score)), key=lambda k: original_conf_score[k]
    )
    if all(original_pred):
        return [(i, original_conf_score[i].item()) for i in indices]
    conf_tuple = [
        (i, original_conf_score[i].item()) for i in indices if not original_pred[i]
    ]
    conf_idx = [i for i, _ in conf_tuple]
    return conf_tuple + [
        (i, original_conf_score[i].item()) for i in indices if i not in conf_idx
    ]


def rank_layers_and_heads(
    args, victim, batch, is_random_mask: bool = False, device="cpu"
) -> Tuple[Tuple[int, float], Tuple[int, float]]:
    """
    algorithim to choosen head, layer and attention mask

    if args.attn_layer is None and args.attn_head is None:
        choosen layer and head are chosen based on the confidence score
    if one of the args.attn_layer or args.attn_head is not None:
        the other is chosen based on the confidence score

    """
    if is_random_mask:
        # random.seed(args.seed)
        chosen_layer = random.sample(
            range(victim.config.num_hidden_layers), args.layer_tuple_size[1]
        )
        # random.seed(args.seed+10)
        chosen_head = random.sample(
            range(victim.config.num_attention_heads), args.head_tuple_size[1]
        )
        chosen_layer = [(i, 1) for i in chosen_layer]
        chosen_head = [(i, 1) for i in chosen_head]
        return chosen_layer, chosen_head

    if args.attn_layer_mask is None:
        # Get the filter predictions and confidence scores for the victim model
        filter_layer_pred, filter_layer_conf_score = get_confidence_score(
            args, victim, batch, device=device
        )
        chosen_layer = get_filter_strategy(filter_layer_pred, filter_layer_conf_score)[
            : args.layer_tuple_size[1]
        ]
    else:
        chosen_layer = [
            (i, 1) for i, a in enumerate([*args.attn_layer_mask]) if a == "1"
        ]
    if args.attn_head_mask is None:
        # Get the filter predictions and confidence scores for the victim model
        filter_head_pred, filter_head_conf_score = get_confidence_score(
            args, victim, batch, by_layer=False, device=device
        )
        chosen_head = get_filter_strategy(filter_head_pred, filter_head_conf_score)[
            : args.head_tuple_size[1]
        ]
    else:
        chosen_head = [(i, 1) for i, a in enumerate([*args.attn_head_mask]) if a == "1"]
    return chosen_layer, chosen_head


def generate_and_save_attention_img(
    attn_score_hackattend,
    attn_score_normal,
    batch,
    tokenizer,
    is_grad=False,
    l_idx=-1,
    h_idx=-1,
):
    softmax_dim = 0
    masked = -1e9
    attn_score_hackattend = attn_score_hackattend.cpu().clone().detach()
    # .round(2)
    attn_score_normal = attn_score_normal.cpu().clone().detach()  # .round(2)
    # Tokens ['[CLS]'
    tokens = tokenizer.convert_ids_to_tokens(batch[0][0][1 : batch[1][0].sum() - 1])

    # Calculate the maximum value in the data
    # max_value = max(np.max(attn_score_hackattend), np.max(attn_score_normal))

    # Set the range for clipping outliers
    clip_min = None  # Minimum value to retain
    clip_max = None  # Maximum value to retain

    # Clip the values in the attention_scores arrays
    # attn_score_hackattend = np.clip(attn_score_hackattend, clip_min, clip_max)
    # attn_score_normal = np.clip(attn_score_normal, clip_min, clip_max)

    # Plotting attention_scores
    plt.figure(figsize=(12, 6))

    # Subplot for attention_scores
    plt.subplot(1, 2, 1)
    cmap = plt.cm.get_cmap("Blues")  # Define the colormap
    heatmap = attn_score_hackattend
    if not is_grad:
        heatmap[attn_score_hackattend == 0] = -1e9
        heatmap = heatmap.softmax(softmax_dim)
        attn_score_normal = attn_score_normal.softmax(softmax_dim)
    heatmap = np.array(heatmap)
    attn_score_normal = np.array(attn_score_normal)

    plt.imshow(
        heatmap, cmap=cmap, interpolation="nearest", vmin=clip_min, vmax=clip_max
    )

    # Add text annotations to each cell with contrasting color
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            value = heatmap[i, j]
            # Choose text color based on the background color
            text_color = "red" if value == 0 else "black" if value < 0.5 else "white"
            if value == 0:
                text = "[M]"
            elif value == 0:
                text = 0
            elif not is_grad and value < 0.001 and value > 0:
                text = "0.001"
            elif not is_grad and value < 0:
                text = "-0.001"
            else:
                text = f"{value:.2f}"

            plt.text(j, i, text, ha="center", va="center", color=text_color)

    # Set ticks and labels for the tokens
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(tokens)), tokens, fontsize=8)

    plt.title(f"HackAttend. Layer {l_idx}, Head {h_idx}")

    # Subplot for attention_score1
    plt.subplot(1, 2, 2)
    cmap = plt.cm.get_cmap("Blues")  # Define the colormap

    heatmap = attn_score_normal

    plt.imshow(
        heatmap, cmap=cmap, interpolation="nearest", vmin=clip_min, vmax=clip_max
    )

    # Add text annotations to each cell with contrasting color and difference indicator
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            value = attn_score_normal[i, j]
            # Choose text color based on the background color
            text_color = "red" if value == 0 else "black" if value < 0.5 else "white"

            if value == 0:
                text = 0
            elif not is_grad and value < 0.001 and value > 0:
                text = "0.001"
            elif not is_grad and value < 0:
                text = "-0.001"
            else:
                text = f"{value:.2f}"

            plt.text(j, i, text, ha="center", va="center", color=text_color)

        # Set ticks and labels for the tokens
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(tokens)), tokens, fontsize=8)

    plt.title(f"Original. Layer {l_idx}, Head {h_idx}")

    # Display the plots
    plt.tight_layout()
    plt.savefig("case_study1.png")
    wandb.log({"attention": plt})
    plt.savefig("case_study1.pdf")
    wandb.save("case_study1.pdf")
    plt.close("all")
    # plt.show()
