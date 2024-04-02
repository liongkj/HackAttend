import itertools
import random
from typing import List, Tuple
import wandb
import torch
import utils
import pandas as pd
from utils import (
    generate_atten_mask,
    generate_pertubed_attention_mask,
    get_confidence_score,
    log_results,
    query_model,
    get_masked_percentage,
)


def generate_attack_combinations(
    args,
    chosen_layer: Tuple[int, float],
    chosen_head: Tuple[int, float],
    strategy="agg",
    random_combo=False,
) -> List[Tuple[int, int]]:
    layer_sizes = range(args.layer_tuple_size[0], args.layer_tuple_size[1] + 1)
    head_sizes = range(args.head_tuple_size[0], args.head_tuple_size[1] + 1)
    min_head_size, max_head_size = args.head_tuple_size[0], args.head_tuple_size[1]
    min_layer_size, max_layer_size = args.layer_tuple_size[0], args.layer_tuple_size[1]
    c = int((max_head_size - min_head_size) / 2) or 1
    d = int((max_layer_size - min_layer_size) / 2) or 1
    combinations = []
    if args.attn_layer_mask is None and args.attn_head_mask is None:
        if strategy == "naive":
            # takes very long to generate combinations when tuple size difference is large
            layers_choices = [a[0] for a in chosen_layer]
            heads_choices = [a[0] for a in chosen_head]
            for i, j in itertools.product(head_sizes, layer_sizes):
                for layer_comb in itertools.combinations(layers_choices, j):
                    for head_comb in itertools.combinations(heads_choices, i):
                        combo = (
                            tuple(sorted(set(layer_comb))),
                            tuple(sorted(set(head_comb))),
                        )
                        if combo not in combinations:
                            combinations.append(combo)
        elif strategy == "agg":
            combo_dict = {}

            def score_combination(
                combination, layer_scores: List[Tuple], head_scores, penalty=1.5
            ):
                """
                Calculates the score of a given combination of layers and heads based on the provided layer scores and head scores, with penalties for repetition and length.

                Args:
                    combination: A tuple containing two lists of integers representing the
                        selected layers and heads.
                    layer_scores: A list of tuples, where each tuple contains an integer
                        representing a layer and a float representing its score.
                    head_scores: A list of tuples, where each tuple contains an integer
                        representing a head and a float representing its score.
                    penalty: A float representing the penalty for repetition and length.
                        Default is 1.5.

                Returns:
                    A tuple containing the score of the combination, a tuple of the selected
                    layers, and a tuple of the selected heads.
                """
                layers, heads = combination
                res_lay = tuple(a[0] for a in layers)
                res_he = tuple(a[0] for a in heads)
                n_layers = len(layers)
                n_heads = len(heads)
                chosen_layer = [a[0] for a in layers]
                layer_scores_dict = {a[0]: a[1] for a in layer_scores}
                head_scores_dict = {a[0]: a[1] for a in head_scores}
                layer_score = sum(layer_scores_dict[a] for a in chosen_layer) / n_layers
                chosen_head = [a[0] for a in heads]
                head_score = sum(head_scores_dict[a] for a in chosen_head) / n_heads

                # score = (layer_score + layer_penalty + head_score + head_penalty) / 2
                score = (layer_score + head_score) / 2
                repeat_penalty = (
                    score * penalty * combo_dict.get((n_layers, n_heads), 0)
                )

                head_penalty = head_score * abs(n_heads - c) / (c)
                layer_penalty = head_score * abs(n_layers - d) / d
                length_penalty = (head_penalty + layer_penalty) / 2

                combo_dict[(n_layers, n_heads)] = (
                    combo_dict.get((n_layers, n_heads), 0) + 1
                )
                score = score + repeat_penalty + length_penalty
                return score, res_lay, res_he

            # layer_size = range(6,12) = head_sizes
            for layer_size, head_size in itertools.product(layer_sizes, head_sizes):
                layer_combinations = sorted(
                    itertools.combinations(chosen_layer, layer_size),
                    key=lambda x: x[0][1],  # sort by layer conf score
                    reverse=False,
                )[: args.max_combinations]
                head_combinations = sorted(
                    itertools.combinations(chosen_head, head_size),
                    key=lambda x: x[0][1],
                    reverse=False,
                )[: args.max_combinations]
                for layer_comb in layer_combinations:
                    for head_comb in head_combinations:
                        combo = tuple(sorted({l for l in layer_comb})), tuple(
                            sorted({h for h in head_comb})
                        )
                        if combo not in combinations:
                            combinations.append(combo)
            scores = []
            chosen_layer = sorted(chosen_layer, key=lambda x: x[0], reverse=False)
            chosen_head = sorted(chosen_head, key=lambda x: x[0], reverse=False)
            for combo in combinations:
                score, res_lay, res_he = score_combination(
                    combo, chosen_layer, chosen_head
                )
                scores.append(((res_lay, res_he), score))
            sor = sorted(scores, key=lambda x: x[1], reverse=False)
            # get lowest 20% of scores
            count_n = max(int(len(sor) * 0.01), 50)
            combinations = [s[0] for s in sor[:count_n]]

    elif args.attn_layer_mask is None or args.attn_head_mask is None:
        if args.attn_layer_mask is None:
            for i in layer_sizes:
                combinations.extend(
                    (tuple(layer_comb), tuple(chosen_head))
                    for layer_comb in itertools.combinations(chosen_layer, i)
                )
        else:
            for i in head_sizes:
                combinations.extend(
                    (tuple(chosen_layer), tuple(head_comb))
                    for head_comb in itertools.combinations(chosen_head, i)
                )
    else:
        combinations = [(tuple(chosen_layer), tuple(chosen_head))]
    # combinations = combinations[: args.num_tries] if args.num_tries else combinations
    # if top_percentage_in_layer = 0.01, then 0.0001, 0.0051, 0.0101
    masked_percentage = get_masked_percentage(args)
    if random_combo:
        num_tries = min(args.num_tries, len(combinations))
        combinations = (
            random.sample(list(combinations), num_tries)
            if args.num_tries
            else combinations
        )
        # split num_tries to length of masked percentage
        num_tries = max(num_tries // len(masked_percentage), num_tries)
        combinations = [
            (combo, masked_percentage[i])
            for i in range(len(masked_percentage))
            for combo in random.sample(combinations, num_tries)
        ]
    else:
        # split num_tries to length of masked percentage
        num_tries = args.num_tries // len(masked_percentage)
        combinations = [
            (combo, masked_percentage[i])
            for i in range(len(masked_percentage))
            for combo in combinations[: num_tries * len(masked_percentage)]
        ]
    return combinations


import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def score_attack(
    args,
    victim,
    success_attack,
    failed_attack,
    total_num_query,
    mask_percentage_list,
    evaluate_score_list,
    attack_layer_dict,
    attack_head_dict,
    attack_idx_list,
    is_random_mask,
    batch_i,
    batch,
    original_attn_scores,
    chosen_layer,
    chosen_head,
    device=None,
    tokenizer=None,
):
    """
    score based attack strategy, returns combinations of layers and heads to be masked
    """
    strat = args.attack_strat.split("_")[0]
    is_random = args.attack_strat.split("_")[1] == "random"
    combinations = generate_attack_combinations(
        args, chosen_layer, chosen_head, strategy=strat, random_combo=is_random
    )
    # log combinations when the number of combinations if dataloader loop less than 2
    if batch_i < 1:
        # log combinations, line break for each combinations
        logger.info("combinations")
        for i in combinations:
            logger.info(i)

        logger.info("\n")
        comb_str = ""
        for i in combinations:
            comb_str += str(i) + "\n"

        sample_dict = {
            "sample": batch_i,
            "combinations": comb_str,
        }
        log_results(sample_dict, "sample_combination")

    attack_succeed = False
    for (l_i, h_i), mask_per in combinations:
        args.top_percentage_in_layer = mask_per
        atten_layer_mask_list = generate_atten_mask(
            l_i, victim.config.num_hidden_layers
        )
        atten_head_mask_list = generate_atten_mask(
            h_i, victim.config.num_attention_heads
        )
        (
            attn_mask_by_layer_list,
            mask_percentage,
            score,
        ) = generate_pertubed_attention_mask(
            original_attn_scores,
            atten_layer_mask_list,
            atten_head_mask_list,
            random_units_mask=is_random_mask,
            batch=batch,
            batch_i=batch_i,
            tokenizer=tokenizer,
            args=args,
            device=device,
        )

        # Run the attack and check for success
        total_num_query[batch_i] += 1
        asa_attack_is_fail, ____, ____, _, __ = query_model(
            model=victim,
            batch=batch,
            attn_mask_by_layer_list=attn_mask_by_layer_list,
            args=args,
            device=device,
        )
        if not asa_attack_is_fail:
            # The attack succeeded, break out of the loop
            success_attack += 1
            attack_succeed = True
            attack_idx_list.append(batch_i)
            mask_percentage_list.append(mask_percentage)
            evaluate_score_list.append(score)
            for chosen_l in l_i:
                attack_layer_dict[chosen_l] = attack_layer_dict.get(chosen_l, 0) + 1
            for chosen_h in h_i:
                attack_head_dict[chosen_h] = attack_head_dict.get(chosen_h, 0) + 1
            return True

    if not attack_succeed:
        # logger.warning("FAILED")
        attack_layer_dict[-1] = attack_layer_dict.get(-1, 0) + 1
        attack_head_dict[-1] = attack_head_dict.get(-1, 0) + 1
        failed_attack += 1
        return False
    
def greedy_attack(
    chosen_layer: List[Tuple[int, float]],
    args,
    victim,
    batch,
    device,
    batch_i: int,
    original_attn_scores,
    tokenizer,
    sa_matrix_pd,
    success_attack=None,
    total_num_query=None,
    attack_idx_list=None,
    evaluate_score_list=None,
    mask_percentage_list=None,
    attack_layer_dict=None,
    attack_head_dict=None,
    failed_attack=None,
    random_units_mask=False,
    is_random_mask=False,
) -> bool | List[torch.Tensor]:  # sourcery skip: assign-if-exp, reintroduce-else
    attack_succeed = False
    masked_percentage = get_masked_percentage(args)
    to_mask_sa_matrix = torch.zeros(victim.config.num_hidden_layers,victim.config.num_attention_heads)
    
    pd_headers = ['sa_{}'.format(i) for i in range(victim.config.num_hidden_layers*victim.config.num_attention_heads)]
    if is_random_mask: #randomly select layers
        chosen_layer = random.sample(
            range(victim.config.num_hidden_layers), args.layer_tuple_size[1]
        )
        chosen_layer = [(a,1) for a in chosen_layer]
        # random.seed(args.seed+10)
    # for top_i_layer in range(1, args.layer_tuple_size[1] + 1):
    for layer_i, target_layers in enumerate(tuple([a[0] for a in chosen_layer[:args.layer_tuple_size[1]]])):
        # layer selection
        # target_layers = tuple([a[0] for a in chosen_layer][:top_i_layer])
        _, conf_score = get_confidence_score(
            args,
            victim=victim,
            batch=batch,
            device=device,
            by_layer=False,
            target_layer=[target_layers],
        )
        # get index of sorted confidence score
        if is_random_mask:
            sorted_head = random.sample(
                range(victim.config.num_attention_heads), args.head_tuple_size[1]
            )
            sorted_head = [(a,1) for a in sorted_head]
        else:
            sorted_head = sorted(list(enumerate(conf_score)), key=lambda x: x[1])
        # translate layer and head to index based for example layer 2 head 2 should be 39th SA matrix
        for mask_percent in masked_percentage:
            # for top_i_head in range(1, args.head_tuple_size[1] + 1):
            for head_i, target_head_list in enumerate(tuple([a[0] for a in sorted_head][:args.head_tuple_size[1]])):
                # head selection
                # target_head_list = tuple([a[0] for a in sorted_head][:top_i_head])
                # sa_index = (target_layers * victim.config.num_attention_heads) + target_head_list
                to_mask_sa_matrix[target_layers][target_head_list] = 1
                # combinations = (target_layers, target_head_list, mask_percent)
                # l_i, h_i, mask_per = combinations
                # args.top_percentage_in_layer = mask_per
                # atten_layer_mask_list = generate_atten_mask(
                #     l_i, victim.config.num_hidden_layers
                # )
                # atten_head_mask_list = generate_atten_mask(
                #     h_i, victim.config.num_attention_heads
                # )
                # generate the M'
                (
                    attn_mask_by_layer_list,
                    mask_percentage,
                    score,
                ) = generate_pertubed_attention_mask(
                    original_attn_scores,
                    to_mask_sa_matrix,
                    random_units_mask=random_units_mask,
                    batch=batch,
                    batch_i=batch_i,
                    tokenizer=tokenizer,
                    args=args,
                    device=device,
                )
                # if total_num_query:
                total_num_query[batch_i] += 1
                asa_attack_is_fail, ____, ____, _, attn_score = query_model(
                    model=victim,
                    batch=batch,
                    attn_mask_by_layer_list=attn_mask_by_layer_list,
                    args=args,
                    device=device,
                    # no_grad for gradient
                )
                # visual = False
                # if visual and not asa_attack_is_fail:
                visual = False
                if not asa_attack_is_fail:
                    if visual and batch[1].sum() < 15:
                        (
                            asa_attack_is_fail,
                            ____,
                            ____,
                            _,
                            attn_score_hackattend,
                        ) = query_model(
                            model=victim,
                            batch=batch,
                            attn_mask_by_layer_list=attn_mask_by_layer_list,
                            args=args,
                            device=device,
                            visualize=visual
                            # no_grad for gradient
                        )
                        (
                            asa_attack_is_fail,
                            ____,
                            ____,
                            _,
                            attn_score_normal,
                        ) = query_model(
                            model=victim,
                            batch=batch,
                            attn_mask_by_layer_list=batch[1].cuda(),
                            args=args,
                            device=device,
                            visualize=visual
                            # no_grad for gradient
                        )
                        if visual:
                            # combination of each layer and head
                            for l_idx, l in enumerate(to_mask_sa_matrix):
                                if not torch.allclose(l,torch.tensor(1.0)):
                                    for h_idx, h in enumerate(l):
                                        if h == 1:
                                            # attn score
                                            seq_len = batch[1][0].sum() - 1
                                            attn_score_hackattend_score = (
                                                attn_score_hackattend[1][l_idx][0][
                                                    h_idx
                                                ][
                                                    1:seq_len,
                                                    1:seq_len,
                                                ]
                                            )
                                            attn_score_hackattend_score = (
                                                attn_score_hackattend_score.cpu()
                                                .clone()
                                                .detach()
                                            )
                                            attn_score_normal_score = attn_score_normal[
                                                1
                                            ][l_idx][0][h_idx][
                                                1:seq_len,
                                                1:seq_len,
                                            ]
                                            utils.generate_and_save_attention_img(
                                                attn_score_hackattend_score,
                                                attn_score_normal_score,
                                                batch=batch,
                                                tokenizer=tokenizer,
                                                l_idx=l_idx,
                                                h_idx=h_idx,
                                            )
                                            # gradient
                                            attn_score_hackattend_grad = (
                                                attn_score_hackattend[0][l_idx][0][
                                                    h_idx
                                                ][
                                                    1:seq_len,
                                                    1:seq_len,
                                                ]
                                            )
                                            attn_score_hackattend_grad = (
                                                attn_score_hackattend_grad.cpu()
                                                .clone()
                                                .detach()
                                            )
                                            attn_score_normal_grad = attn_score_normal[
                                                0
                                            ][l_idx][0][h_idx][
                                                1:seq_len,
                                                1:seq_len,
                                            ]
                                            utils.generate_and_save_attention_img(
                                                attn_score_hackattend_grad,
                                                attn_score_normal_grad,
                                                batch=batch,
                                                tokenizer=tokenizer,
                                                is_grad=True,
                                                l_idx=l_idx,
                                                h_idx=h_idx,
                                            )
                    # The attack succeeded, break out of the loop
                    if success_attack is None:  # is adversarial preprocessing
                        return [a.detach().cpu() for a in attn_mask_by_layer_list]
                    success_attack += 1
                    attack_succeed = True
                    wandb.log({"hamming":score})
                    attack_idx_list.append(batch_i)
                    mask_percentage_list.append(mask_percentage)
                    evaluate_score_list.append(score)
                    # for chosen_l in l_i:
                    for attacked_layers in [a[0] for a in chosen_layer[:args.layer_tuple_size[1]]][:layer_i+1]:
                        attack_layer_dict[attacked_layers] = (
                            attack_layer_dict.get(attacked_layers, 0) + 1
                        )
                    # wandb.log({"attacked_sa":wandb.Image(to_mask_sa_matrix)})
                    # flatten to log in pandas
                    sa_matrix_pd = pd.concat([sa_matrix_pd, pd.DataFrame(to_mask_sa_matrix.view(1,-1).numpy(),columns = pd_headers)], ignore_index=True)
                    attack_head_dict[target_head_list] = (
                        attack_head_dict.get(target_head_list, 0) + 1
                    )

                    return True

                if attack_succeed:
                    break
            if attack_succeed:
                break
        if attack_succeed:
            break
    if not attack_succeed:
        if success_attack is None:
            return None
        attack_layer_dict[-1] = attack_layer_dict.get(-1, 0) + 1
        attack_head_dict[-1] = attack_head_dict.get(-1, 0) + 1
        failed_attack += 1
        wandb.log({"failed_attack":failed_attack})
        print(f"failed {total_num_query[batch_i]} with tries")
        return False


def greedy_attack_old(
    chosen_layer: List[Tuple[int, float]],
    args,
    victim,
    batch,
    device,
    batch_i: int,
    original_attn_scores,
    tokenizer,
    success_attack=None,
    total_num_query=None,
    attack_idx_list=None,
    evaluate_score_list=None,
    mask_percentage_list=None,
    attack_layer_dict=None,
    attack_head_dict=None,
    failed_attack=None,
    random_units_mask=False,
    is_random_mask=False
) -> bool | List[torch.Tensor]:  # sourcery skip: assign-if-exp, reintroduce-else
    attack_succeed = False
    masked_percentage = get_masked_percentage(args)
    for top_i_layer in range(1, args.layer_tuple_size[1] + 1):
        # layer selection
        target_layers = tuple([a[0] for a in chosen_layer][:top_i_layer])
        _, conf_score = get_confidence_score(
            args,
            victim=victim,
            batch=batch,
            device=device,
            by_layer=False,
            target_layer=target_layers,
        )
        # get index of sorted confidence score
        sorted_head = sorted(list(enumerate(conf_score)), key=lambda x: x[1])
        for mask_percent in masked_percentage:
            for top_i_head in range(1, args.head_tuple_size[1] + 1):
                # head selection
                target_head_list = tuple([a[0] for a in sorted_head][:top_i_head])
                combinations = (target_layers, target_head_list, mask_percent)
                l_i, h_i, mask_per = combinations
                args.top_percentage_in_layer = mask_per
                atten_layer_mask_list = generate_atten_mask(
                    l_i, victim.config.num_hidden_layers
                )
                atten_head_mask_list = generate_atten_mask(
                    h_i, victim.config.num_attention_heads
                )
                (
                    attn_mask_by_layer_list,
                    mask_percentage,
                    score,
                ) = generate_pertubed_attention_mask(
                    original_attn_scores,
                    atten_layer_mask_list,
                    atten_head_mask_list,
                    random_units_mask=random_units_mask,
                    batch=batch,
                    batch_i=batch_i,
                    tokenizer=tokenizer,
                    args=args,
                    device=device,
                )
                # if total_num_query:
                total_num_query[batch_i] += 1
                asa_attack_is_fail, ____, ____, _, attn_score = query_model(
                    model=victim,
                    batch=batch,
                    attn_mask_by_layer_list=attn_mask_by_layer_list,
                    args=args,
                    device=device,
                    # no_grad for gradient
                )
                visual = True
                if not asa_attack_is_fail:
                    if batch[1].sum() < 15:
                        (
                            asa_attack_is_fail,
                            ____,
                            ____,
                            _,
                            attn_score_hackattend,
                        ) = query_model(
                            model=victim,
                            batch=batch,
                            attn_mask_by_layer_list=attn_mask_by_layer_list,
                            args=args,
                            device=device,
                            visualize=visual
                            # no_grad for gradient
                        )
                        (
                            asa_attack_is_fail,
                            ____,
                            ____,
                            _,
                            attn_score_normal,
                        ) = query_model(
                            model=victim,
                            batch=batch,
                            attn_mask_by_layer_list=batch[1].cuda(),
                            args=args,
                            device=device,
                            visualize=visual
                            # no_grad for gradient
                        )
                        if visual:
                            # combination of each layer and head
                            for l_idx, l in enumerate(atten_layer_mask_list):
                                if l == 1:
                                    for h_idx, h in enumerate(atten_head_mask_list):
                                        if h == 1:
                                            # attn score
                                            seq_len = batch[1][0].sum() - 1
                                            attn_score_hackattend_score = (
                                                attn_score_hackattend[1][l_idx][0][
                                                    h_idx
                                                ][
                                                    1:seq_len,
                                                    1:seq_len,
                                                ]
                                            )
                                            attn_score_hackattend_score = (
                                                attn_score_hackattend_score.cpu()
                                                .clone()
                                                .detach()
                                            )
                                            attn_score_normal_score = attn_score_normal[
                                                1
                                            ][l_idx][0][h_idx][
                                                1:seq_len,
                                                1:seq_len,
                                            ]
                                            utils.generate_and_save_attention_img(
                                                attn_score_hackattend_score,
                                                attn_score_normal_score,
                                                batch=batch,
                                                tokenizer=tokenizer,
                                                l_idx=l_idx,
                                                h_idx=h_idx,
                                            )
                                            # gradient
                                            attn_score_hackattend_grad = (
                                                attn_score_hackattend[0][l_idx][0][
                                                    h_idx
                                                ][
                                                    1:seq_len,
                                                    1:seq_len,
                                                ]
                                            )
                                            attn_score_hackattend_grad = (
                                                attn_score_hackattend_grad.cpu()
                                                .clone()
                                                .detach()
                                            )
                                            attn_score_normal_grad = attn_score_normal[
                                                0
                                            ][l_idx][0][h_idx][
                                                1:seq_len,
                                                1:seq_len,
                                            ]
                                            utils.generate_and_save_attention_img(
                                                attn_score_hackattend_grad,
                                                attn_score_normal_grad,
                                                batch=batch,
                                                tokenizer=tokenizer,
                                                is_grad=True,
                                                l_idx=l_idx,
                                                h_idx=h_idx,
                                            )
                    # The attack succeeded, break out of the loop
                    if success_attack is None:  # is adversarial preprocessing
                        return [a.detach().cpu() for a in attn_mask_by_layer_list]
                    success_attack += 1
                    attack_succeed = True
                    wandb.log({"success_attack":success_attack})
                    attack_idx_list.append(batch_i)
                    mask_percentage_list.append(mask_percentage)
                    evaluate_score_list.append(score)
                    for chosen_l in l_i:
                        attack_layer_dict[chosen_l] = (
                            attack_layer_dict.get(chosen_l, 0) + 1
                        )
                    for chosen_h in h_i:
                        attack_head_dict[chosen_h] = (
                            attack_head_dict.get(chosen_h, 0) + 1
                        )

                    return True

                if attack_succeed:
                    break
            if attack_succeed:
                break
        if attack_succeed:
            break
    if not attack_succeed:
        if success_attack is None:
            return None
        attack_layer_dict[-1] = attack_layer_dict.get(-1, 0) + 1
        attack_head_dict[-1] = attack_head_dict.get(-1, 0) + 1
        # failed_attack += 1
        print(f"failed {total_num_query[batch_i]} with tries")
        return False
