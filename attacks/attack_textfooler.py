import copy
import os
from argparse import ArgumentParser
from types import SimpleNamespace

import textattack
import torch
import transformers
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoConfig

import wandb
from run_hackattend import (
    convert_examples_to_features_for_multiple_choice,
    convert_examples_to_features_for_sequence_classification,
)
from modelling import bert
from victim.HackAttend.run_multi_cho import DreamProcessor  # InputFeatures,
from victim.HackAttend.run_multi_cho import (
    AlphaNliProcessor,
    HellaswagProcessor,
    Metrics,
    ReclorProcessor,
    SwagProcessor,
)
from victim.HackAttend.run_sent_clas import (
    MnliProcessor,
    PawsqqpProcessor,
    QqpProcessor,
    SstProcessor,
)
from utils import mcq

# from ..TextAttack import textattack


def main(args):
    task_name_model = {"sst-2": "sst2", "reclor": "voidful/ReClor", "dream": "dream"}
    # map to huggingface dataset
    args.load_model_path = f"model/{args.task_name}_{args.model_type}"
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #     # "textattack/bert-base-uncased-SST-2"
    #     model_path
    # )

    recipe_dict = {
        "bertattack": textattack.attack_recipes.BERTAttackLi2020,
        "textfooler": textattack.attack_recipes.TextFoolerJin2019,
        "clare": textattack.attack_recipes.CLARE2020,
    }
    recipe_strat = recipe_dict.get(args.recipe)
    assert recipe_strat, "Please choose a attack recipe"
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
    }
    if task_name not in processors:
        raise ValueError(f"Task not found: {task_name}")
    processor = processors[task_name]()
    # if args.task_name in ["sst-2","reclor"]:
    if args.generate_adv_samples:
        dataset = processor.get_train_examples(os.path.join(args.data_dir, task_name))
    else:
        dataset = processor.get_test_examples(os.path.join(args.data_dir, task_name))
    if args.start != None and args.end != None:
        dataset = dataset[args.start : args.end]
    label_list = processor.get_labels()
    # dataset = processor.get_test_examples(os.path.join(args.data_dir, task_name))
    config = AutoConfig.from_pretrained(args.model_and_path)
    if args.task_name in mcq:
        model = bert.BertForMultipleChoice(config)
        converter = convert_examples_to_features_for_multiple_choice
    elif args.task_name in ["sst-2", "mnli", "paws-qqp"]:
        model = bert.BertForSequenceClassification(config)
        converter = convert_examples_to_features_for_sequence_classification
    else:
        model = None
        converter = None
    assert model is not None and converter is not None
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_and_path)
    eval_features = converter(dataset, label_list, args.max_seq_length)
    data = []
    if args.task_name in mcq:
        for ev in eval_features:
            list_of_lists = [[*a][1] for a in ev.choices_features]
            flat_list = (
                tuple([ev.choices_features[0][0]] + list_of_lists),
                ev.label_id,
            )
            data.append(flat_list)
        column_name = tuple(["ctx"] + [f"choice{i}" for i in range(len(label_list))])
    elif args.task_name in ["paws-qqp"]:
        for ev in eval_features:
            list_of_lists = list(ev.features)
            flat_list = (
                tuple(list_of_lists),
                ev.label_id,
            )
            data.append(flat_list)
        column_name = tuple(["ctx", "choice0"])
    else:
        data.extend((ev.features, ev.label_id) for ev in eval_features)
        column_name = tuple(["text"])
    print(f"column_name: {column_name}")
    dataset = textattack.datasets.Dataset(data, input_columns=column_name)
    # dataset = textattack.datasets.HuggingFaceDataset(
    #     task_name_model[args.task_name], split="validation" if args.task_name in ["sst-2","reclor"] else "test"
    # )
    model.load_state_dict(
        torch.load(
            os.path.join(args.load_model_path, f"{args.best_epoch}_pytorch_model.bin")
        )
    )
    # model2 = copy.deepcopy(model)
    model.eval()
    print(f"model: {os.path.join(args.load_model_path)}")
    # import .textattack
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
    # defend
    # data_temp = list(dataset[1][0].values())
    # text_input_list = []
    # text_input_list_temp = []
    # a = []
    # for x,y in zip([data_temp[0]] *( len(data_temp) -1) ,data_temp[1:]):
    #     a.append(x)
    #     a.append(y)
    # max_length = 128
    # text_input_list = [a]
    # dataset[1]
    # for input_list in text_input_list:
    #     result = [input_list[i:i+2] for i in range(0, len(input_list), 2)]
    #     for context, ending in result:
    #         text_a = context
    #         if ending.find("_") != -1:
    #             text_b = ending.replace("_", ending)
    #         else:
    #             text_b = f"{ending}"
    #         text_input_list_temp.append((text_a,text_b))
    # inputs_dict = tokenizer(
    #     text_input_list_temp,
    #     max_length=max_length,
    #     padding="max_length",
    #     truncation=True,

    #     add_special_tokens=True,
    #     return_tensors="pt",
    # )
    # inputs_dict["input_ids"] = inputs_dict["input_ids"].reshape(len(text_input_list),-1,max_length)
    # inputs_dict["attention_mask"] = inputs_dict["attention_mask"].reshape(len(text_input_list),-1,max_length)
    # inputs_dict["token_type_ids"] = inputs_dict["token_type_ids"].reshape(len(text_input_list),-1,max_length)
    # model.eval()
    # model2.eval()
    # with torch.no_grad():
    #     outputs = model(
    #         **inputs_dict,
    #         args=attack_args,
    #         # output_hidden_states=False,
    #         output_attentions=True,
    #         # # head_mask=head_mask,
    #         # return_dict=False,
    #     )
    #     outputs1 = model2(
    #         **inputs_dict,
    #         args=attack_args,
    #         # output_hidden_states=False,
    #         output_attentions=True,
    #         # # head_mask=head_mask,
    #         # return_dict=False,
    #     )

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
        model, tokenizer, attack_args
    )

    attack = recipe_strat.build(model_wrapper)
    filename = (
        f"train_{args.recipe}_{args.task_name}_log.csv"
        if args.generate_adv_samples
        else f"{args.recipe}_{args.task_name}_log.csv"
    )
    if args.start != None:
        filename = f"{args.start}_filename"
    attack_args = textattack.AttackArgs(
        num_examples=-1,
        log_to_csv=os.path.join("data", "asa", filename),
        checkpoint_interval=1000 if args.generate_adv_samples else 1000,
        checkpoint_dir="checkpoints",
        # parallel=True,
        # log_to_wandb={},
        # num_examples_offset=24711 if args.recipe == "textfooler" else 0,
        disable_stdout=True,
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/glue/",
        help="Directory to contain the input data for all tasks.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="bertattack",
        choices=["textfooler", "bertattack", "clare"],
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum total input sequence length after word-piece tokenization.",
    )
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--model_and_path", default="bert-base-uncased")
    parser.add_argument("--model_type", default="bert")
    parser.add_argument("--best_epoch", default="best")
    parser.add_argument("--generate_adv_samples", default=False, action="store_true")
    parser.add_argument("--start", default=None, type=int)
    parser.add_argument("--end", default=-1, type=int)
    args = parser.parse_args()
    wandb.init(project="asa", config=args)
    main(args)
