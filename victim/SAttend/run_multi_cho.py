from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import math
import os
import random

import numpy as np
import sklearn.metrics as mtc
import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              RandomSampler, SequentialSampler, TensorDataset)
from tqdm.auto import tqdm, trange
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          SchedulerType, get_scheduler)

import wandb

from .creat import CreATTrainer, Trainer
from .trainer.freelb import FreeLBTrainer
from .trainer.r3f import R3FTrainer
from .trainer.smart import SMARTTrainer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, question, contexts, endings, label=None):
        self.guid = guid
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, choices_features, label_id):
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label_id = label_id

class ReclorProcessor:
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test"
        )

    def get_labels(self):
        return ["0", "1", "2", "3"]

    @staticmethod
    def _read_json(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(json.load(f))

    @staticmethod
    def _create_examples(lines, set_type):
        examples = [
            InputExample(
                guid="%s-%s" % (set_type, line["id_string"]),
                question=line["question"],
                contexts=[line["context"]] * 4,
                endings=[
                    line["answers"][0],
                    line["answers"][1],
                    line["answers"][2],
                    line["answers"][3],
                ],
                label=str(line["label"]) if set_type != "dev" else "0",
            )
            for line in lines
        ]

        return examples
    
class DreamProcessor:
    """Processor for the DREAM data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        return ['0', '1', '2']

    @staticmethod
    def _read_json(input_file):
        with open(input_file, 'r', encoding="utf-8") as f:
            return list(json.load(f))

    @staticmethod
    def _create_examples(lines, set_type):
        examples = [
            InputExample(
                guid="%s-%s" % (set_type, line[-1]),
                question=line[1][0]["question"],
                contexts=[' '.join(line[0]), ' '.join(line[0]), ' '.join(line[0])],
                endings=line[1][0]["choice"],
                label=str(line[1][0]["choice"].index(line[1][0]["answer"])),
            )
            for line in lines
        ]

        return examples


class HellaswagProcessor:
    """Processor for the HELLASWAG data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "hellaswag_train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "hellaswag_val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "hellaswag_test.jsonl")), "test")

    def get_labels(self):
        return ['0', '1', '2', '3']

    @staticmethod
    def _read_json(input_file):
        with open(input_file, 'r', encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = [
            InputExample(
                guid="%s-%s" % (set_type, line["ind"]),
                question="_",
                contexts=[line["ctx"], line["ctx"], line["ctx"], line["ctx"]],
                endings=line["endings"],
                label=str(line["label"]) if set_type != "test" else '0',
            )
            for line in lines
        ]

        return examples


class AlphanliProcessor:
    """Processor for the Alpha NLI data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "alphanli_train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "alphanli_dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "alphanli_test.jsonl")), "test")

    def get_labels(self):
        return ['1', '2']

    @staticmethod
    def _read_json(input_file):
        with open(input_file, 'r', encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = [
            InputExample(
                guid="%s-%s" % (set_type, i),
                question="_",
                contexts=[' '.join([line["obs1"], line["obs2"]]), ' '.join([line["obs1"], line["obs2"]])],
                endings=[line["hyp1"], line["hyp2"]],
                label=str(line["label"]) if set_type != "test" else '1',
            )
            for i, line in enumerate(lines)
        ]

        return examples


class RaceProcessor:
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        return ["A", "B", "C", "D"]

    @staticmethod
    def _read(input_file):
        with open(input_file, 'r', encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = [
                InputExample(
                    guid="%s-%s-%s" % (set_type, line["id"][:-4], i),
                    question=line["questions"][i],
                    contexts=[line["article"], line["article"], line["article"], line["article"]],
                    endings=line["options"][i],
                    label=line["answers"][i],
                )
                for line in lines
            for i in range(len(line["answers"]))
        ]

        return examples

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
    ):
        # self.data = data
        self.args = args
        # self.attack_args = attack_args
        self.tokenizer = tokenizer
        self.is_clean = is_clean
        if not asa_data_dir:
            asa_data_dir = f"data/asa/{recipe}_{task}_log.csv"
        self.normal_data = normal_data
        # with open(normal_data,"r") as file:
        #     self.normal_data = list()
            
        with open(asa_data_dir, "r") as file:
            self.data = list(
                csv.reader(
                    file,
                )
            )[1:]

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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (i, example) in tqdm(enumerate(examples), desc="Converting examples to features"):
        choices_features = []
        for _, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # This is for cloze questions.
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            encoded_inputs = tokenizer(
                text_a,
                text_b,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True
            )

            choices_features.append(
                (encoded_inputs["input_ids"], encoded_inputs["attention_mask"], encoded_inputs["token_type_ids"])
            )

        label_id = label_map[example.label]

        if i < 2:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                logger.info("tokens: %s" % ' '.join(tokens))
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                logger.info("label: {}".format(label_id))

        features.append(
            InputFeatures(choices_features=choices_features,
                          label_id=label_id)
        )

    return features


class Metrics:
    @staticmethod
    def acc(predictions, labels):
        return mtc.accuracy_score(labels, predictions)

    @staticmethod
    def mcc(predictions, labels):
        return mtc.matthews_corrcoef(labels, predictions)

    @staticmethod
    def f1(predictions, labels, average="micro"):
        return mtc.f1_score(labels, predictions, average=average)


def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="data/glue",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="dream",
                        help="Name of the training task.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-uncased",
                        help="Pre-trained language model to load.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="creat_model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_state_dict", type=str, default="",
                        help="Trained model weights to load for evaluation.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--eval_on", type=str, default="test",
                        help="Whether to evaluate on the test set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="Scheduler type for learning rate warmup.")
    parser.add_argument("--warmup_proportion", type=float, default=0.06,
                        help="Proportion of training to perform learning rate warmup for.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    # parser.add_argument("--creat", action="store_true",
    #                     help="To do CreAT if passed.")
    parser.add_argument("--trainer_name", default="creat",
                        help="trainer_strategy")
    parser.add_argument("--adv_steps", type=int, default=2,
                        help="Inner ascent steps for AT.")
    parser.add_argument("--adv_lr", type=float, default=1e-1,
                        help="Step size for AT.")
    parser.add_argument("--adv_max_norm", type=float, default=1e-1,
                        help="Decision boundary for AT.")
    parser.add_argument("--adv_temp", type=float, default=1.0,
                        help="Temperature coefficient for AT.")

    args = parser.parse_args()
    args.output_dir = os.path.join("defense_model",args.trainer_name,f"{args.task_name}_bert")
    wandb.init(project="asa_defense", config=args)
    processors = {
        "dream": DreamProcessor,
        "hellaswag": HellaswagProcessor,
        "alphanli": AlphanliProcessor,
        "race": RaceProcessor,
        "reclor": ReclorProcessor
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, "train_args.bin"))

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=not args.use_slow_tokenizer)

    if args.do_train:
        def select_field(features, field):
            return [[choice[field] for choice in feature.choices_features] for feature in features]

        train_examples = processor.get_train_examples(os.path.join(args.data_dir, task_name))
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor(select_field(train_features, "input_ids"), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, "input_mask"), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, "segment_ids"), dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        model = AutoModelForMultipleChoice.from_pretrained(args.load_model_path,
                                                           num_labels=num_labels,
                                                           return_dict=True,
                                                           cache_dir=cache_dir)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_scheduler(name=args.lr_scheduler_type,
                                  optimizer=optimizer,
                                  num_warmup_steps=args.max_train_steps * args.warmup_proportion,
                                  num_training_steps=args.max_train_steps)

        if args.do_eval:
            if args.eval_on == "dev":
                eval_examples = processor.get_dev_examples(os.path.join(args.data_dir, task_name))
            else:
                eval_examples = processor.get_test_examples(os.path.join(args.data_dir, task_name))
            eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor(select_field(eval_features, "input_ids"), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, "input_mask"), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, "segment_ids"), dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
            )

            # ###################
            # textfooler
            #######################
            eval_data_textfooler = EvalDataset(
                recipe="textfooler",
                task=args.task_name,
                tokenizer=tokenizer,
                args=args,
            )
            eval_sampler_textfooler = SequentialSampler(eval_data_textfooler)
            eval_dataloader_textfooler = DataLoader(
                eval_data_textfooler,
                sampler=eval_sampler_textfooler,
                batch_size=args.eval_batch_size,
            )

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        
        train_dict = {
            "creat": CreATTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                   args.adv_steps, args.adv_lr, args.adv_max_norm, args.adv_temp),
            "freelb": FreeLBTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                   args.adv_steps, args.adv_lr, args.adv_max_norm,),
            "smart" :SMARTTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                   args.adv_steps, args.adv_lr, args.adv_max_norm, args.adv_temp),
            "r3f":  R3FTrainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16,
                                args.adv_max_norm, args.adv_temp),
        }

        if args.trainer_name in train_dict:
            trainer = train_dict[args.trainer_name]
        else: #pgd
            trainer = Trainer(model, optimizer, scheduler, args.max_train_steps, args.gradient_accumulation_steps, args.fp16)
        best_result_bertattack = 0.0
        best_result = 0.0
        best_result_textfooler = 0.0
        for epoch in trange(int(args.num_train_epochs)):

            loss, train_step = trainer.step(train_dataloader)
            global_step = trainer.global_step

            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir,"{}_pytorch_model.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

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
                # args=args,
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
    
    
    main()
