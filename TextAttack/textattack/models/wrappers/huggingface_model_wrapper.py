"""
HuggingFace Model Wrapper
--------------------------
"""

import textattack
import torch
import transformers
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

from .pytorch_model_wrapper import PyTorchModelWrapper

torch.cuda.empty_cache()
torch.manual_seed(43)

class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, attack_args=None):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        self.attack_args = attack_args
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        # max_length = (
        #     512
        #     if self.tokenizer.model_max_length == int(1e30)
        #     else self.tokenizer.model_max_length
        # )
        max_length = 128
        ismcq = isinstance(text_input_list[0],list) and len(text_input_list[0]) > 2
        if ismcq:
            text_input_list_temp = []
            for input_list in text_input_list:
                result = [input_list[i:i+2] for i in range(0, len(input_list), 2)]
                for context, ending in result:
                    text_a = context
                    if ending.find("_") != -1:
                        text_b = ending.replace("_", ending)
                    else:
                        text_b = f"{ending}"
                    text_input_list_temp.append((text_a,text_b))
            inputs_dict = self.tokenizer(
                text_input_list_temp,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                
                add_special_tokens=True,
                return_tensors="pt",
            )
            inputs_dict["input_ids"] = inputs_dict["input_ids"].reshape(len(text_input_list),-1,max_length)
            inputs_dict["attention_mask"] = inputs_dict["attention_mask"].reshape(len(text_input_list),-1,max_length)
            inputs_dict["token_type_ids"] = inputs_dict["token_type_ids"].reshape(len(text_input_list),-1,max_length)
            
        else:
            inputs_dict = self.tokenizer(
                text_input_list,
                add_special_tokens=True,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)
        # print(text_input_list)
        with torch.no_grad():
            outputs = self.model(
                **inputs_dict,
                args=self.attack_args,
                # output_hidden_states=False,
                output_attentions=True,
                # # head_mask=head_mask,
                # return_dict=False,
            )
        # print(outputs)
        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            # print(outputs[1])
            # input()
            return outputs[1].detach().cpu() if outputs[0] is None else outputs[0].detach().cpu()

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
