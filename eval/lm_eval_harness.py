from contextlib import nullcontext
from typing import Iterable

import torch

import lm_eval
# from lm_eval.base import BaseLM
# from lm_eval.evaluator import evaluate, make_table

from lm_eval.api.model import TemplateLM
from lm_eval.utils import make_table
from lm_eval.models.utils import Collator

from lm_eval import utils
import os
import json
import lm_eval

from model import GPT
from tokenizer import Tokenizer

from tqdm import tqdm
import torch.nn.functional as F

def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []

    if arr:
        yield arr

class NanoGPTModel(TemplateLM):

    def __init__(self, model: GPT, tokenizer: Tokenizer, device="cuda", temperature=0.8, top_k=200,
                 max_gen_tokens=128, batch_size=1, eot_token=50256):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_gen_tokens = max_gen_tokens
        self._batch_size = batch_size
        self._eot_token = eot_token
        self._temperature = temperature
        self._top_k = top_k
        # Store the block_size directly to avoid accessing it through the wrapped model
        self._block_size = model.module.config.block_size if hasattr(model, 'module') else model.config.block_size

    @property
    def eot_token_id(self):
        return self._eot_token

    @property
    def max_length(self):
        return self._block_size  # Use the stored block_size instead of accessing through model

    @property
    def max_gen_toks(self):
        return self._max_gen_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self._tokenizer.encode(string)

    def tok_decode(self, tokens: Iterable[int]):
        return self._tokenizer.decode(list(tokens))

    def _model_generate(self, context, max_length, eos_token_id):
        # Access the underlying model through .module if it's wrapped in DDP
        if hasattr(self._model, 'module'):
            return self._model.module.generate(context, max_length, temperature=self._temperature, top_k=self._top_k, eos_token=eos_token_id)
        else:
            return self._model.generate(context, max_length, temperature=self._temperature, top_k=self._top_k, eos_token=eos_token_id)

    def _model_call(self, inps):
        targets = torch.zeros_like(inps) - 1
        return self._model(inps, targets=targets)[0]

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(self.rank != 0),
            )
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=(self.rank != 0)
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # TODO: automatic (variable) batch size detection for vectorization
        re_ord = utils.Reorderer(requests, _collate)
        for chunk in chunks(
            tqdm(re_ord.get_reordered(), disable=disable_tqdm), self.batch_size
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)

    def generate_until(self, requests):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]
        # def _collate(req: Tuple[str, dict]):
        #     """Defines the key for the sorted method"""
        #     # the negative sign on len(toks) sorts descending - this has a few advantages:
        #     # - time estimates will always be over not underestimates, which is more useful for planning
        #     # - to know the size of a batch when going through the list, you know the first one is always the batch
        #     #   padded context length. this is useful to simplify the batching logic and more importantly to make
        #     #   automatic adaptive batches much much easier to implement
        #     # - any OOMs will happen right away rather than near the end
        #     toks = self.tok_encode(req[0])
        #     return -len(toks), req[0]

        re_ord = utils.Reorderer([reg.args for reg in requests], _collate)

        for context, until in tqdm(re_ord.get_reordered()):
            if isinstance(until, str):
                until = [until]
            if isinstance(until, dict):
                until = until['until']

            (primary_until,) = self.tok_encode(until[0])

            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)

            cont = self._model_generate(
                context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until
            )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("generate_until", (context, until), s)

            res.append(s)

        return re_ord.get_original(res)



if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out'  # ignored if init_from is not 'resume'
    max_new_tokens = 1024  # number of tokens generated in each sample
    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    batch_size = 256
    dtype = 'float16' # 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
    tasks = ["mmlu_continuation", "lambada_openai", "hellaswag", "gpqa_main_zeroshot", "winogrande", "wikitext", "arc_easy", "arc_challenge", "piqa", "social_iqa", "truthfulqa"]  # ,  
    exec(open('configurator.py').read())  # overrides from command line or config file
    # -----------------------------------------------------------------------------

    # Initialize accelerator
    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device.type else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model, tokenizer = GPT.init_from(init_from, out_dir=out_dir, device=device)
    model.eval()
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)

    with torch.no_grad():
        # Only the main process should save results
        is_main_process = accelerator.is_main_process
        
        full_results = lm_eval.simple_evaluate(
            model=NanoGPTModel(model, tokenizer, device=device, max_gen_tokens=max_new_tokens, temperature=temperature, top_k=top_k),
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size, 
            log_samples=True,
            write_out=True,
            # limit=10,
            # use_cache=out_dir+f"/cache",
        )
        
        # Only save results on the main process
        if is_main_process:
            results = list(full_results['results'].values())
            samples = full_results['samples']
            print(f"results = {results}")

            # Use out_dir instead of args.output_dir
            metrics_file = os.path.join(out_dir, f"metrics_{'_'.join(tasks)}_{0}.json")
            print(f"metrics_file = {metrics_file}")
            # save metrics to metrics_file (if exists, overwrite)
            with open(metrics_file, 'w') as f:
                json.dump(results, f)
            # save samples
            samples_file = os.path.join(out_dir, f"samples_{'_'.join(tasks)}_{0}.json")
            with open(samples_file, 'w') as f:
                json.dump(samples, f)