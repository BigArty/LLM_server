import os
from sys import path
from abc import ABC, abstractmethod
import glob
import time
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

path += [os.path.abspath(r"exllama")]
from transformers import (
    AutoConfig,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    logging,
    pipeline,
)
from huggingface_hub import snapshot_download
from huggingface_hub import scan_cache_dir
from exllama.tokenizer import ExLlamaTokenizer
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.generator import ExLlamaGenerator
from exllama.alt_generator import ExLlamaAltGenerator
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch


class ChatRequest(BaseModel):
    messages: list[str]
    temperature: float = 0
    top_p: float = 1
    n: int = 1
    max_tokens: int = 1


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: dict


def get_path(model_name_or_path):
    hf_cache_info = scan_cache_dir()
    for i in hf_cache_info.repos:
        if model_name_or_path == i.repo_id:
            for j in i.revisions:
                return j.snapshot_path
    snapshot_download(model_name_or_path)
    return get_path(model_name_or_path)


class ModelTemplate(ABC):
    @abstractmethod
    def generate(self,
                 prompt,
                 max_new_tokens=1024,
                 stop_conditions=None,
                 temperature=0.1,
                 top_p=0.9,
                 top_k=50,
                 typical=0, ):
        pass


class Model(ModelTemplate):
    def make_prompt(self, messages, max_length=None):
        if "WizardCoder" in self.name:
            strings = ["<s>"]
            for message in messages["history"]:
                if message["role"] == "system":
                    strings.append(message["content"]+"\n\n")
                elif message["role"] == "user":
                    strings.append("### Instruction:\n" +
                                   message["content"]+"\n\n")
                elif message["role"] == "assistant":
                    strings.append("### Response:\n" +
                                   message["content"]+"\n\n")
            strings.append("### Response:\n")

            return "".join(strings)

        elif "Phind" in self.name:
            raise NotImplementedError()
            main_format = "<s>{history:list}### Assistant\n"
            data_format = {
                "history": "{history:condition_dict:role}",
                "history:system": "### System Prompt\n{content}\n\n",
                "history:user": "### User Message\n{content}\n\n",
                "history:assistant": "### Assistant\n{content}\n\n",
            }
        elif "CodeLlama" in self.name:
            raise NotImplementedError()
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

            SPECIAL_TAGS_DICT = {"B_INST": B_INST,
                                 "E_INST": E_INST, "B_SYS": B_SYS, "E_SYS": E_SYS}
            main_format = "<s>[INST] {history:list}"
            data_format = {
                "history": "{history:condition_dict:role} ",
                "history:system": "<<SYS>>\n{content}\n<</SYS>>\n\n",
                "history:user": "{content} [/INST]",
                "history:assistant": " {content} </s><s>[INST] ",
            }
            data_format.update(SPECIAL_TAGS_DICT)
        elif "MegaCoder" in self.name:
            raise NotImplementedError()
            main_format = "{history:list}<|im_start|>assistant\n"
            data_format = {
                "history": "{history:condition_dict:role}",
                "history:system": "<|im_start|>system\n{content}<|im_end|>\n",
                "history:user": "<|im_start|>user\n{content}<|im_end|>\n",
                "history:assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
            }
        else:
            raise ReferenceError("Unknown model prompt type")

    def get_len(self, string):
        if self.backend == "exllama":
            return len(self.tokenizer.encode(string, encode_special_characters=True).reshape(-1))
        else:
            return len(self.tokenizer.encode(string, add_special_tokens=False))

    def generate(self,
                 prompt,
                 max_new_tokens=1024,
                 stop_conditions=None,
                 temperature=0.1,
                 top_p=0.9,
                 top_k=50,
                 typical=0, ):
        if stop_conditions is None:
            stop_conditions = [
                "##End##",
                "[INST]",
                "[USER]",
                self.tokenizer.eos_token,
                self.tokenizer.eos_token_id,
                "\n\n\n",
                "[/INST]",
                "[/USER]",
                "###",
                '<|im_end|>',
            ]
        if self.backend == "exllama":
            # Configure generator
            settings = ExLlamaAltGenerator.Settings()
            settings.token_repetition_penalty_max = 1
            settings.temperature = temperature
            settings.top_p = top_p
            settings.top_k = top_k
            settings.typical = typical

            # Produce a simple generation
            st = time.time()
            output = self.generator.generate(
                prompt=prompt,
                stop_conditions=stop_conditions,
                max_new_tokens=max_new_tokens,
                gen_settings=settings,
                encode_special_characters=True,
            ).strip()
            et = time.time()
            generation_time = et - st
            output.replace("[PYTHON]", "```python")
            output.replace("[/PYTHON]", "```")
            return output, generation_time
        if self.backend == "gptq":
            if typical == 0:
                typical = 1
            batch = self.tokenizer(prompt, return_tensors="pt")
            batch["input_ids"] = batch["input_ids"].cuda()

            # Produce a simple generation
            with torch.cuda.amp.autocast():
                st = time.time()
                output_tokens = self.model.generate(
                    **batch,
                    generation_config=self.config,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_beams=1,
                    typical_p=typical
                )
                et = time.time()
            generation_time = et - st
            output = self.tokenizer.decode(
                output_tokens[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True)
            return output, generation_time

    def generate_beam(self,
                      prompt,
                      max_new_tokens=1024,
                      stop_conditions=None,
                      temperature=0.5,
                      top_p=0.9,
                      top_k=50,
                      typical=0,
                      beams=5,
                      beam_length=10, ):
        if stop_conditions is None:
            stop_conditions = [
                "##End##",
                "[INST]",
                "[USER]",
                self.tokenizer.eos_token,
                self.tokenizer.eos_token_id,
                "\n\n\n",
                "[/INST]",
                "[/USER]",
                "###",
                '<|im_end|>',
            ]

        stop_strings = []
        stop_tokens = []
        for t in stop_conditions:
            if isinstance(t, int):
                stop_tokens += [t]
            elif isinstance(t, str):
                stop_strings += [t]

        min_expected_complition = 256
        self.generator_beam.settings = ExLlamaGenerator.Settings()
        self.generator_beam.settings.token_repetition_penalty_max = 1
        self.generator_beam.settings.token_repetition_penalty_sustain = 256
        self.generator_beam.settings.token_repetition_penalty_decay = self.generator_beam.settings.token_repetition_penalty_sustain // 2
        self.generator_beam.disallow_tokens(None)

        self.generator_beam.settings.temperature = temperature
        self.generator_beam.settings.top_k = top_k
        self.generator_beam.settings.top_p = top_p
        self.generator_beam.settings.typical = typical
        self.generator_beam.settings.beams = beams
        self.generator_beam.settings.beam_length = beam_length

        st = time.time()
        ids = self.tokenizer.encode(prompt, encode_special_characters=True)
        ids = ids[:, -self.config.max_seq_len + min_expected_complition:]
        self.generator_beam.gen_begin_reuse(ids)
        # self.generator_beam.gen_begin(ids)

        res_line = ""

        # SentencePiece doesn't tokenize spaces separately, so we can't know from individual tokens if they start a new word
        # or not. Instead, repeatedly decode the generated response as it's being built, starting from the last newline,
        # and print out the differences between consecutive decodings to stream out the response.

        # If we're approaching the context limit, prune some whole lines from the start of the context. Also prune a
        # little extra, so we don't end up rebuilding the cache on every line when up against the limit.

        self.generator_beam.begin_beam_search()
        num_res_tokens = 0  # res_tokens.shape[-1]  # Decode from here

        for i in range(max_new_tokens):
            if self.generator_beam.gen_num_tokens() >= (self.config.max_seq_len - beam_length):
                self.generator_beam.end_beam_search()
                print("Cutting: ", self.generator_beam.gen_num_tokens(), end='')
                self.generator_beam.gen_begin(
                    self.generator_beam.sequence_actual[:,
                                                        -(self.config.max_seq_len - min_expected_complition - beam_length):]
                )
                print(". After: ", self.generator_beam.gen_num_tokens())
                self.generator_beam.begin_beam_search()
            # Get a token

            gen_token = self.generator_beam.beam_search()

            # Decode the current line and print any characters added

            num_res_tokens += 1
            text = self.tokenizer.decode(
                self.generator_beam.sequence_actual[:, -num_res_tokens:][0], )
            # new_text = text[len(res_line):]
            # print(new_text, end="")  # (character streaming output is here)

            res_line = text

            # End conditions

            if gen_token.item() in stop_tokens:
                break

            time_to_stop = False
            for ss in stop_strings:
                if ss in res_line:
                    res_line = res_line.split(ss)[0]
                    time_to_stop = True
            if time_to_stop:
                break

        self.generator_beam.end_beam_search()
        et = time.time()

        return res_line, et - st

    def __init__(self, model_name, backend="exllama"):
        self.backend = backend
        self.name = model_name
        if backend == "exllama":
            # Directory containing model, tokenizer, generator

            model_directory = get_path(model_name)
            print(model_directory)
            # Locate files we need within that directory

            tokenizer_path = os.path.join(model_directory, "tokenizer.model")
            model_config_path = os.path.join(model_directory, "config.json")
            st_pattern = os.path.join(model_directory, "*.safetensors")
            model_path = glob.glob(st_pattern)[0]

            # Create config, model, tokenizer and generator

            # create config from config.json
            self.config = ExLlamaConfig(model_config_path)
            self.config.model_path = model_path  # supply path to model weights file
            # if "WizardCoder" in model_name:
            self.config.max_seq_len = 4096

            # create ExLlama instance and load the weights
            self.model = ExLlama(self.config)
            # create tokenizer from tokenizer model file
            self.tokenizer = ExLlamaTokenizer(tokenizer_path)

            self.cache = ExLlamaCache(self.model)  # create cache for inference
            self.generator = ExLlamaAltGenerator(
                self.model, self.tokenizer, self.cache)  # create generator

            self.generator_beam = ExLlamaGenerator(
                self.model, self.tokenizer, self.cache)
            self.generator_beam.settings = ExLlamaGenerator.Settings()
            self.generator_beam.settings.token_repetition_penalty_max = 1
            self.generator_beam.settings.token_repetition_penalty_sustain = 256
            self.generator_beam.settings.token_repetition_penalty_decay = self.generator_beam.settings.token_repetition_penalty_sustain // 2
            self.generator_beam.disallow_tokens(None)

        if backend == "gptq":
            model_name_or_path = model_name

            use_triton = False

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, use_fast=True)
            self.config = GenerationConfig.from_pretrained(model_name_or_path, do_sample=True,
                                                           max_new_tokens=1500,
                                                           num_beams=1,
                                                           temperature=0.5,
                                                           top_p=0.7,
                                                           top_k=50, )

            self.model = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path,
                use_safetensors=True,
                trust_remote_code=True,
                device_map="auto",
                use_triton=use_triton,
                quantize_config=None,
                # inject_fused_attention=False
            )
        if backend == "causal":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = LlamaForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                #     load_in_8bit=True,
                # torch_dtype=torch.float16,
                device_map="auto",
            )

            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            self.backend = "gptq"

    def make_inference(self, messages,
                       max_new_tokens=1024,
                       stop_conditions=None,
                       temperature=0.1,
                       top_p=0.9,
                       top_k=50,
                       typical=0,
                       beams=None,
                       beam_length=10, ):
        prompt = self.make_prompt(messages)
        # print(self.get_len(prompt), "\n=============\n", prompt)
        if beams is None:
            output, generation_time = self.generate(prompt,
                                                    max_new_tokens=max_new_tokens,
                                                    stop_conditions=stop_conditions,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    top_k=top_k,
                                                    typical=typical, )
        else:
            if self.backend == "exllama":
                output, generation_time = self.generate_beam(prompt,
                                                             max_new_tokens=max_new_tokens,
                                                             stop_conditions=stop_conditions,
                                                             temperature=temperature,
                                                             top_p=top_p,
                                                             top_k=top_k,
                                                             typical=typical,
                                                             beams=beams,
                                                             beam_length=beam_length)
            else:
                self.config.num_beams = beams
                self.config.do_sample = (temperature > 0.02)
                output, generation_time = self.generate(prompt,
                                                        max_new_tokens=max_new_tokens,
                                                        stop_conditions=stop_conditions,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        typical=typical)

        print(
            f"{generation_time:.2f} seconds\nPrompt: {self.get_len(prompt)}, output: {self.get_len(output)}\n======================")
        print(prompt, '\n\n\n')
        return output, self.get_len(prompt), self.get_len(output)


app = FastAPI()


class Server:
    def __init__(self, model_name="TheBloke/WizardCoder-Python-13B-V1.0-GPTQ", backend="exllama", port=8088):
        self.model = Model(model_name, backend)
        self.app = app
        self.app.get("/models")(self.models)
        self.app.post("/chat/completions")(self.main)
        self.app.post("/generate")(self.generate)
        self.port = port

    async def models(self):
        try:
            return {
                "object": "list",
                "data": [
                    {
                        "id": "gpt-3.5-turbo-0613",
                        "object": "model",
                        "created": 1686935002,
                        "owned_by": "organization-owner"
                    },
                    {
                        "id": "gpt-asdf",
                        "object": "model",
                        "created": 1686935002,
                        "owned_by": "organization-owner",
                    },
                    {
                        "id": "model-id-2",
                        "object": "model",
                        "created": 1686935002,
                        "owned_by": "openai"
                    },
                ]
            }
        except HTTPException:
            pass

    async def main(self, data: dict = Body(...)):
        try:
            messages = {"history": data["messages"]}
            # history = []
            # syst = False
            # for message in messages["history"]:
            #     if message["role"] != "system" or not syst:
            #         history.append(message)
            #         syst = True
            # messages["history"] = history
            temperature = 0.5  # data.get("temperature", 0) + 0.01
            top_p = 0.5  # data.get("top_p", 1)
            number_of_answers = data.get("n", 1)
            max_tokens = data.get("max_tokens", 1536)
            response, prompt_tokens, out_tokens = self.model.make_inference(
                messages, max_tokens, temperature=temperature, top_p=top_p, typical=0, top_k=50,
                beams=5,
                beam_length=10,
            )

            chat_response = ChatResponse(
                id="chatcmpl-123",
                object="chat.completion",
                created=1677652288,
                model="gpt-3.5-turbo-0613",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop"
                }] * number_of_answers,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": out_tokens,
                    "total_tokens": prompt_tokens + out_tokens
                }
            )
            return chat_response.model_dump()

        except HTTPException:
            print(data)
            raise HTTPException(status_code=400, detail="Invalid request data")

    async def generate(self, data: dict):
        try:
            messages = data['generate']
            temperature = data.get("temperature", 0.1)
            top_p = data.get("top_p", 0.9)
            max_tokens = data.get("max_tokens", 1536)
            response, timer = self.model.generate(
                messages, max_tokens, temperature=temperature, top_p=top_p, typical=0, top_k=50,
                # beams=5,
                # beam_length=10,
            )

            chat_response = {
                "content": response,
            }
            return chat_response

        except HTTPException:
            print(data)
            raise HTTPException(status_code=400, detail="Invalid request data")

    def run(self):
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)


if __name__ == "__main__":

    server = Server("TheBloke/WizardCoder-Python-13B-V1.0-GPTQ", "exllama")
    server.run()
