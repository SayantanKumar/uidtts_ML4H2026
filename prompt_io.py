from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import gzip
import random
import argparse
import time
import requests
import re
from call_lcpp import query_llama_cpp
import json
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from pathlib import Path

def query_openai_chat(messages: List[Dict], server_url: str, max_tokens: int = 100000, extra_body: Dict = {}, model_name='', enforce_tag_structure=True) -> str:
    """Query OpenAI-compatible chat completions endpoint."""
    url = f"{server_url}/v1/chat/completions"
    if enforce_tag_structure:
        extra_body['structured_outputs'] = {
            # 'regex': r'^(?:[^<]|<tag id="[^"<>]{1,64}">[^<]{0,5000}</tag>)+$'
            # 'regex': "^(?:[^<]|<tag_(?:[1-9][0-9]*) mention=".*>[^<]{0,5000}</tag_(?:[1-9][0-9]*)>)+$"
            # 'regex': "^(?:[^<]|<tag_([1-9][0-9]*)(?:\\s+mention=\"[^\"]*\")?>[^<]{0,5000}</tag_\\1>)+$"
            # 'regex': r"^(?:[^<]|<tag_([1-9][0-9]*)\s+mention=\"[^\"]*\">[^<]{0,5000}</tag_\1>)+$"
            'regex': r'^(?:[^<]|<tag_([1-9][0-9]*)\s+mention="(?:(?!</tag_[1-9][0-9]*>)[^"])*">[^<]{0,5000}</tag_\1>)+$'
        }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": min(max_tokens, 127500),  # Cap to avoid server issues
        "temperature": 0.7,
        # "repeat_penalty": 1.18,  # These are also in query_llama_cpp
        "repeat_last_n": 256,
        "presence_penalty": 0.6,
        "extra_body": extra_body
    }
    
    try:
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            response_data = json.loads(r.text)
            return response_data['choices'][0]['message']['content']
        else:
            print(f"OpenAI endpoint error: {r.status_code}")
            return ""
    except Exception as e:
        print(f"Error querying OpenAI endpoint: {e}")
        return ""

def query_openai_chat2(messages: List[Dict], server_url: str, max_tokens: int = 100000, extra_body: Dict = {}, model_name='', enforce_tag_structure=True) -> str:
    """Query via OpenAI python module."""
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{server_url}/v1",
        timeout=10_000
    )

    models = client.models.list()
    model = models.data[0].id
    # Use the model ID from the server (respects --served-model-name in vLLM)
    model_name = model

    if enforce_tag_structure:
        extra_body['structured_outputs'] = {
            # 'regex': r'^(?:[^<]|<tag id="[^"<>]{1,64}">[^<]{0,5000}</tag>)+$'
            # 'regex': "^(?:[^<]|<tag_(?:[1-9][0-9]*)>[^<]{0,5000}</tag_(?:[1-9][0-9]*)>)+$"
            # 'regex': r'^(?:[^<]|<tag_([1-9][0-9]*)\s+mention="[^"<]*">[^<]{0,5000}</tag_\1>)+$'
            'regex': r'^(?:[^<]|<tag_([1-9][0-9]*)\s+mention="[^"<]*">[^<]{0,5000}</tag_(?:[1-9][0-9]*)>)+$'
            # 'regex': r'^(?:[^<]|<tag_([1-9][0-9]*)\s+mention="[^"<]*">[^<]{0,5000}</tag_\1>)+$'. # backreferences are not supported
        }
    if "DeepSeek-V3.2" in model_name:
        extra_body['separate_reasoning'] = True
        extra_body["chat_template_kwargs"] = {"thinking": True}
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": min(max_tokens, 127500),  # Cap to avoid server issues
        "temperature": 0.7,
        "frequency_penalty": 0.01,
        # "repeat_last_n": 256,
        "presence_penalty": 0.6,
        "extra_body": extra_body
    }

    # For granite, add: `extra_body={"chat_template_kwargs": {"thinking": True}}`
    response = client.chat.completions.create(**payload)
    try:
        reasoning = response.choices[0].message.reasoning_content
        print(f"Reason: {reasoning[:200]}")
    except Exception:
        print("No reasoning")
    content = response.choices[0].message.content
    return content
    # try:
    #     r = requests.post(url, json=payload)
    #     if r.status_code == 200:
    #         response_data = json.loads(r.text)
    #         return response_data['choices'][0]['message']['content']
    #     else:
    #         print(f"OpenAI endpoint error: {r.status_code}")
    #         return ""
    # except Exception as e:
    #     print(f"Error querying OpenAI endpoint: {e}")
    #     return ""

TAG_TOKEN_RE = re.compile(r'(</?)tag_(\d+)(?=[\s>])')

def has_nonblank_output(path: str, is_gzipped: bool = False) -> bool:
    """Return True only when an existing output has non-whitespace content."""
    output_path = Path(path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False

    try:
        opener = gzip.open if is_gzipped else open
        with opener(output_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False

    return False

def read_case_text(path: str, input_type: str, encoding: str = 'utf-8') -> str:
    if input_type == "txt":
        with open(path, 'r', encoding=encoding) as file:
            return file.read()
    with gzip.open(path, 'rt', encoding=encoding) as file:
        return file.read()

def split_case_text(raw_text: str,
                    max_input_lines: int,
                    chunk_overlength: bool,
                    chunk_line_count: int) -> Optional[List[str]]:
    """Return one or more prompt-sized chunks while preserving old newline handling."""
    num_lines = raw_text.count("\n")

    if max_input_lines > 0 and num_lines > max_input_lines:
        if not chunk_overlength:
            return None

        lines = raw_text.splitlines()
        chunks = []
        for start in range(0, len(lines), chunk_line_count):
            chunk = ''.join(lines[start:start + chunk_line_count]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    return [raw_text.replace('\n', '')]

def renumber_tag_ids(text: str, next_tag_id: int) -> Tuple[str, int]:
    """
    Renumber LLM tag ids so concatenated chunk outputs have globally unique tags.
    Step 2 later maps tag_number -> uid4, so duplicate tag_1 values across chunks
    would otherwise overwrite each other.
    """
    mapping: Dict[str, int] = {}

    def replace(match):
        nonlocal next_tag_id
        tag_id = match.group(2)
        if tag_id not in mapping:
            mapping[tag_id] = next_tag_id
            next_tag_id += 1
        return f"{match.group(1)}tag_{mapping[tag_id]}"

    return TAG_TOKEN_RE.sub(replace, text), next_tag_id

def run(args):
    in_dir = args.in_dir
    model_name = args.model_name
    cache_dir = args.cache_dir
    batch_size = args.batch_size
    llm_mechanism = args.llm_mechanism
    lcpp_url = args.lcpp_url
    case_list = args.case_list
    prompt_file = args.prompt_file
    wrapper_tokens = args.wrapper_tokens
    extra_body = {} if args.extra_body == '' else json.loads(args.extra_body)
    out_extension = args.out_extension
    zip_output = out_extension.endswith('.gz')
    endpoint_type = args.endpoint_type

    # Load prompt from file
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()

    # Build prefix/suffix only if wrapper_tokens is True
    if wrapper_tokens:
        prefix = f"<|begin_of_text|></think>\n{prompt_text}\n<|eot_id|></think>\n"
        suffix = "\n<|eot_id|></think>\n"
    else:
        prefix = prompt_text + "\n"
        suffix = ""

    if llm_mechanism == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        device = 'cuda'
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            device_map="auto",
        )

    import os

    txtfiles_dir = in_dir + '/'

    # Determine input extension based on input_type flag
    input_extension = ".txt" if args.input_type == "txt" else ".txt.gz"
    
    if case_list is None:
        case_filenames = [f for f in os.listdir(txtfiles_dir) if f.endswith(input_extension)]
    else:
        with open(case_list, 'r') as file:
            case_filenames = [f for f in file.read().split("\n") if f.strip() != '']
    random.shuffle(case_filenames)

    # Use custom out_dir if provided, otherwise use default path
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = in_dir + '/../' + model_name + '/'
    
    if out_dir == txtfiles_dir:
        exit()
    os.makedirs(out_dir, exist_ok=True)

    max_new_tokens = 1e5
    case_fn_batch = []
    case_txt_batch = []

    base_ext = out_extension.rsplit('.gz', 1)[0]  # strip .gz suffix for stem
    if zip_output:
        out_suffix = base_ext + '.gz'
        open_func = gzip.open
    else:
        out_suffix = base_ext
        open_func = open

    # for overcompletei in range(5):
    for idx, case in enumerate(case_filenames):
        out_file = out_dir + '/' + os.path.splitext(case)[0] + out_suffix
        if has_nonblank_output(out_file, zip_output):
            print('Skipping ', idx, "/", len(case_filenames), ": ", case, ". Already annotated")
            continue
        if os.path.exists(out_file):
            print(f"Existing output for {case} is empty or invalid; regenerating")
        print(idx, "/", len(case_filenames), ": ", case)

        try:
            case_txt = read_case_text(txtfiles_dir + case, args.input_type)
        except:
            case_txt = read_case_text(txtfiles_dir + case, args.input_type, encoding='unicode_escape')

        num_lines = case_txt.count("\n")
        case_chunks = split_case_text(
            case_txt,
            args.max_input_lines,
            args.chunk_overlength,
            args.chunk_line_count,
        )
        if case_chunks is None:
            print(f'{case} is overlength ({num_lines}); skipping')
            continue
        if len(case_chunks) > 1:
            print(f'{case} is overlength ({num_lines}); processing in {len(case_chunks)} chunks')
        case_txt_batch.append(case_chunks)
        case_fn_batch.append(case)

        if len(case_txt_batch) < batch_size and idx != len(case_filenames) - 1:
            continue

        queries = [[prefix + chunk + suffix for chunk in chunks] for chunks in case_txt_batch]

        print('Passing batch to pipeline: ', ', '.join(case_fn_batch))

        out_files = []
        if llm_mechanism == "hf":
            if any(len(case_queries) > 1 for case_queries in queries):
                raise ValueError("Chunked overlength cases are currently supported only with llama.cpp")
            flat_queries = [case_queries[0] for case_queries in queries]
            for case_fn in case_fn_batch:
                out_files.append(out_dir + '/' + os.path.splitext(case_fn)[0] + out_suffix)
            if batch_size > 1:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenized_batch = tokenizer(
                    flat_queries,
                    padding=True,
                    padding_side='left',
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                start_time = time.perf_counter()
                generated_ids = model.generate(
                    input_ids=tokenized_batch["input_ids"],
                    attention_mask=tokenized_batch["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )
                end_time = time.perf_counter()
                print(f'Generation took {end_time - start_time:.4f} seconds')

                streams = tokenizer.batch_decode(
                    generated_ids[:, tokenized_batch['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                sequences = pipeline(
                    flat_queries[0],
                    do_sample=True,
                    top_k=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                )
                streams = [seq['generated_text'][len(flat_queries[0]):] for seq in sequences]

        elif llm_mechanism == "llama.cpp":
            streams = []
            for bi, case_fn in enumerate(case_fn_batch):
                out_file = out_dir + '/' + os.path.splitext(case_fn_batch[bi])[0] + out_suffix
                if has_nonblank_output(out_file, zip_output):
                    print(f'Skipping {out_file}; exists')
                    continue
                with open_func(out_file, "wt") as text_file:
                        text_file.write('')  # placeholder for concurrency
                out_files.append(out_file)

                chunk_streams = []
                next_tag_id = 1
                for chunk_idx, query in enumerate(queries[bi], start=1):
                    print(f'Querying {case_fn}, chunk {chunk_idx}/{len(queries[bi])}')
                    streami = ''
                    if endpoint_type == "openai":
                        oai_prompt = [
                            {"role": "system", "content": "You are a top-shelf, underrated clinician at the pinnacle of your ability. Be concise and comprehensive. Do not skip any content."},
                            {"role": "user", "content": query}
                        ]
                        streami = query_openai_chat2(oai_prompt, server_url=lcpp_url, max_tokens=max_new_tokens, model_name=model_name, extra_body=extra_body)
                    else:  # completion endpoint
                        completion_url = f"{lcpp_url}/completion"
                        for wsi in range(10):
                            try:
                                streami = query_llama_cpp(query, server_url=completion_url, n_predict=max_new_tokens, extra_args=extra_body)
                                if streami.strip():
                                    break
                            except:
                                print('Process error (ctx window?)')
                                streami = ''
                                break
                            print(f'Whitespace output ({wsi}/10)')

                    streami = streami.replace('.*</think>','') if streami is not None else streami
                    if streami and streami.strip():
                        streami, next_tag_id = renumber_tag_ids(streami, next_tag_id)
                        chunk_streams.append(streami)
                    else:
                        print(f'Issue with {case_fn}, chunk {chunk_idx}; empty llm response.')

                streams.append("\n\n".join(chunk_streams))

        for case_fn, ofile, stream in zip(case_fn_batch, out_files, streams):
            if stream is None or not stream.strip():
                print(f'Issue with {case_fn}; empty llm response.')
                path = Path(ofile)
                if path.exists():
                    path.unlink()
                continue
            with open_func(ofile, "wt") as text_file:
                text_file.write(stream)
            # Delete if empty
            path = Path(ofile)
            if path.exists() and path.stat().st_size == 0:
                path.unlink()

        case_txt_batch = []
        case_fn_batch = []

        if idx == len(case_filenames) - 1:
            break

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for generic LLM prompting.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/DeepSeek-R1-0528-GGUF",
        help="Name of the model to use."
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="/data/weissjc/workspace/pmoa_man/pmoa241217/body/",
        help="Directory for annotating files (outputs to a subfolder)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Custom output directory. If not specified, uses in_dir/../model_name/"
    )
    parser.add_argument(
        "--case_list",
        type=str,
        default=None,
        help="Filename with caselist (one file per row), a subset of the in_dir files"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data/weissjc/.cache/huggingface/hub/",
        help="Directory for caching model files."
    )
    parser.add_argument(
        "--llm_mechanism",
        type=str,
        default="llama.cpp",
        help="Mechanism for querying the llm (hf, llama.cpp)"
    )
    parser.add_argument(
        "--lcpp_url",
        type=str,
        default="http://127.0.0.1:8080",
        help="Llama server url"
    )
    parser.add_argument(
        "--endpoint_type",
        type=str,
        choices=["completion", "openai"],
        default="completion",
        help="Endpoint type to use with llama.cpp (completion or openai)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to the plain-text prompt file (no special tokens)."
    )
    parser.add_argument(
        "--wrapper_tokens",
        action='store_true',
        default=False,
        help="Wrap prompt with special tokens (e.g., </think>, </think>, </think>). Default False."
    )
    parser.add_argument(
        "--extra_body",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--out_extension",
        type=str,
        default=".txt.gz",
        help="Output file extension (e.g., .txt.gz, .bsv, .jsonl). If ends with .gz, output is gzipped."
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["txt", "txt.gz"],
        default="txt",
        help="Input file type to process (.txt or .txt.gz)"
    )
    parser.add_argument(
        "--max_input_lines",
        type=int,
        default=500,
        help="Maximum input newlines before a file is considered overlength. Set <=0 to disable."
    )
    parser.add_argument(
        "--chunk_overlength",
        action="store_true",
        default=False,
        help="Process overlength files in chunks instead of skipping them."
    )
    parser.add_argument(
        "--chunk_line_count",
        type=int,
        default=450,
        help="Number of original lines per overlength chunk when --chunk_overlength is used."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(f"LLM Mechanism: {args.llm_mechanism}")
    print(f"In folder: {args.in_dir}")
    print(f"Out folder: {args.out_dir if args.out_dir is not None else args.in_dir + '/../' + args.model_name + '/'}")
    print(f"Custom case list: {args.case_list}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Wrapper tokens: {args.wrapper_tokens}")
    print(f"Output extension: {args.out_extension}")
    print(f"Input type: {args.input_type}")
    print(f"Max input lines: {args.max_input_lines}")
    print(f"Chunk overlength: {args.chunk_overlength}")
    print(f"Chunk line count: {args.chunk_line_count}")
    print(f'Endpoint type: {args.endpoint_type}')
    if args.llm_mechanism == "hf":
        print(f"Model Name: {args.model_name}")
        print(f"Cache Directory: {args.cache_dir}")
        print(f"Batch size: {args.batch_size}")
    elif args.llm_mechanism == "llama.cpp":
        print(f'Model name: {args.model_name}')
        print(f"Batch size: 1")
    run(args)
