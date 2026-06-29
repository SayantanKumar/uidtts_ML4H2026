#!/usr/bin/env python3
"""
Process case reports with extracted tags using temporal prompts.
Creates LLM outputs combining case reports with their mention tables and optional context.
"""

import argparse
import gzip
import os
import re
import time
import json
import requests
import random
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from openai import OpenAI
from call_lcpp import query_llama_cpp

def load_extracted_tags(tags_file: str, keep_columns: Optional[List[int]] = None) -> List[Dict[str, str]]:
    """Load tag information from BSV file."""
    tags = []
    open_func = gzip.open if tags_file.endswith('.gz') else open
    
    with open_func(tags_file, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '|' in line:
                parts = line.split('|')
                tag_dict = {}
                if keep_columns is None:
                    # Keep all columns by default
                    for i, part in enumerate(parts):
                        tag_dict[f"col_{i}"] = part.strip()
                else:
                    # Keep only specified columns
                    for col_idx in keep_columns:
                        if col_idx < len(parts):
                            tag_dict[f"col_{col_idx}"] = parts[col_idx].strip()
                tags.append(tag_dict)
    return tags

def load_case_report(case_file: str) -> str:
    """Load case report content."""
    open_func = gzip.open if case_file.endswith('.gz') else open
    with open_func(case_file, 'rt', encoding='utf-8') as f:
        return f.read().strip()

def load_context(context_file: Optional[str]) -> str:
    """Load context content."""
    if not context_file or not os.path.exists(context_file):
        return ""
    open_func = gzip.open if context_file.endswith('.gz') else open
    with open_func(context_file, 'rt', encoding='utf-8') as f:
        return f.read().strip()

def has_nonblank_output(output_file: str) -> bool:
    """Return True only when an existing output can be read and has content."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return False

    open_func = gzip.open if output_file.endswith('.gz') else open
    try:
        with open_func(output_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False

    return False

def remove_invalid_output(output_file: str) -> None:
    """Remove an existing output only when it is empty, whitespace, or unreadable."""
    if os.path.exists(output_file) and not has_nonblank_output(output_file):
        print(f"Removing empty or invalid output: {output_file}")
        os.unlink(output_file)

def format_mention_table(tags: List[Dict[str, str]]) -> str:
    """Format tags as a mention table for the prompt."""
    lines = []
    for tag in tags:
        line_parts = []
        for key, value in tag.items():
            line_parts.append(f"{key}: {value}")
        lines.append(" | ".join(line_parts))
    return '\n'.join(lines)

def build_prompt(case_content: str, mention_table: str, context_content: str,
                prompt_template: str, wrapper_tokens: bool, endpoint_type: str) -> str:
    """Build final prompt with placeholders replaced."""
    prompt = prompt_template.replace('{{tagged_case_report}}', case_content)
    prompt = prompt.replace('{{mention_table}}', mention_table)
    prompt = prompt.replace('{{context}}', context_content)
    
    if endpoint_type == "openai":
        # For OpenAI chat completions, return structured message
        return [
            {"role": "system", "content": "You are a top-shelf, underrated clinician at the pinnacle of your ability. Be concise and comprehensive. Do not skip any rows."},
            {"role": "user", "content": prompt}
        ]
    elif wrapper_tokens:
        prefix = f"<|begin_of_text|><|system|>\n{prompt}\n<|eot_id|><|user|>\n"
        suffix = "\n<|eot_id|><|assistant|>\n"
        return prefix + case_content + suffix
    else:
        return prompt + "\n" + case_content

TEMPORAL_HEADER = "uid4 | mention | time | bounds | known | context uid4s"

def chunk_list(items: List[Dict[str, str]], chunk_size: int) -> List[List[Dict[str, str]]]:
    """Split tag rows into stable chunks; chunk_size <= 0 means no chunking."""
    if chunk_size <= 0 or len(items) <= chunk_size:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def extract_answer_table(content: str) -> str:
    """Extract the BSV table from a temporal LLM response."""
    if not content:
        return ""

    last_answer_start = content.rfind("<answer>")
    if last_answer_start != -1:
        answer_end = content.find("</answer>", last_answer_start)
        if answer_end != -1:
            return content[last_answer_start + len("<answer>"):answer_end].strip()

    for pattern in [r'</think>\s*\n(.*)', r'</thinking>\s*\n(.*)']:
        matches = list(re.finditer(pattern, content, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()

    header_pos = content.find("uid4 | mention | time | bounds | known | context uid4s")
    if header_pos != -1:
        return content[header_pos:].strip()

    return content.strip()

def normalize_temporal_rows(table_content: str) -> List[str]:
    """Keep only temporal BSV data rows and remove duplicate headers/noise."""
    rows = []
    for raw_line in table_content.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        if line.lower() == TEMPORAL_HEADER.lower():
            continue
        if re.match(r'^[\s\-\|:]+$', line):
            continue
        if line.startswith("<") and line.endswith(">"):
            continue

        parts = [part.strip() for part in line.split("|")]
        parts = [part for part in parts if part]
        if len(parts) < 6:
            continue
        rows.append(" | ".join(parts[:6]))
    return rows

def build_merged_response(rows: List[str], chunk_count: int, tag_count: int) -> str:
    """Build one parseable answer block after batched temporal generation."""
    return (
        "<think>\n"
        f"Temporal table generated in {chunk_count} mention-table batches for {tag_count} mentions.\n"
        "</think>\n"
        "<answer>\n"
        f"{TEMPORAL_HEADER}\n"
        + "\n".join(rows)
        + "\n</answer>\n"
    )

def query_openai_chat(messages: List[Dict], server_url: str, max_tokens: int = 100000, model_name='', extra_body=None, enforce_tag_structure=True) -> str:
    """Query OpenAI-compatible chat completions endpoint."""
    # OLD
    # url = f"{server_url}/v1/chat/completions"

    # NEW
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{server_url}/v1",
        timeout=10_000
    )

    models = client.models.list()
    model = models.data[0].id

    request_extra_body = dict(extra_body or {})

    if enforce_tag_structure:
        regex = r"^uid4 \| mention \| time \| bounds \| known \| context uid4s\n(?:[^|\n]* \| [^|\n]* \| [^|\n]* \| [^|\n]* \| [^|\n]* \| [^|\n]*)(?:\n(?:[^|\n]* \| [^|\n]* \| [^|\n]* \| [^|\n]* \| [^|\n]* \| [^|\n]*))*\s*$"
        request_extra_body['structured_outputs'] = {'regex': regex}

    if "DeepSeek-V3.2" in model_name:
        request_extra_body['separate_reasoning'] = True
        request_extra_body["chat_template_kwargs"] = {"thinking": True}

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": min(max_tokens, 125000),  # Cap to avoid server issues
        "temperature": 0.7,
        "frequency_penalty": 0.01,
        # "repeat_penalty": 1.18,  # These are also in query_llama_cpp
        # "repeat_last_n": 256,
        "presence_penalty": 0.6,
        "extra_body": request_extra_body
    }
    
    try:
        response_data = client.chat.completions.create(**payload)
        try:
            # reasoning = response_data['choices'][0]['message']['reasoning_content']
            reasoning = response_data.choices[0].message.reasoning_content
            print(f"Reason: {reasoning[:200]}")
        except Exception as e:
            print(f"No reasoning, due to: {e}")
        content = response_data.choices[0].message.content
        return content
        # return response_data['choices'][0]['message']['content']

        # OLD
        # r = requests.post(url, json=payload)
        # if r.status_code == 200:
        #     response_data = json.loads(r.text)
        #     import pdb
        #     pdb.set_trace()
        #     try:
        #         reasoning = response_data['choices'][0]['message']['reasoning_content']
        #         print(f"Reason: {reasoning[:200]}")
        #     except Exception as e:
        #         print(f"No reasoning, due to: {e}")
        #     return response_data['choices'][0]['message']['content']
        # else:
        #     print(f"OpenAI endpoint error: {r.status_code}")
        #     return ""
    except Exception as e:
        raise RuntimeError(f"Error querying OpenAI endpoint: {e}") from e

def query_model(final_prompt, llm_mechanism: str, model_name: str, cache_dir: str,
                lcpp_url: str, batch_size: int, endpoint_type: str,
                extra_body: Optional[Dict] = None) -> str:
    """Run one prompt through the configured LLM backend."""
    response = ""

    if llm_mechanism == "hf":
        # Initialize model and tokenizer
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

        # Generate response
        max_new_tokens = 100000
        if batch_size > 1:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenized = tokenizer(
                final_prompt,
                padding=True,
                padding_side='left',
                truncation=True,
                return_tensors="pt"
            ).to(device)

            start_time = time.perf_counter()
            generated_ids = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            end_time = time.perf_counter()
            print(f'Generation took {end_time - start_time:.4f} seconds')

            response = tokenizer.decode(
                generated_ids[0, tokenized['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        else:
            sequences = pipeline(
                final_prompt,
                do_sample=True,
                top_k=1,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )
            response = sequences[0]['generated_text'][len(final_prompt):]

    elif llm_mechanism == "llama.cpp":
        if endpoint_type == "openai":
            response = query_openai_chat(
                final_prompt,
                server_url=lcpp_url,
                extra_body=extra_body,
                model_name=model_name
            )
        else:  # completion endpoint
            response = ''
            for wsi in range(10):
                try:
                    response = query_llama_cpp(final_prompt, server_url=lcpp_url, n_predict=127500,
                        extra_args = {
                            "repeat_penalty": 1.18,
                            "repeat_last_n": 256,
                            "presence_penalty": 0.6
                        }
                    )
                    if response.strip():
                        break
                except Exception as exc:
                    raise RuntimeError(f"Error querying llama.cpp completion endpoint: {exc}") from exc
                print(f'Whitespace output ({wsi}/10)')

    if not response or not response.strip():
        raise RuntimeError("LLM returned an empty temporal response")

    return response

def process_single_pair(case_file: str, tags_file: str, context_file: Optional[str],
                       output_file: str, prompt_template: str, llm_mechanism: str,
                       model_name: str, cache_dir: str, lcpp_url: str,
                       wrapper_tokens: bool, batch_size: int = 1,
                       endpoint_type: str = "completion",
                       extra_body: Optional[Dict] = None,
                       keep_columns: Optional[List[int]] = None,
                       tag_chunk_size: int = 0) -> None:
    """Process a single case report with its tags and optional context using LLM."""
    
    # Load inputs
    case_content = load_case_report(case_file)
    tags = load_extracted_tags(tags_file, keep_columns=keep_columns)
    context_content = load_context(context_file)

    tag_batches = chunk_list(tags, tag_chunk_size)
    if len(tag_batches) > 1:
        print(f"Processing {len(tags)} mentions in {len(tag_batches)} temporal batches of up to {tag_chunk_size}.")
        merged_rows = []
        for batch_idx, tag_batch in enumerate(tag_batches, start=1):
            print(f"  Temporal batch {batch_idx}/{len(tag_batches)}: {len(tag_batch)} mentions")
            mention_table = format_mention_table(tag_batch)
            final_prompt = build_prompt(case_content, mention_table, context_content,
                                      prompt_template, wrapper_tokens, endpoint_type)
            batch_response = query_model(final_prompt, llm_mechanism, model_name, cache_dir,
                                         lcpp_url, batch_size, endpoint_type, extra_body)
            batch_table = extract_answer_table(batch_response)
            batch_rows = normalize_temporal_rows(batch_table)
            if not batch_rows:
                raise RuntimeError(f"Temporal batch {batch_idx} returned no parseable BSV rows")
            merged_rows.extend(batch_rows)

        if not merged_rows:
            raise RuntimeError("No temporal rows were generated from batched Step 3 processing")
        response = build_merged_response(merged_rows, len(tag_batches), len(tags))
    else:
        mention_table = format_mention_table(tags)
        final_prompt = build_prompt(case_content, mention_table, context_content,
                                  prompt_template, wrapper_tokens, endpoint_type)
        response = query_model(final_prompt, llm_mechanism, model_name, cache_dir,
                               lcpp_url, batch_size, endpoint_type, extra_body)

    # Write output
    open_func = gzip.open if output_file.endswith('.gz') else open
    with open_func(output_file, 'wt', encoding='utf-8') as f:
        f.write(response)

    if not has_nonblank_output(output_file):
        remove_invalid_output(output_file)
        raise RuntimeError("Temporal output was empty or unreadable after writing")

def process_directories(case_dir: str, tags_dir: str, context_dir: Optional[str],
                       output_dir: str, prompt_template: str, llm_mechanism: str,
                       model_name: str, cache_dir: str, lcpp_url: str,
                       wrapper_tokens: bool, batch_size: int = 1,
                       endpoint_type: str = "completion",
                       extra_body: Optional[Dict] = None, 
                       keep_columns: Optional[List[int]] = None,
                       tag_chunk_size: int = 0) -> None:
    """Process all matching pairs in directories."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all case files
    case_fns = os.listdir(case_dir)
    random.shuffle(case_fns)
    for case_filename in case_fns:
        if case_filename.endswith('.txt.gz') or case_filename.endswith('.txt'):
            base_name = case_filename.replace('.txt.gz', '').replace('.txt', '').replace('_modified','')
            print(base_name)
            case_path = os.path.join(case_dir, case_filename)
            tags_path = os.path.join(tags_dir, f"{base_name}_tags.bsv.gz")
            
            # Check if tags file exists (try gzipped and plain)
            if not os.path.exists(tags_path):
                tags_path = tags_path.replace('.gz', '')
            
            # Check if context file exists (try gzipped and plain)
            context_path = None
            if context_dir:
                context_path = os.path.join(context_dir, f"{base_name}_context.txt.gz")
                if not os.path.exists(context_path):
                    context_path = context_path.replace('.gz', '')
                if not os.path.exists(context_path):
                    context_path = os.path.join(context_dir, f"{base_name}_context.txt")
                    if not os.path.exists(context_path):
                        context_path = None
            
            # print(f'TAGPATH {tags_path} exists {os.path.exists(tags_path)}')
            if os.path.exists(tags_path):
                output_path = os.path.join(output_dir, f"{base_name}.txt.gz")
                # print(f'OUTPATH {output_path} exists {os.path.exists(output_path)}')
                if has_nonblank_output(output_path):
                    print(f"Skipping {output_path}, exists and is non-empty.")
                    continue
                remove_invalid_output(output_path)

                # Put in a placeholder to avoid collisions
                placeholder_success = False
                try:
                    open_func = gzip.open if output_path.endswith('.gz') else open
                    with open_func(output_path, 'wt', encoding='utf-8') as f:
                        f.write('placeholder')
                    placeholder_success = True
                except:
                    pass
                try:
                    print(f"Processing {case_filename}...")
                    process_single_pair(case_path, tags_path, context_path,
                                      output_path, prompt_template, llm_mechanism, model_name,
                                      cache_dir, lcpp_url, wrapper_tokens, batch_size,
                                      endpoint_type, extra_body, keep_columns=keep_columns,
                                      tag_chunk_size=tag_chunk_size)
                    print(f"  Created: {os.path.basename(output_path)}")
                except Exception as e:
                    print(f"  Error processing {case_filename}: {e}")
                    if placeholder_success and os.path.exists(output_path):
                        print(f"Removing placeholder at {output_path}")
                        os.unlink(output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Process case reports with extracted tags using temporal prompts."
    )
    
    parser.add_argument(
        "--case-file",
        type=str,
        help="Case report file path (single file mode)"
    )
    parser.add_argument(
        "--tags-file",
        type=str,
        help="Extracted tags BSV file path (single file mode)"
    )
    parser.add_argument(
        "--context-file",
        type=str,
        help="Context file path (single file mode, optional)"
    )
    parser.add_argument(
        "--case-dir",
        type=str,
        help="Directory containing case reports (batch mode)"
    )
    parser.add_argument(
        "--tags-dir",
        type=str,
        help="Directory containing extracted tags BSV files (batch mode)"
    )
    parser.add_argument(
        "--context-dir",
        type=str,
        help="Directory containing context files (batch mode, optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="temporal_output",
        help="Output directory for temporal prompts"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="src/templates/temporal_prompt.txt",
        help="Path to temporal prompt template file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/DeepSeek-R1-0528-GGUF",
        help="Name of the model to use."
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
        default="http://127.0.0.1:8080/completion",
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
        "--wrapper_tokens",
        action='store_true',
        default=False,
        help="Wrap prompt with special tokens (e.g., <|system|>, <|user|>, <|assistant|>). Default False."
    )
    parser.add_argument(
        "--keep_columns",
        type=int,
        nargs='*',
        help="Specify which columns (0-based indices) to keep from the BSV file. Default keeps all columns."
    )
    parser.add_argument(
        "--tag_chunk_size",
        type=int,
        default=0,
        help="If >0, split mention-table rows into batches of this size and merge one temporal output. Helps long cases fit the context window."
    )
    parser.add_argument(
        "--extra_body",
        type=str,
        default='',
        help=""
    )
    
    args = parser.parse_args()
    extra_body = {} if args.extra_body == '' else json.loads(args.extra_body)

    # Load prompt template
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read().strip()
    
    # Validate arguments
    single_mode = bool(args.case_file and args.tags_file)
    batch_mode = bool(args.case_dir and args.tags_dir)
    
    if not single_mode and not batch_mode:
        parser.error("Must specify either (--case-file and --tags-file) OR (--case-dir and --tags-dir)")
    
    if single_mode and batch_mode:
        parser.error("Cannot specify both single file mode and batch mode")
    
    if single_mode:
        base_name = args.case_file.replace('.txt.gz', '').replace('.txt', '').replace('_modified','')
        output_file = os.path.join(args.output_dir, f"{base_name}.txt.gz")
        # os.path.join(args.output_dir, "temporal.txt.gz")
        os.makedirs(args.output_dir, exist_ok=True)
        process_single_pair(args.case_file, args.tags_file, args.context_file,
                          output_file, prompt_template, args.llm_mechanism, args.model_name,
                          args.cache_dir, args.lcpp_url, args.wrapper_tokens,
                          args.batch_size, args.endpoint_type, extra_body,
                          keep_columns=args.keep_columns,
                          tag_chunk_size=args.tag_chunk_size)
        print(f"Processed single triplet: {args.case_file} + {args.tags_file} + {args.context_file or 'None'}")
        
    elif batch_mode:
        if not os.path.exists(args.case_dir):
            print(f"Error: Case directory {args.case_dir} does not exist")
            return
        if not os.path.exists(args.tags_dir):
            print(f"Error: Tags directory {args.tags_dir} does not exist")
            return
            
        process_directories(args.case_dir, args.tags_dir, args.context_dir,
                          args.output_dir, prompt_template, args.llm_mechanism, args.model_name,
                          args.cache_dir, args.lcpp_url, args.wrapper_tokens,
                          args.batch_size, args.endpoint_type, extra_body,
                          keep_columns=args.keep_columns,
                          tag_chunk_size=args.tag_chunk_size)
        print(f"Processed all matching pairs in directories")

if __name__ == "__main__":
    main()
