import os
import re
import argparse
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

# ---------------------------- PDF Processing Functions ----------------------------

def get_all_pdfs(root_dir):
    """Retrieve all PDF files in the specified directory and its subdirectories."""
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files if file.lower().endswith('.pdf')
    ]

def parse_pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def clean_text(text):
    """Clean extracted text by removing unnecessary newlines and extra spaces."""
    text = re.sub(r"(?<![\.\!\?])\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def save_text_to_file(text, output_dir, index):
    """Save cleaned text to a file with a specified index."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{index:03}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {file_path}")

def process_pretraining_data(root_dir, output_dir):
    """Process all PDFs in the root directory and save the cleaned text to output directory."""
    pdf_files = get_all_pdfs(root_dir)
    print(f"Found {len(pdf_files)} PDF files.")

    for index, pdf_path in enumerate(pdf_files):
        print(f"Processing ({index + 1}/{len(pdf_files)}): {pdf_path}")
        raw_text = parse_pdf_to_text(pdf_path)
        processed_text = clean_text(raw_text)
        save_text_to_file(processed_text, output_dir, index)

# ---------------------------- Finetuning Data Functions ----------------------------

def process_finetuning_data(csv_path, output_path):
    """Process finetuning data from a CSV file and save it as a Huggingface Dataset."""
    # Load CSV
    data = pd.read_csv(csv_path).dropna(subset=["답변"])
    data = data.drop(columns=['주제 대분류', 'No. '], errors='ignore')
    data = data.rename(columns={"질문": "question", "답변": "answer"})

    # Split into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Convert to Huggingface Dataset
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_data.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_data.reset_index(drop=True))
    })

    # Transform to message format
    def transform_example(example):
        return {"messages": [{"role": "user", "content": example["question"]},
                             {"role": "assistant", "content": example["answer"]}]}

    dataset_dict = dataset_dict.map(transform_example)
    dataset_dict.save_to_disk(output_path)

    print(f"Huggingface dataset saved to {output_path}")

# ---------------------------- CPT Data Preprocessing ----------------------------

def preprocess_cpt_data(indexed_paths, tokenizer_path, output_path):
    """Preprocess indexed text data into tokenized chunks and save as Huggingface Dataset."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    max_length = 2048
    eos_id = tokenizer.eos_token_id

    if eos_id is None:
        raise ValueError("The tokenizer does not have an eos_token_id.")

    token_buffer, samples = [], []

    def load_text_data(paths):
        for path in paths:
            for root, _, files in os.walk(path):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            tokens = tokenizer(line, add_special_tokens=False).input_ids
                            token_buffer.extend(tokens)
                            token_buffer.append(eos_id)

                            while len(token_buffer) >= max_length:
                                samples.append(token_buffer[:max_length])
                                token_buffer[:] = token_buffer[max_length:]

    load_text_data(indexed_paths)

    dataset = Dataset.from_dict({"input_ids": samples})
    dataset.save_to_disk(output_path)
    print(f"Preprocessing complete. Saved dataset to {output_path}")

# ---------------------------- Main Execution ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for pretraining, finetuning, or CPT.")
    subparsers = parser.add_subparsers(dest="command", help="Choose a processing task: pretraining, finetuning, or cpt")

    # Pretraining subcommand
    pretrain_parser = subparsers.add_parser("pretraining", help="Process pretraining data from PDFs")
    pretrain_parser.add_argument("--root_dir", required=True, help="Root directory containing PDF files")
    pretrain_parser.add_argument("--output_dir", required=True, help="Output directory for processed text files")

    # Finetuning subcommand
    finetune_parser = subparsers.add_parser("finetuning", help="Process finetuning data from a CSV file")
    finetune_parser.add_argument("--csv_path", required=True, help="Path to the finetuning CSV file")
    finetune_parser.add_argument("--output_path", required=True, help="Output directory for Huggingface Dataset")

    # CPT subcommand
    cpt_parser = subparsers.add_parser("cpt", help="Preprocess CPT data for tokenization")
    cpt_parser.add_argument("--indexed_paths", nargs="+", required=True, help="Paths to indexed text files")
    cpt_parser.add_argument("--tokenizer_path", required=True, help="Path or name of your tokenizer")
    cpt_parser.add_argument("--output_path", required=True, help="Path to save the processed dataset")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "pretraining":
        process_pretraining_data(args.root_dir, args.output_dir)
    elif args.command == "finetuning":
        process_finetuning_data(args.csv_path, args.output_path)
    elif args.command == "cpt":
        preprocess_cpt_data(args.indexed_paths, args.tokenizer_path, args.output_path)
    else:
        parser.print_help()
