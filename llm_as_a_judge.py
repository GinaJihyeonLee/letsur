import os
import openai
import argparse
from datasets import load_from_disk
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def evaluate_model(model_name: str, trust_remote_code: bool, dataset_path: str, openai_model: str = "gpt-4", lora_path: str = None):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    dataset = load_from_disk(dataset_path)
    dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=trust_remote_code)

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    evaluation_prompt = """
You are a strict and fair judge. I will provide a question, a gold (reference) answer, and a candidate answer. Your task is to compare the candidate answer to the gold answer and give a single integer score between 0 and 10 (0 means completely wrong, 10 means perfectly correct). Only provide the number as the final output. Do not provide any explanation.

Question:
{question}

Gold answer:
{gold_answer}

Candidate answer:
{candidate_answer}

Score:
"""
    scores = []
    for example in dataset:
        question = example["question"]
        gold_answer = example["answer"]
        messages = [
            {"role": "user", "content": question}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        output = model.generate(
            input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256
        )
        if "EXAONE" in model_name:
            candidate_answer = tokenizer.decode(output[0]).split("[|assistant|]",1)[1]
        elif "gemma" in model_name:
            candidate_answer = tokenizer.decode(output[0]).split("<start_of_turn>model\n",1)[1]
        messages = [
            {"role": "system", "content": "You are a helpful assistant who evaluates answers."},
            {"role": "user", "content": evaluation_prompt.format(
                question=question,
                gold_answer=gold_answer,
                candidate_answer=candidate_answer,
            )}
        ]

        response = openai.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=0.0
        )

        score_str = response.choices[0].message.content.strip()
        try:
            score = int(score_str)
        except ValueError:
            score = None

        scores.append((question, gold_answer, candidate_answer, score))
        print("Question:", question)
        print("Gold Answer:", gold_answer)
        print("Candidate Answer:", candidate_answer)
        print("Score:", score)
        print("=====")

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model's generated answers against a gold standard using ChatGPT scoring.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--trust_remote_code", action='store_true', help="If set, allows loading remote code for custom models")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier of the HuggingFace dataset")
    parser.add_argument("--openai_model", type=str, default="gpt-4", help="OpenAI model for evaluation (e.g., gpt-4 or gpt-3.5-turbo)")
    parser.add_argument("--output_path", type=str, required=True, help="Path of the output dir")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights if using a LoRA fine-tuned model")

    args = parser.parse_args()

    scores = evaluate_model(
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
        dataset_path=args.dataset_path,
        openai_model=args.openai_model,
        lora_path=args.lora_path
    )

    avg_score = sum(s[-1] for s in scores) / len(scores)
    scores_list = [list(s) for s in scores]
    results = {
        "scores": scores_list,
        "avg_score": avg_score
    }

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    output_file = os.path.join(args.output_path, "results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)