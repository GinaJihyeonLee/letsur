# 데이터 전처리

## 환경 설정 

```bash
conda create -n preprocess python=3.9
conda activate preprocess
pip install PyPDF2 pandas scikit-learn datasets transformers
```

### 1. 사전 학습 데이터 처리

지정된 디렉터리 내의 모든 PDF 파일을 처리하고 텍스트를 추출하여 별도의 `.txt` 파일로 저장합니다.

**명령어:**

```bash
python process_data.py pretraining --root_dir <PDF_파일_디렉터리> --output_dir <결과_저장_디렉터리>
```

### 2. 파인튜닝 데이터 처리

질문-답변 쌍이 포함된 CSV 파일을 처리하여 Huggingface `Dataset` 형식으로 저장합니다.

**명령어:**

```bash
python process_data.py finetuning --csv_path <CSV_파일_경로> --output_path <결과_저장_디렉터리>
```

### 3. CPT 데이터 전처리

인덱싱된 텍스트 데이터를 토큰화된 고정 길이 청크로 변환하고 Huggingface `Dataset` 형식으로 저장합니다.

**명령어:**

```bash
python process_data.py cpt --indexed_paths <인덱싱된_텍스트_경로> --tokenizer_path <토크나이저_경로> --output_path <결과_저장_디렉터리>
```

# 모델 학습

## 환경 설정 

```bash
conda create -n train python=3.9
conda activate train
pip install -r train_requirements.txt
```

### 1. CPT
`trl/examples/scripts` 디렉터리 하위에 `letsur_cpt.py` 스크립트를 생성하였습니다.

```bash
cd trl/examples
accelerate launch \
  --config_file=accelerate_configs/multi_gpu.yaml \
  --num_processes 8 \
  scripts/letsur_cpt.py \
  --model_name_or_path <모델_경로>> \
  --dataset_name <데이터셋_경로> \
  --output_dir <결과_저장_디렉터리> \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 25 \
  --eval_strategy no \
  --trust_remote_code \
  --gradient_checkpointing \
  --use_peft \
  --ddp_find_unused_parameters=False
```

### 2. Finetuning
`trl/examples/scripts` 디렉터리 하위에 `letsur_sft.py` 스크립트를 생성하였습니다.

```bash
cd trl/examples
accelerate launch \
  --config_file=accelerate_configs/multi_gpu.yaml \
  --num_processes 8 \
  scripts/letsur_sft.py \
  --model_name_or_path <모델_경로>> \
  --dataset_name <데이터셋_경로> \
  --output_dir <결과_저장_디렉터리> \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 25 \
  --eval_strategy no \
  --trust_remote_code \
  --gradient_checkpointing \
  --peft_path <cpt_peft_weight_경로> \
  --use_peft \
  --ddp_find_unused_parameters=False
```

# 평가

## 환경 설정 

```bash
conda create -n lm_eval python=3.9
conda activate lm_eval
pip install -r eval_requirements.txt
```

### 1. LM-Eval-Harness

`lm-eval-harness`를 사용하여 한국어 평가 데이터셋인 **KMMLU_direct**, **KOBEST**, **KoBBQ**, **KoSBi**를 평가합니다.

```bash
lm_eval \
  --model hf \
  --model_args pretrained=<모델_경로>,trust_remote_code=True \
  --tasks kmmlu_direct,kobest,kobbq,kosbi \
  --device cuda:0 \
  --batch_size 16 \
  --output_path <결과_경로>
```

### 2. llm-as-a-judge

```bash
python llm_as_a_judge.py \
  --model_name <모델_경로> \
  --trust_remote_code \
  --dataset_path qa_data_hf \
  --output_path <결과_경로> \
  --lora_path <sft_peft_weigth_경로>
```
