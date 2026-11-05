# ATC-RAG

## Environment Setup

```bash
conda create --name ATC-RAG python=3.12
conda activate ATC-RAG
pip install -r requirements.txt
```

The PyTorch version depends on the CUDA version. Install a compatible PyTorch build when necessary.

## Folder Structure

```text
.
├── README.md
├── RQ1
│   ├── asserttype
│   ├── result
│   └── venn
├── RQ2
│   ├── no_f2
│   ├── no_atc
│   ├── diff_only
│   └── split_result.py
├── RQ3
│   ├── CodeBERT
│   ├── GraphCodeBERT
│   └── UniXcoder
├── atc_rag
├── third_party
├── dataset.tar.gz
├── args_config.py
├── evaluator
├── model.py
├── requirements.txt
├── run-old.sh
├── run-new.sh
└── run.py
```

## Models and Data

Place the four model directories under `models/`:

```text
models/
├── codet5-base
├── codebert-base
├── graphcodebert-base
└── unixcoder-base
```

The OldDataSet and NewDataSet files are stored in `dataset.tar.gz`. The scripts read this archive directly.

## Train and Test ATC-RAG

```bash
bash run-new.sh <gpu_ids> <passage_number> \
  --model models/codet5-base \
  --retriever-model models/codebert-base

bash run-old.sh <gpu_ids> <passage_number> \
  --model models/codet5-base \
  --retriever-model models/codebert-base
```

For example:

```bash
bash run-new.sh 0 3 \
  --model models/codet5-base \
  --retriever-model models/codebert-base
```

The Python entry can also be used directly:

```bash
python run.py --dataset old --stage all \
  --model models/codet5-base \
  --retriever-model models/codebert-base \
  --output-dir runs/full_old
```

## RQ1

### Metrics

```bash
python evaluator/acc.py <gold_file_path> <output_file_path>
python evaluator/bleu.py <gold_file_path> <output_file_path>
python evaluator/CodeBLEU/calc_code_bleu.py \
  --refs <gold_file_path> \
  --hyp <output_file_path> \
  --lang java
```

### Assert Type

```bash
cd RQ1/asserttype
bash calc_assert_type_new.sh
bash calc_assert_type_old.sh
```

### Venn Diagram

```bash
cd RQ1/venn
python to_excel.py --dataset new --baselines EditAS AG-RAG
python generate_venn.py --input baselines_new.csv --out venn_new.pdf
```

## RQ2

RQ2 contains `no_f2`, `no_atc`, and `diff_only`.

```bash
cd RQ2/no_f2
bash run-new.sh --model ../../models/codet5-base \
  --retriever-model ../../models/codebert-base
bash run-old.sh --model ../../models/codet5-base \
  --retriever-model ../../models/codebert-base
```

Replace `no_f2` with `no_atc` or `diff_only` to run the other ablations.

## RQ3

```bash
cd RQ3/CodeBERT
MODEL_NAME_OR_PATH=../../models/codebert-base bash run-new.sh <gpu_ids> <passage_number>
MODEL_NAME_OR_PATH=../../models/codebert-base bash run-old.sh <gpu_ids> <passage_number>
```

For the other backbones:

```bash
cd RQ3/GraphCodeBERT
MODEL_NAME_OR_PATH=../../models/graphcodebert-base bash run-new.sh <gpu_ids> <passage_number>

cd RQ3/UniXcoder
MODEL_NAME_OR_PATH=../../models/unixcoder-base bash run-new.sh <gpu_ids> <passage_number>
```
