"""ATC-RAG data, retrieval, frozen change cues, and generation.

Uses the bundled frozen F1/F2/ATC implementation unchanged.
"""
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from atc_rag.features import retrieval_text
from atc_rag.serialization import prepare_example, serialize

BACKBONES = {"CodeBERT": "microsoft/codebert-base", "GraphCodeBERT": "microsoft/graphcodebert-base",
             "UniXcoder": "microsoft/unixcoder-base"}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_device(name):
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_rows(args, split, limit=None):
    from args_config import NAMES
    path = args.data
    if path.is_dir():
        path = path / NAMES[args.dataset] / f"assert_{split}_{args.dataset}.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(itertools.islice(csv.DictReader(stream), limit))
    else:
        # The original archive is read directly: no extraction paths or duplicate dataset.
        import io
        import tarfile
        with tarfile.open(path, "r:gz") as archive:
            suffix = f"{NAMES[args.dataset]}/assert_{split}_{args.dataset}.csv"
            member = next(item for item in archive.getmembers() if item.name.endswith(suffix))
            with io.TextIOWrapper(archive.extractfile(member), encoding="utf-8-sig", newline="") as stream:
                records = list(itertools.islice(csv.DictReader(stream), limit))
    return [{"id": f"{args.dataset}:{split}:{i}", "source": row["source"], "target": row["target"]}
            for i, row in enumerate(records)]


class DenseEncoder:
    def __init__(self, name, device, length):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device
        self.length = length

    def encode(self, texts, batch_size):
        vectors = []
        for offset in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[offset:offset + batch_size], padding=True, truncation=True,
                                   max_length=self.length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                vector = (hidden * mask).sum(1) / mask.sum(1)
                vectors.append(torch.nn.functional.normalize(vector, dim=1).cpu().numpy())
        return np.concatenate(vectors)


def build_index(args):
    import joblib
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows = dataset_rows(args, "train", args.index_limit)
    use_evidence = args.experiment != "no_f2"
    texts = [retrieval_text(row["source"], include_evidence=use_evidence) for row in rows]
    path = args.output_dir / "index"
    path.mkdir(parents=True, exist_ok=True)
    mode = "lexical" if args.experiment == "IR_retriever" else args.retrieval_mode
    encoder_name = args.retriever_model
    if args.experiment == "finetuned_retriever":
        if not args.retriever_checkpoint:
            raise ValueError("finetuned_retriever needs --retriever-checkpoint (HF save_pretrained directory)")
        encoder_name = args.retriever_checkpoint
    if mode == "lexical":
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2),
                                     max_features=80000, sublinear_tf=True, dtype=np.float32)
        sparse.save_npz(path / "tfidf.npz", vectorizer.fit_transform(texts))
        joblib.dump(vectorizer, path / "vectorizer.joblib")
    else:
        encoder = DenseEncoder(encoder_name, get_device(args.device), args.query_length)
        np.save(path / "embeddings.npy", encoder.encode(texts, args.retrieval_batch_size))
    write_jsonl(path / "train.jsonl", rows)
    write_json(path / "config.json", {"mode": mode, "encoder": encoder_name, "dataset": args.dataset,
                                     "query_length": args.query_length, "train_rows": len(rows),
                                     "use_evidence": use_evidence})
    print(f"Index: {len(rows)} train rows ({mode})", flush=True)


class Retriever:
    def __init__(self, args):
        import joblib
        from scipy import sparse
        path = args.output_dir / "index"
        self.rows = read_jsonl(path / "train.jsonl")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config["dataset"] != args.dataset:
            raise ValueError("Index dataset differs from query dataset")
        self.use_evidence = config.get("use_evidence", True)
        if self.use_evidence != (args.experiment != "no_f2"):
            raise ValueError("no_f2 requires a separately built index without F2 evidence")
        self.encoder = None
        if config["mode"] == "lexical":
            self.vectorizer = joblib.load(path / "vectorizer.joblib")
            self.matrix = sparse.load_npz(path / "tfidf.npz")
        else:
            self.matrix = np.load(path / "embeddings.npy", mmap_mode="r")
            self.encoder = DenseEncoder(config["encoder"], get_device(args.device), config["query_length"])

    def search(self, rows, k):
        texts = [retrieval_text(row["source"], include_evidence=self.use_evidence) for row in rows]
        if self.encoder:
            scores = self.encoder.encode(texts, len(texts)) @ self.matrix.T
        else:
            scores = (self.vectorizer.transform(texts) @ self.matrix.T).toarray()
        found = []
        for row, values in zip(rows, scores):
            # Exclude before top-k: identical sources must never enter retrieved context.
            for index, historical in enumerate(self.rows):
                if historical["id"] == row["id"] or historical["source"] == row["source"]:
                    values[index] = -np.inf
            count = min(k, int(np.isfinite(values).sum()))
            indices = np.argpartition(-values, count - 1)[:count] if count else []
            indices = sorted(indices, key=lambda index: (-float(values[index]), int(index)))
            found.append([{"ref_id": self.rows[i]["id"], "source": self.rows[i]["source"],
                           "assertion": self.rows[i]["target"], "score": float(values[i])} for i in indices])
        return found


def prepare_data(args):
    retriever = None if args.experiment == "no_retriever" else Retriever(args)
    variant = args.experiment if args.experiment in ["no_f2", "no_atc", "diff_only"] else "full"
    from atc_rag.common import assertion_type
    for split in ["train", "val", "test"]:
        rows = dataset_rows(args, split, getattr(args, f"{split}_limit"))
        path = args.output_dir / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for offset in range(0, len(rows), args.retrieval_batch_size):
                batch = rows[offset:offset + args.retrieval_batch_size]
                # Query and feature construction receive source and ID only.
                queries = [{"id": row["id"], "source": row["source"]} for row in batch]
                references = retriever.search(queries, args.k) if retriever else [[] for _ in batch]
                for row, query, refs in zip(batch, queries, references):
                    hidden = {**query, "target": "", "assertion_type": ""}
                    example = prepare_example(hidden, refs, k=args.k,
                        variant="code_only" if retriever is None else variant,
                        atc_prefix=args.atc_prefix, atc_focal=args.atc_focal)
                    example["target"] = row["target"]
                    example["assertion_type"] = assertion_type(row["target"])
                    stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Prepared {split}: {len(rows)}", flush=True)


def load_generator(args, checkpoint=None):
    if checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return tokenizer, AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return AutoTokenizer.from_pretrained(args.model), AutoModelForSeq2SeqLM.from_pretrained(args.model)


def make_loader(path, tokenizer, args, shuffle):
    rows = read_jsonl(path)
    def collate(batch):
        inputs = tokenizer([serialize(row, tokenizer, args.max_source_length) for row in batch],
                           padding=True, truncation=True, max_length=args.max_source_length, return_tensors="pt")
        labels = tokenizer(text_target=[row["target"] for row in batch], padding=True, truncation=True,
                           max_length=args.max_target_length, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return {**inputs, "labels": labels}
    return DataLoader(rows, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)


def train_generator(args):
    set_seed(args.seed)
    tokenizer, generator = load_generator(args)
    device = get_device(args.device)
    generator.to(device)
    if args.gradient_checkpointing:
        generator.gradient_checkpointing_enable()
        generator.config.use_cache = False
    train = make_loader(args.output_dir / "train.jsonl", tokenizer, args, True)
    validation = make_loader(args.output_dir / "val.jsonl", tokenizer, args, False)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.learning_rate)
    steps_per_epoch = (len(train) + args.grad_accum - 1) // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps_per_epoch * args.epochs * .05),
                                                steps_per_epoch * args.epochs)
    amp = args.precision != "fp32"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device, enabled=amp and args.precision == "fp16")
    history = []
    best_loss = float("inf")
    updates = 0
    write_json(args.output_dir / "run_config.json",
               {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    for epoch in range(args.epochs):
        generator.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.
        for step, batch in enumerate(train):
            batch = {key: value.to(device) for key, value in batch.items()}
            window_start = step // args.grad_accum * args.grad_accum
            window_size = min(args.grad_accum, len(train) - window_start)
            with torch.autocast(device_type=device, dtype=dtype, enabled=amp):
                loss = generator(**batch).loss
            train_loss += loss.item()
            scaler.scale(loss / window_size).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    updates += 1
                optimizer.zero_grad(set_to_none=True)
        generator.eval()
        val_loss = 0.
        with torch.inference_mode():
            for batch in validation:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device, dtype=dtype, enabled=amp):
                    val_loss += generator(**batch).loss.item()
        val_loss /= len(validation)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train),
                        "val_loss": val_loss, "optimizer_updates": updates})
        print(json.dumps(history[-1]), flush=True)
        write_json(args.output_dir / "history.json", history)
        if val_loss < best_loss:
            best_loss = val_loss
            generator.save_pretrained(args.output_dir / "saved_models")
            tokenizer.save_pretrained(args.output_dir / "saved_models")


def predict(args):
    tokenizer, generator = load_generator(args, args.output_dir / "saved_models")
    device = get_device(args.device)
    generator.to(device).eval()
    rows = read_jsonl(args.output_dir / "test.jsonl")
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as stream, \
         (args.output_dir / "test.gold").open("w", encoding="utf-8") as gold, \
         (args.output_dir / "test.output").open("w", encoding="utf-8") as output:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            inputs = tokenizer([serialize(row, tokenizer, args.max_source_length) for row in batch],
                               padding=True, truncation=True, max_length=args.max_source_length,
                               return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = generator.generate(**inputs, max_new_tokens=args.max_target_length,
                                               num_beams=args.beams, num_return_sequences=1)
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, prediction in zip(batch, predictions):
                record = {"id": row["id"], "reference": row["target"], "prediction": prediction,
                          "assertion_type": row["assertion_type"]}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                # Legacy evaluators use one ID + tab + assertion per line.
                reference_line = row["target"].replace("\t", " ").replace("\r", " ").replace("\n", " ")
                prediction_line = prediction.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                gold.write(f'{row["id"]}\t{reference_line}\n')
                output.write(f'{row["id"]}\t{prediction_line}\n')
    print(f"Predicted {len(rows)} single-output assertions", flush=True)

