"""Export ATC-RAG JSONL predictions to AG-RAG ID<TAB>text gold/output files."""
import json
import sys

def format_data(file_path, gold_path, output_path):
    with open(file_path, encoding="utf-8") as source, open(gold_path, "w", encoding="utf-8") as gold, \
         open(output_path, "w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            reference = row["reference"].replace("\t", " ").replace("\n", " ").replace("\r", " ")
            prediction = row["prediction"].replace("\t", " ").replace("\n", " ").replace("\r", " ")
            gold.write(f'{row["id"]}\t{reference}\n')
            output.write(f'{row["id"]}\t{prediction}\n')

if __name__ == "__main__":
    format_data(*sys.argv[1:])

