import os
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score
from tqdm import tqdm
import argparse

# ------------- NLTK resources -------------
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ------------- utility -------------
def clean_line(line):
    parts = line.strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line.strip()

# ------------- METEOR -------------
def compute_meteor(base_dir, out_file='meteor_results.txt'):
    results = []
    task_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pbar = tqdm(total=len(task_folders), desc='METEOR folder progress')
    for task_folder in sorted(task_folders):
        task_path = os.path.join(base_dir, task_folder)
        for sub in ('decomC', 'demi', 'stripped'):
            sub_path = os.path.join(task_path, sub)
            gold_file = os.path.join(sub_path, 'prediction', 'test_best-bleu.gold')
            out_file_path = os.path.join(sub_path, 'prediction', 'test_best-bleu.output')
            if not (os.path.exists(gold_file) and os.path.exists(out_file_path)):
                continue
            with open(gold_file, encoding='utf-8') as f:
                gold_lines = [clean_line(l) for l in f if l.strip()]
            with open(out_file_path, encoding='utf-8') as f:
                out_lines = [clean_line(l) for l in f if l.strip()]
            n = min(len(gold_lines), len(out_lines))
            gold_lines, out_lines = gold_lines[:n], out_lines[:n]
            scores = [meteor_score([word_tokenize(g)], word_tokenize(p)) for g, p in zip(gold_lines, out_lines)]
            avg = sum(scores) / len(scores) if scores else 0.0
            results.append((f'{task_folder}-{sub}', avg))
        pbar.update(1)
    pbar.close()
    with open(out_file, 'w', encoding='utf-8') as fw:
        header = f'{"Task-SubTask":<40} {"METEOR":>10}\n' + '-' * 52 + '\n'
        print(header, end=''); fw.write(header)
        for key, meteor in results:
            line = f'{key:<40} {meteor:>10.4f}\n'
            print(line, end=''); fw.write(line)
    print(f'Results written to {os.path.abspath(out_file)}')

# ------------- ROUGE-L -------------
def compute_rouge(base_dir, out_file='rouge_results.txt'):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    task_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pbar = tqdm(total=len(task_folders), desc='ROUGE folder progress')
    for task_folder in sorted(task_folders):
        task_path = os.path.join(base_dir, task_folder)
        for sub in ('decomC', 'demi', 'stripped'):
            sub_path = os.path.join(task_path, sub)
            gold_file = os.path.join(sub_path, 'prediction', 'test_best-bleu.gold')
            out_file_path = os.path.join(sub_path, 'prediction', 'test_best-bleu.output')
            if not (os.path.exists(gold_file) and os.path.exists(out_file_path)):
                continue
            with open(gold_file, encoding='utf-8') as f:
                gold_lines = [clean_line(l) for l in f if l.strip()]
            with open(out_file_path, encoding='utf-8') as f:
                out_lines = [clean_line(l) for l in f if l.strip()]
            n = min(len(gold_lines), len(out_lines))
            gold_lines, out_lines = gold_lines[:n], out_lines[:n]
            scores = [scorer.score(g, p)['rougeL'] for g, p in zip(gold_lines, out_lines)]
            avg_r = sum(s.recall for s in scores) / n
            avg_f = sum(s.fmeasure for s in scores) / n
            results.append((f'{task_folder}-{sub}', avg_r, avg_f))
        pbar.update(1)
    pbar.close()
    with open(out_file, 'w', encoding='utf-8') as fw:
        header = f'{"Task-SubTask":<40} {"ROUGE-L-R":>10} {"ROUGE-L-F":>10}\n' + '-' * 62 + '\n'
        print(header, end=''); fw.write(header)
        for key, r, f in results:
            line = f'{key:<40} {r:>10.4f} {f:>10.4f}\n'
            print(line, end=''); fw.write(line)
    print(f'Results written to {os.path.abspath(out_file)}')

# ------------- BERTScore -------------
def compute_bertscore(base_dir, out_file='bertscore_results.txt'):
    # offline roberta-large path
    local_model_path = 'roberta-large'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = local_model_path
    os.environ['HF_DATASETS_OFFLINE'] = '1'

    results = []
    task_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pbar = tqdm(total=len(task_folders), desc='BERTScore folder progress')
    for task_folder in sorted(task_folders):
        task_path = os.path.join(base_dir, task_folder)
        for sub in ('decomC', 'demi', 'stripped'):
            sub_path = os.path.join(task_path, sub)
            gold_file = os.path.join(sub_path, 'prediction', 'test_best-bleu.gold')
            out_file_path = os.path.join(sub_path, 'prediction', 'test_best-bleu.output')
            if not (os.path.exists(gold_file) and os.path.exists(out_file_path)):
                continue
            with open(gold_file, encoding='utf-8') as f:
                gold = [clean_line(l) for l in f if l.strip()]
            with open(out_file_path, encoding='utf-8') as f:
                pred = [clean_line(l) for l in f if l.strip()]
            n = min(len(gold), len(pred))
            gold, pred = gold[:n], pred[:n]
            P, R, F1 = score(pred, gold,
                             model_type='roberta-large',
                             lang='en',
                             verbose=False,
                             rescale_with_baseline=False)
            results.append((f'{task_folder}-{sub}',
                            P.mean().item(),
                            R.mean().item(),
                            F1.mean().item()))
        pbar.update(1)
    pbar.close()
    with open(out_file, 'w', encoding='utf-8') as fw:
        header = f'{"Task-SubTask":<40} {"Precision":>10} {"Recall":>10} {"F1":>10}\n' + '-' * 72 + '\n'
        print(header, end=''); fw.write(header)
        for key, p, r, f in results:
            line = f'{key:<40} {p:>10.4f} {r:>10.4f} {f:>10.4f}\n'
            print(line, end=''); fw.write(line)
    print(f'Results written to {os.path.abspath(out_file)}')

# ------------- main -------------
def main(base_dir):
    compute_meteor(base_dir)
    compute_rouge(base_dir)
    compute_bertscore(base_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics for generated text.')
    parser.add_argument('--base_dir', type=str, default='output', help='Base directory for output files')
    args = parser.parse_args()
    main(args.base_dir)