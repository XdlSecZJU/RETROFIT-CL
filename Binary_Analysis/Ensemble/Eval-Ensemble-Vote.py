def eval_bleu_epoch(args, eval_data, eval_examples, _, tokenizer, split_tag, criteria):
    import copy
    import os
    import torch
    import numpy as np
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.utils.data import DataLoader, SequentialSampler

    logger.info("  ***** Running BLEU evaluation on {} data *****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
            num_workers=4, pin_memory=True
        )
    else:
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

    # --------------------------------------------------------
    # Reload three CodeT5 models
    # --------------------------------------------------------
    model_paths = [
        "/CodeT5-Decom",
        "/CodeT5-Demi",
        "/CodeT5-Stripped"
    ]
    models = []
    for path in model_paths:
        tmp_args = copy.deepcopy(args)
        tmp_args.model_name_or_path = path
        _, m, _ = build_or_load_gen_model(tmp_args)
        m.to(args.device)
        m.eval()
        models.append(m)
        logger.info(f"Loaded model from {path}")

    logger.info(f"Loaded {len(models)} models for ensemble (best-sequence by normalized log-likelihood)")

    # --------------------------------------------------------
    # Core: each model generates independently, scores, and selects highest-scoring sequence
    # --------------------------------------------------------
    def compute_seq_score(model, tokenizer, input_ids, attention_mask, generated_ids):
        """Compute normalized log-likelihood score"""
        labels = generated_ids[:, 1:].clone()
        decoder_input_ids = generated_ids[:, :-1].clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != tokenizer.pad_token_id)
        seq_log_prob = (token_log_probs * mask).sum(dim=-1)
        seq_len = mask.sum(dim=-1).clamp(min=1)
        return (seq_log_prob / seq_len).cpu().numpy()

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval BLEU on {split_tag}"):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        # Each model generates and scores
        model_preds, model_scores = [], []
        with torch.no_grad():
            for m in models:
                preds = m.generate(
                    source_ids,
                    attention_mask=source_mask,
                    use_cache=True,
                    num_beams=args.beam_size,
                    early_stopping=args.task == 'summarize',
                    max_length=args.max_target_length
                )
                model_preds.append(preds)
                model_scores.append(compute_seq_score(m, tokenizer, source_ids, source_mask, preds))

        # Select model with highest score for each sample
        model_scores = np.stack(model_scores, axis=0)  # [num_models, batch_size]
        best_model_idx = np.argmax(model_scores, axis=0)

        for i, idx in enumerate(best_model_idx):
            pred_ids.append(model_preds[idx][i])

    # --------------------------------------------------------
    # Post-processing & BLEU logic (identical to original)
    # --------------------------------------------------------
    pred_nls = [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for ids in pred_ids
    ]

    output_fn = os.path.join(args.res_dir, f"test_{criteria}.output")
    gold_fn = os.path.join(args.res_dir, f"test_{criteria}.gold")
    src_fn = os.path.join(args.res_dir, f"test_{criteria}.src")

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    f.write(pred_nl.strip() + '\n')
                    f1.write(gold.target.strip() + '\n')
                    f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)
            else:
                codebleu = 0.0

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result