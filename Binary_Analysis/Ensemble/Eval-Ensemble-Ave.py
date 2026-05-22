def eval_bleu_epoch(args, eval_data, eval_examples, _, tokenizer, split_tag, criteria):
    import copy
    from torch.utils.data import DataLoader, SequentialSampler
    from tqdm import tqdm
    import numpy as np
    import torch
    import os

    logger.info("  ***** Running BLEU evaluation on {} data *****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True
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
    logger.info(f"Loaded {len(models)} models for ensemble (logit averaging)")

    # --------------------------------------------------------
    # Start evaluation (logit averaging)
    # --------------------------------------------------------
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc=f"Eval BLEU on {split_tag}"):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            # Record per-step logits from each model during generation
            all_scores = []
            for m in models:
                out = m.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    max_length=args.max_target_length,
                    num_beams=1,  # disable beam search to avoid inconsistency
                    output_scores=True,
                    return_dict_in_generate=True
                )
                all_scores.append(out.scores)  # list of [B, V] per step

            # Align generation lengths
            min_len = min(len(s) for s in all_scores)
            truncated_scores = [s[:min_len] for s in all_scores]

            # Average logits step by step
            avg_scores = []
            for step_logits in zip(*truncated_scores):  # pick M [B, V] per step
                stacked = torch.stack(step_logits, dim=0)  # [M, B, V]
                avg_logits = torch.mean(stacked, dim=0)    # [B, V]
                avg_scores.append(avg_logits)

            # Take argmax from averaged logits to obtain predicted sequences
            preds = torch.stack([torch.argmax(step, dim=-1) for step in avg_scores], dim=1)  # [B, T]
            pred_ids.extend(preds.cpu().numpy())

    # --------------------------------------------------------
    # Post-processing & BLEU computation
    # --------------------------------------------------------
    pred_nls = [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for ids in pred_ids
    ]

    output_fn = os.path.join(args.res_dir, f"test_{criteria}.output")
    gold_fn = os.path.join(args.res_dir, f"test_{criteria}.gold")
    src_fn = os.path.join(args.res_dir, f"test_{criteria}.src")

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

    # Compute BLEU / CodeBLEU
    if args.task == 'summarize':
        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        codebleu = 0.0
    else:
        bleu = round(_bleu(gold_fn, output_fn), 2)
        if args.task in ['concode', 'translate', 'refine']:
            codebleu = calc_code_bleu.get_code_bleu(gold_fn, output_fn, args.lang)
        else:
            codebleu = 0.0

    result = {
        'em': np.mean(dev_accs) * 100,
        'bleu': bleu,
        'codebleu': codebleu * 100 if args.task == 'concode' else 0
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result