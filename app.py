import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sacrebleu

# 准备GPT-2模型和tokenizer用于计算Perplexity
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def evaluate_metrics(y_true, y_pred):
    references = [[ref] for ref in y_true]
    hypotheses = y_pred

    # 准备ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 计算Perplexity
    perplexity_scores = [calculate_perplexity(text) for text in y_pred]

    # 计算ROUGE
    rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(y_true, y_pred)]
    rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    # 计算BLEU
    bleu_scores = [sentence_bleu([ref], pred) for ref, pred in zip(references, hypotheses)]
    avg_bleu = np.mean(bleu_scores)

    # 计算SacreBLEU
    sacre_bleu = sacrebleu.corpus_bleu(hypotheses, references)

    # 计算其他指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)

    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "BLEU": avg_bleu,
        "SacreBLEU": sacre_bleu.score,
        "Perplexity": np.mean(perplexity_scores)
    }
    
    return results

st.title("文本评估工具")

true_labels_input = st.text_area("输入真实标签 (每个标签用逗号和空格分隔)", "this is a test, another test")
pred_labels_input = st.text_area("输入预测标签 (每个标签用逗号和空格分隔)", "this is a test, another example")

if st.button("评估"):
    y_true = true_labels_input.split(", ")
    y_pred = pred_labels_input.split(", ")
    
    if len(y_true) != len(y_pred):
        st.error("真实标签和预测标签数量不匹配")
    else:
        results = evaluate_metrics(y_true, y_pred)
        
        st.subheader("评估结果")
        for metric, score in results.items():
            st.write(f"{metric}: {score:.4f}")


