import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import sacrebleu

def evaluate_metrics(y_true, y_pred):
    references = [[ref] for ref in y_true]
    hypotheses = y_pred

    # 准备ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

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
        "SacreBLEU": sacre_bleu.score
    }
    
    return results

st.title("文本评估工具")

true_labels_input = st.text_area("输入真实标签 (专家认定符合要求的参考教学设计)", "this is a test, another test")
pred_labels_input = st.text_area("输入预测标签 (大语言模型输出的教学设计，切记在内容和形式上与专家认定标准保持一致)", "this is a test, another example")

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
