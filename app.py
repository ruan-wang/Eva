'''
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sacrebleu

# 示例输入
y_true = ["this is a test", "another test"]
y_pred = ["this is a test", "another example"]
references = [["this is a test"], ["another test"]]
hypotheses = ["this is a test", "another example"]

# 准备GPT-2模型和tokenizer用于计算Perplexity
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

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
sacre_bleu = sacrebleu.corpus_bleu(hypotheses, [[ref[0]] for ref in references])

# 计算其他指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
mcc = matthews_corrcoef(y_true, y_pred)

# 打印结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"ROUGE-1: {rouge1:.4f}")
print(f"ROUGE-2: {rouge2:.4f}")
print(f"ROUGE-L: {rougeL:.4f}")
print(f"BLEU: {avg_bleu:.4f}")
print(f"SacreBLEU: {sacre_bleu.score:.4f}")
print(f"Perplexity: {np.mean(perplexity_scores):.4f}")

'''
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

uploaded_file = st.file_uploader("上传一个包含真实标签和预测标签的文本文件", type="txt")

if uploaded_file is not None:
    # 读取文件内容
    file_content = uploaded_file.read().decode("utf-8")
    lines = file_content.strip().split("\n")
    
    if len(lines) < 2:
        st.error("文件内容格式错误，应包含至少两行：第一行是真实标签，第二行是预测标签")
    else:
        y_true = lines[0].split(", ")
        y_pred = lines[1].split(", ")
        
        if len(y_true) != len(y_pred):
            st.error("真实标签和预测标签数量不匹配")
        else:
            results = evaluate_metrics(y_true, y_pred)
            
            st.subheader("评估结果")
            for metric, score in results.items():
                st.write(f"{metric}: {score:.4f}")








# python  /nfs/home/1002_sunbo/RW_Experiments/02_红楼梦教学设计生成/Eva/app.py
