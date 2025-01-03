import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
import jieba
from collections import Counter

# 中文分词函数
def chinese_tokenize(text):
    return list(jieba.cut(text))

# 计算ROUGE-1、ROUGE-2、ROUGE-L
def compute_rouge(reference, hypothesis):
    ref_tokens = chinese_tokenize(reference)
    hyp_tokens = chinese_tokenize(hypothesis)

    # 计算 ROUGE-1 (unigram)
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    common_tokens = ref_set & hyp_set
    precision_1 = len(common_tokens) / len(hyp_set) if len(hyp_set) > 0 else 0
    recall_1 = len(common_tokens) / len(ref_set) if len(ref_set) > 0 else 0
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    # 计算 ROUGE-2 (bigram)
    ref_bigrams = set(zip(ref_tokens, ref_tokens[1:]))
    hyp_bigrams = set(zip(hyp_tokens, hyp_tokens[1:]))
    common_bigrams = ref_bigrams & hyp_bigrams
    precision_2 = len(common_bigrams) / len(hyp_bigrams) if len(hyp_bigrams) > 0 else 0
    recall_2 = len(common_bigrams) / len(ref_bigrams) if len(ref_bigrams) > 0 else 0
    f1_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 else 0

    # 计算 ROUGE-L (Longest Common Subsequence)
    def longest_common_subsequence(a, b):
        m = len(a)
        n = len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    lcs_len = longest_common_subsequence(ref_tokens, hyp_tokens)
    precision_L = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall_L = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
    f1_L = (2 * precision_L * recall_L) / (precision_L + recall_L) if (precision_L + recall_L) > 0 else 0

    return f1_1, f1_2, f1_L

# 计算BLEU和SacreBLEU
def calculate_bleu_and_sacrebleu(references, hypotheses):
    bleu_scores = [sentence_bleu([ref.split()], hyp.split()) for ref, hyp in zip(references, hypotheses)]
    avg_bleu = np.mean(bleu_scores)
    
    sacre_bleu = sacrebleu.corpus_bleu(hypotheses, references)
    return avg_bleu, sacre_bleu.score

# 计算Precision和Recall
def calculate_precision_recall(reference, hypothesis):
    ref_tokens = chinese_tokenize(reference)
    hyp_tokens = chinese_tokenize(hypothesis)

    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)

    common_tokens = sum((hyp_counter & ref_counter).values())
    precision = common_tokens / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = common_tokens / len(ref_tokens) if len(ref_tokens) > 0 else 0

    return precision, recall

def evaluate_metrics(y_true, y_pred, metric):
    # 对真实标签和预测标签进行分词
    references = y_true
    hypotheses = y_pred

    # 计算 ROUGE 分数
    rouge1, rouge2, rougeL = [], [], []
    for ref, hyp in zip(references, hypotheses):
        f1_1, f1_2, f1_L = compute_rouge(ref, hyp)
        rouge1.append(f1_1)
        rouge2.append(f1_2)
        rougeL.append(f1_L)
    
    rouge1_score = np.mean(rouge1)
    rouge2_score = np.mean(rouge2)
    rougeL_score = np.mean(rougeL)

    # 计算BLEU 和 SacreBLEU
    bleu_score, sacre_bleu_score = calculate_bleu_and_sacrebleu(references, hypotheses)

    # 计算其他指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)

    # 计算Precision和Recall for generation task
    generation_precision, generation_recall = calculate_precision_recall(' '.join(y_true), ' '.join(y_pred))

    results = {
        "Accuracy": accuracy,
        "Precision (classification)": precision,
        "Recall (classification)": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "ROUGE-1": rouge1_score,
        "ROUGE-2": rouge2_score,
        "ROUGE-L": rougeL_score,
        "BLEU": bleu_score,
        "SacreBLEU": sacre_bleu_score,
        "Precision (generation)": generation_precision,
        "Recall (generation)": generation_recall
    }
    
    return results.get(metric, "Metric not found")

st.title("文本评估工具")

st.header("计算其他指标")
true_labels_input = st.text_area("输入真实标签 (专家认定符合要求的参考教学设计)", "这 是 一个 测试， 另一个 测试")
pred_labels_input = st.text_area("输入预测标签 (大语言模型输出的教学设计，切记在内容和形式上与专家认定标准保持一致)", "这 是 一个 测试， 另一个 示例")

metric = st.selectbox("选择评估指标", ["Accuracy", "Precision (classification)", "Recall (classification)", "F1 Score", "MCC", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "SacreBLEU", "Precision (generation)", "Recall (generation)"])

if st.button("评估"):
    y_true = true_labels_input.split(", ")
    y_pred = pred_labels_input.split(", ")
    
    if len(y_true) != len(y_pred):
        st.error("真实标签和预测标签数量不匹配")
    else:
        result = evaluate_metrics(y_true, y_pred, metric)
        
        st.subheader("评估结果")
        st.write(f"{metric}: {result:.4f}" if isinstance(result, (int, float)) else result)

st.header("计算困惑度")
st.markdown("""
**困惑度** 是衡量语言模型预测文本的平均分支因子。它反映了模型对文本预测的不确定性。较低的困惑度表示较低的不确定性和较好的模型性能。
""")
perplexity_input = st.text_area("输入文本计算困惑度", "this is a sample text for perplexity calculation")

if st.button("计算困惑度"):
    perplexity_result = calculate_perplexity(perplexity_input)
    
    st.subheader("困惑度结果")
    st.write(f"Perplexity: {perplexity_result:.2f}%")
