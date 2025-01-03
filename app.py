import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
import jieba  # 中文分词工具
from collections import Counter

# 计算困惑度并归一化
def calculate_perplexity(text):
    words = text.split()
    total_log_prob = 0
    total_length = len(words)

    # 假设一个非常简单的基于词频的概率模型
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

    for word in words:
        prob = word_freq[word] / total_length
        total_log_prob += np.log(prob)

    avg_log_prob = total_log_prob / total_length
    perplexity = np.exp(-avg_log_prob)

    # 归一化处理，将困惑度值映射到0到100之间
    normalized_perplexity = 100 / (1 + perplexity)
    return normalized_perplexity

# 中文分词函数
def chinese_tokenize(text):
    return list(jieba.cut(text))

# 去除停用词函数（这里可以自定义停用词列表）
def remove_stopwords(tokens, stopwords=None):
    if stopwords is None:
        stopwords = set()  # 如果没有指定停用词表，则默认不去除停用词
    return [word for word in tokens if word not in stopwords]

# 计算Precision和Recall
def calculate_precision_recall(reference, hypothesis, stopwords=None):
    # 对中文文本进行分词
    ref_tokens = chinese_tokenize(reference)
    hyp_tokens = chinese_tokenize(hypothesis)

    # 去除停用词
    ref_tokens = remove_stopwords(ref_tokens, stopwords)
    hyp_tokens = remove_stopwords(hyp_tokens, stopwords)

    # 计算词频
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)

    # 计算 Precision
    common_tokens = sum((hyp_counter & ref_counter).values())  # 计算交集部分
    precision = common_tokens / len(hyp_tokens) if len(hyp_tokens) > 0 else 0

    # 计算 Recall
    recall = common_tokens / len(ref_tokens) if len(ref_tokens) > 0 else 0

    return precision, recall

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

# 计算 BLEU(new) 和 SacreBLEU(new)
def compute_bleu_and_sacrebleu_new(references, hypotheses):
    # BLEU(new): 使用加权的n-gram平滑方法
    smoothing = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothing) for ref, hyp in zip(references, hypotheses)]
    avg_bleu_new = np.mean(bleu_scores)
    
    # SacreBLEU(new): 对 BLEU 分数进行修正
    sacre_bleu_new = sacrebleu.corpus_bleu(hypotheses, references, tokenize='zh')
    return avg_bleu_new, sacre_bleu_new.score

def evaluate_metrics(y_true, y_pred, metric, stopwords=None):
    # 对真实标签和预测标签进行分词
    references = y_true
    hypotheses = y_pred

    # 去除停用词
    references = [remove_stopwords(chinese_tokenize(ref), stopwords) for ref in references]
    hypotheses = [remove_stopwords(chinese_tokenize(hyp), stopwords) for hyp in hypotheses]

    # 将分词后的文本转换为字符串（空格连接）
    references_str = [' '.join(ref) for ref in references]
    hypotheses_str = [' '.join(hyp) for hyp in hypotheses]

    # 计算 ROUGE 分数
    rouge1, rouge2, rougeL = [], [], []
    for ref, hyp in zip(references_str, hypotheses_str):
        f1_1, f1_2, f1_L = compute_rouge(ref, hyp)
        rouge1.append(f1_1)
        rouge2.append(f1_2)
        rougeL.append(f1_L)
    
    rouge1_score = np.mean(rouge1)
    rouge2_score = np.mean(rouge2)
    rougeL_score = np.mean(rougeL)

    # 计算 BLEU(new) 和 SacreBLEU(new)
    bleu_score_new, sacre_bleu_score_new = compute_bleu_and_sacrebleu_new(references_str, hypotheses_str)

    # 计算其他指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)

    # 计算Precision和Recall for generation task
    generation_precision, generation_recall = calculate_precision_recall(' '.join(y_true), ' '.join(y_pred), stopwords)

    results = {
        "Accuracy": accuracy,
        "Precision (classification)": precision,
        "Recall (classification)": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "ROUGE-1": rouge1_score,
        "ROUGE-2": rouge2_score,
        "ROUGE-L": rougeL_score,
        "BLEU(new)": bleu_score_new,
        "SacreBLEU(new)": sacre_bleu_score_new,
        "Precision (generation)": generation_precision,
        "Recall (generation)": generation_recall
    }
    
    return results.get(metric, "Metric not found")

st.title("文本评估工具")

st.header("计算其他指标")
true_labels_input = st.text_area("输入真实标签 (专家认定符合要求的参考教学设计)", "这 是 一个 测试， 另一个 测试")
pred_labels_input = st.text_area("输入预测标签 (大语言模型输出的教学设计，切记在内容和形式上与专家认定标准保持一致)", "这 是 一个 测试， 另一个 示例")

metric = st.selectbox("选择评估指标", ["Accuracy", "Precision (classification)", "Recall (classification)", "F1 Score", "MCC", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU(new)", "SacreBLEU(new)", "Precision (generation)", "Recall (generation)"])

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
