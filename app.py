import streamlit as st
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import sacrebleu
from collections import Counter
import string

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

# 计算Precision和Recall
def calculate_precision_recall(reference, hypothesis):
    # 将参考文本和生成文本分词
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # 计算词频
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)

    # 计算 Precision
    common_tokens = sum((hyp_counter & ref_counter).values())  # 计算交集部分
    precision = common_tokens / len(hyp_tokens) if len(hyp_tokens) > 0 else 0

    # 计算 Recall
    recall = common_tokens / len(ref_tokens) if len(ref_tokens) > 0 else 0

    return precision, recall

# 清洗文本（去除标点和小写化）
def clean_text(text):
    # 删除标点符号，并将所有文字转化为小写
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

def evaluate_metrics(y_true, y_pred, metric):
    references = [clean_text(ref) for ref in y_true]  # 清洗参考文本
    hypotheses = [clean_text(pred) for pred in y_pred]  # 清洗生成文本

    # 准备ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 计算ROUGE
    rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(references, hypotheses)]
    rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    # 计算BLEU
    bleu_scores = [sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(references, hypotheses)]
    avg_bleu = np.mean(bleu_scores)

    # 计算SacreBLEU
    sacre_bleu = sacrebleu.corpus_bleu(hypotheses, [[ref] for ref in references])

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
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "BLEU": avg_bleu,
        "SacreBLEU": sacre_bleu.score,
        "Precision (generation)": generation_precision,
        "Recall (generation)": generation_recall
    }
    
    return results.get(metric, "Metric not found")

st.title("文本评估工具")

st.header("计算其他指标")
true_labels_input = st.text_area("输入真实标签 (专家认定符合要求的参考教学设计)", "this is a test, another test")
pred_labels_input = st.text_area("输入预测标签 (大语言模型输出的教学设计，切记在内容和形式上与专家认定标准保持一致)", "this is a test, another example")

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

**困惑度公式：**
- 使用基于词频的简单模型计算词概率。
- 计算每个词的概率的对数总和。
- 归一化，使困惑度值映射到0到100之间。
""")
perplexity_input = st.text_area("输入文本计算困惑度", "this is a sample text for perplexity calculation")

if st.button("计算困惑度"):
    perplexity_result = calculate_perplexity(perplexity_input)
    
    st.subheader("困惑度结果")
    st.write(f"Perplexity: {perplexity_result:.2f}%")
