import os
import re
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from torch.cuda import amp
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
import pyspark.sql.functions as F

# 初始化Spark会话
spark = SparkSession.builder \
    .appName("Spark-BERT") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()


# =============================================================================
# 分布式配置参数
# =============================================================================

class SparkConfig:
    EXECUTORS = 4  # Executor数量
    CORES_PER_EXECUTOR = 4  # 每个Executor核心数
    DATA_DIR = "hdfs:///glue/SST-2"  # HDFS上的数据集路径
    OUTPUT_DIR = "hdfs:///results"  # HDFS输出目录
    PARTITIONS = 64  # 数据分区数


# =============================================================================
# 模型配置参数 (保持与原始代码一致)
# =============================================================================

class ModelConfig:
    HIDDEN_SIZE = 384
    NUM_HIDDEN_LAYERS = 6
    NUM_ATTENTION_HEADS = 6
    INTERMEDIATE_SIZE = 1536
    MAX_LENGTH = 96
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    SEED = 42
    USE_CUDA = torch.cuda.is_available()
    DROPOUT_PROB = 0.1
    USE_AMP = True
    VOCAB_SIZE = 20000


# =============================================================================
# 分布式数据预处理
# =============================================================================

class DistributedTokenizer:
    """分布式Tokenizer实现"""

    def __init__(self):
        self.vocab = self.create_vocab()
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"

        self.unk_token_id = self.vocab[self.unk_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.cls_token_id = self.vocab[self.cls_token]

        self.clean_pattern = re.compile(r'[^\w\s]')
        self.space_pattern = re.compile(r'\s+')

    def create_vocab(self):
        """创建优化的词汇表"""
        vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
        }

        # 添加高频词
        common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
                        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
                        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
                        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
                        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
                        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
                        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"]

        for i, word in enumerate(common_words, len(vocab)):
            vocab[word] = i

        # 添加电影评论相关词汇
        movie_words = ["movie", "film", "good", "bad", "great", "terrible", "acting", "plot",
                       "story", "director", "performance", "scene", "character", "directed",
                       "actors", "cinema", "entertaining", "boring", "watch", "recommend"]

        for i, word in enumerate(movie_words, len(vocab)):
            vocab[word] = i

        return vocab

    def clean_text(self, text):
        """优化的文本清洗"""
        text = self.clean_pattern.sub('', str(text))
        text = self.space_pattern.sub(' ', text)
        return text.strip().lower()

    def tokenize(self, text):
        """基本的分词函数"""
        return text.split()

    def encode(self, text, max_length=ModelConfig.MAX_LENGTH):
        """优化的编码函数"""
        # 清洗文本
        text = self.clean_text(text)

        # 分词并添加特殊token
        tokens = [self.cls_token] + self.tokenize(text)[:max_length - 2] + [self.sep_token]

        # 转换为ID
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        # 填充序列
        pad_len = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.pad_token_id] * pad_len

        return input_ids, attention_mask


# =============================================================================
# 分布式数据加载
# =============================================================================

def load_spark_data():
    """分布式加载SST-2数据集"""
    print("分布式加载SST-2数据集...")

    # 创建分布式Tokenizer
    tokenizer = DistributedTokenizer()

    # 定义编码UDF
    def encode_udf(text):
        input_ids, attention_mask = tokenizer.encode(text)
        return (input_ids, attention_mask)

    # 注册UDF
    encode_udf_spark = udf(
        encode_udf,
        StructType([
            StructField("input_ids", ArrayType(IntegerType())),
            StructField("attention_mask", ArrayType(IntegerType()))
        ])
    )

    # 加载训练数据
    train_df = spark.read.csv(
        f"{SparkConfig.DATA_DIR}/train.tsv",
        sep="\t",
        header=True,
        inferSchema=True
    ).repartition(SparkConfig.PARTITIONS)

    # 应用编码
    train_df = train_df.withColumn(
        "encoded",
        encode_udf_spark(F.col("sentence"))
    ).select(
        F.col("encoded.input_ids").alias("input_ids"),
        F.col("encoded.attention_mask").alias("attention_mask"),
        F.col("label")
    )

    # 加载验证数据
    dev_df = spark.read.csv(
        f"{SparkConfig.DATA_DIR}/dev.tsv",
        sep="\t",
        header=True,
        inferSchema=True
    ).repartition(SparkConfig.PARTITIONS)

    dev_df = dev_df.withColumn(
        "encoded",
        encode_udf_spark(F.col("sentence"))
    ).select(
        F.col("encoded.input_ids").alias("input_ids"),
        F.col("encoded.attention_mask").alias("attention_mask"),
        F.col("label")
    )

    # 加载测试数据
    test_df = spark.read.csv(
        f"{SparkConfig.DATA_DIR}/test.tsv",
        sep="\t",
        header=True,
        inferSchema=True
    ).repartition(SparkConfig.PARTITIONS)

    test_df = test_df.withColumn(
        "encoded",
        encode_udf_spark(F.col("sentence"))
    ).select(
        F.col("encoded.input_ids").alias("input_ids"),
        F.col("encoded.attention_mask").alias("attention_mask")
    )

    return train_df, dev_df, test_df


# =============================================================================
# PyTorch模型定义 (保持与原始代码一致)
# =============================================================================

class BertEmbeddings(nn.Module):
    # 保持原始实现不变
    pass


class BertSelfAttention(nn.Module):
    # 保持原始实现不变
    pass


class BertSelfOutput(nn.Module):
    # 保持原始实现不变
    pass


class BertAttention(nn.Module):
    # 保持原始实现不变
    pass


class BertIntermediate(nn.Module):
    # 保持原始实现不变
    pass


class BertOutput(nn.Module):
    # 保持原始实现不变
    pass


class BertLayer(nn.Module):
    # 保持原始实现不变
    pass


class BertEncoder(nn.Module):
    # 保持原始实现不变
    pass


class BertPooler(nn.Module):
    # 保持原始实现不变
    pass


class BertModel(nn.Module):
    # 保持原始实现不变
    pass


class BertForSequenceClassification(nn.Module):
    # 保持原始实现不变
    pass


# =============================================================================
# 分布式训练函数
# =============================================================================

def distributed_train(model, train_df, dev_df):
    """分布式模型训练"""
    print("开始分布式训练...")

    # 将数据转换为Pandas DataFrame用于训练
    # 在实际生产环境中，这里应该使用Spark的分布式机器学习库
    train_pd = train_df.toPandas()
    dev_pd = dev_df.toPandas()

    # 转换为PyTorch Dataset
    class SparkDataset(Dataset):
        def __init__(self, df):
            self.input_ids = df['input_ids'].apply(lambda x: torch.tensor(x, dtype=torch.long)).tolist()
            self.attention_mask = df['attention_mask'].apply(lambda x: torch.tensor(x, dtype=torch.long)).tolist()
            self.labels = df['label'].apply(lambda x: torch.tensor(x, dtype=torch.long)).tolist()

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]
            }

    train_dataset = SparkDataset(train_pd)
    dev_dataset = SparkDataset(dev_pd)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=ModelConfig.BATCH_SIZE)

    # 设备设置
    device = torch.device("cuda" if ModelConfig.USE_CUDA else "cpu")
    model.to(device)

    # 优化器
    optimizer = Adam(model.parameters(), lr=ModelConfig.LEARNING_RATE)

    # 自动混合精度
    scaler = amp.GradScaler(enabled=ModelConfig.USE_AMP and ModelConfig.USE_CUDA)

    best_accuracy = 0
    best_model_path = ""

    for epoch in range(ModelConfig.EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{ModelConfig.EPOCHS}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with amp.autocast(enabled=ModelConfig.USE_AMP and ModelConfig.USE_CUDA):
                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{ModelConfig.EPOCHS} | Time: {epoch_time:.1f}s | "
              f"Loss: {total_loss / len(train_loader):.4f} | Acc: {accuracy:.4f}")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = f"{SparkConfig.OUTPUT_DIR}/spark_bert_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型至: {best_model_path}")

    return best_model_path


# =============================================================================
# 分布式预测函数
# =============================================================================

def distributed_predict(model, test_df, model_path):
    """分布式预测"""
    print("开始分布式预测...")

    # 加载模型
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if ModelConfig.USE_CUDA else "cpu")
    model.to(device)
    model.eval()

    # 将数据转换为Pandas DataFrame用于预测
    test_pd = test_df.toPandas()

    class TestDataset(Dataset):
        def __init__(self, df):
            self.input_ids = df['input_ids'].apply(lambda x: torch.tensor(x, dtype=torch.long)).tolist()
            self.attention_mask = df['attention_mask'].apply(lambda x: torch.tensor(x, dtype=torch.long)).tolist()

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            }

    test_dataset = TestDataset(test_pd)
    test_loader = DataLoader(test_dataset, batch_size=ModelConfig.BATCH_SIZE * 2)

    # 生成预测
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # 保存预测结果
    results = pd.DataFrame({
        "index": range(len(all_predictions)),
        "prediction": all_predictions
    })

    # 映射为原始标签
    results["prediction"] = results["prediction"].map({0: "negative", 1: "positive"})

    # 保存为TSV
    output_path = f"{SparkConfig.OUTPUT_DIR}/sst2_predictions.tsv"
    results.to_csv(output_path, sep="\t", index=False, header=False)

    return output_path


# =============================================================================
# 主执行流程
# =============================================================================

def main():
    print("=" * 50)
    print("Spark-BERT分布式情感分析系统")
    print(f"Executor数量: {SparkConfig.EXECUTORS}")
    print(f"每个Executor核心数: {SparkConfig.CORES_PER_EXECUTOR}")
    print(f"使用设备: {'GPU' if ModelConfig.USE_CUDA else 'CPU'}")
    print("=" * 50)

    start_time = time.time()

    try:
        # 步骤1: 分布式加载数据
        train_df, dev_df, test_df = load_spark_data()
        print(f"训练集大小: {train_df.count()}")
        print(f"验证集大小: {dev_df.count()}")
        print(f"测试集大小: {test_df.count()}")

        # 步骤2: 初始化模型
        print("初始化BERT模型...")
        model = BertForSequenceClassification(num_labels=2)

        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {total_params:,}")

        # 步骤3: 分布式训练
        model_path = distributed_train(model, train_df, dev_df)

        # 步骤4: 分布式预测
        predictions_path = distributed_predict(model, test_df, model_path)

        # 计算总时间
        elapsed = time.time() - start_time
        print(f"\n总耗时: {elapsed:.2f}秒")

        print("\n=" * 50)
        print("分布式执行完成!")
        print(f"模型已保存: {model_path}")
        print(f"预测结果已保存: {predictions_path}")
        print("=" * 50)

    finally:
        # 停止Spark会话
        spark.stop()


if __name__ == "__main__":
    main()