import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import time
from bs4 import BeautifulSoup


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")

def plot_history(history, title_prefix):
    """绘制并保存模型的训练历史曲线"""
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'rs-', label='Validation loss')
    plt.title(f'{title_prefix} - Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'rs-', label='Validation acc')
    plt.title(f'{title_prefix} - Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # 保存图像文件
    filename = f"{title_prefix.replace(' ', '_').lower()}_history.png"
    plt.savefig(filename)
    print(f"训练历史图已保存为: {filename}")

def get_rnn_lstm_models(vocab_size, embedding_dim=128):
    """构建 RNN 和 LSTM 模型"""
    # RNN 模型
    rnn_model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        layers.SimpleRNN(128),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])
    
    # LSTM 模型
    lstm_model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
        layers.LSTM(128),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # 编译模型
    for model in [rnn_model, lstm_model]:
        model.compile(optimizer='adam',
                      loss=losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
    return rnn_model, lstm_model

def main():

    # 加载 Keras 内置数据集
    VOCAB_SIZE = 10000
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")

    print("\n随机样本展示 (3例):")
    for i in range(3):
        idx = random.randint(0, len(train_data) - 1)
        print(f"--- 随机样本 {i+1} ---")
        print(f"  - 编码序列 (部分): {train_data[idx][:15]}...")
        print(f"  - 标签值: {train_labels[idx]}")
        print(f"  - 序列长度: {len(train_data[idx])}")

    # 获取词-索引映射字典
    word_index = tf.keras.datasets.imdb.get_word_index()

    # 创建 索引-词 映射字典, 注意: Keras 索引偏移了 3 (0: pad, 1: start, 2: unk)
    reverse_word_index = {v + 3: k for k, v in word_index.items()}
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<OOV>' 

    def decode_review(text_sequence):
        """将整数序列解码为可读文本"""
        return ' '.join([reverse_word_index.get(i, '?') for i in text_sequence])

    print("\n--- 解码验证 (抽样3例) ---")
    for i in [0, 100, 200]: # 选择固定的几个样本以便复现
        print(f"\n--- 样本 {i} ---")
        print(f"原始编码序列 (部分): {train_data[i][:20]}...")
        print(f"解码后文本: {decode_review(train_data[i])}")

    MAX_LEN = 250
    train_data_padded = pad_sequences(train_data, maxlen=MAX_LEN, padding='post', truncating='post')
    test_data_padded = pad_sequences(test_data, maxlen=MAX_LEN, padding='post', truncating='post')
    print(f"\nPadding后训练数据维度: {train_data_padded.shape}")

    rnn_model_pre, lstm_model_pre = get_rnn_lstm_models(vocab_size=VOCAB_SIZE)
    
    print("\n--- 开始训练 RNN 模型 (内置数据) ---")
    rnn_history = rnn_model_pre.fit(train_data_padded, train_labels,
                                    epochs=1,
                                    batch_size=128,
                                    validation_data=(test_data_padded, test_labels),
                                    verbose=1)
    
    print("\n--- 开始训练 LSTM 模型 (内置数据) ---")
    lstm_history = lstm_model_pre.fit(train_data_padded, train_labels,
                                      epochs=1,
                                      batch_size=128,
                                      validation_data=(test_data_padded, test_labels),
                                      verbose=1)

    # 性能对比
    print("\n--- 性能对比 (内置数据) ---")
    print("RNN 模型参数量:", rnn_model_pre.count_params())
    print("LSTM 模型参数量:", lstm_model_pre.count_params())
    
    rnn_loss, rnn_acc = rnn_model_pre.evaluate(test_data_padded, test_labels, verbose=0)
    lstm_loss, lstm_acc = lstm_model_pre.evaluate(test_data_padded, test_labels, verbose=0)
    print(f"RNN 最终测试集准确率: {rnn_acc*100:.2f}%")
    print(f"LSTM 最终测试集准确率: {lstm_acc*100:.2f}%")
    
    plot_history(rnn_history, "Pre-processed RNN")
    plot_history(lstm_history, "Pre-processed LSTM")

    # 下载并解压原始数据集
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    # 移除不相关的 unsup 文件夹
    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.exists(remove_dir):
        shutil.rmtree(remove_dir)

    # 使用 text_dataset_from_directory 创建数据集
    BATCH_SIZE = 32
    SEED = 42
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        os.path.join(dataset_dir, 'train'),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=SEED)
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        os.path.join(dataset_dir, 'train'),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=SEED)
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        os.path.join(dataset_dir, 'test'),
        batch_size=BATCH_SIZE)
    
    # 构建 TextVectorization 层来处理分词和词汇表
    VOCAB_SIZE_RAW = 10000
    MAX_LEN_RAW = 250
    vectorize_layer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE_RAW,
        output_mode='int',
        output_sequence_length=MAX_LEN_RAW)
    
    # 适配层到训练数据上，构建词汇表
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    # 模型构建、训练与对比
    def create_model_from_raw(model_type='lstm'):
        model = models.Sequential([
            vectorize_layer,
            layers.Embedding(input_dim=VOCAB_SIZE_RAW, output_dim=128, mask_zero=True),
            layers.LSTM(128) if model_type == 'lstm' else layers.SimpleRNN(128),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss=losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        return model

    rnn_model_raw = create_model_from_raw('rnn')
    lstm_model_raw = create_model_from_raw('lstm')
    
    print("\n--- 开始训练 RNN 模型 (原始数据) ---")
    rnn_history_raw = rnn_model_raw.fit(raw_train_ds,
                                        validation_data=raw_val_ds,
                                        epochs=1,
                                        verbose=1)
    
    print("\n--- 开始训练 LSTM 模型 (原始数据) ---")
    lstm_history_raw = lstm_model_raw.fit(raw_train_ds,
                                          validation_data=raw_val_ds,
                                          epochs=1,
                                          verbose=1)

    # 结果对比
    print("\n--- 性能对比 (原始数据) ---")
    print("RNN (raw) 模型参数量:", rnn_model_raw.count_params())
    print("LSTM (raw) 模型参数量:", lstm_model_raw.count_params())
    
    rnn_loss_raw, rnn_acc_raw = rnn_model_raw.evaluate(raw_test_ds, verbose=0)
    lstm_loss_raw, lstm_acc_raw = lstm_model_raw.evaluate(raw_test_ds, verbose=0)
    print(f"RNN (raw) 最终测试集准确率: {rnn_acc_raw*100:.2f}%")
    print(f"LSTM (raw) 最终测试集准确率: {lstm_acc_raw*100:.2f}%")

    plot_history(rnn_history_raw, "Raw Text RNN")
    plot_history(lstm_history_raw, "Raw Text LSTM")

if __name__ == '__main__':
    main()