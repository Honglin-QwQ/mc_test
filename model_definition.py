# model_definition.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Reshape, Permute, Multiply
from tensorflow.keras.models import Model

def build_lstm_ffn_model(
    T_lookback,                   # 时间回看窗口长度 (e.g., 20)
    num_f_features,               # F#数值因子的数量
    symbol_vocab_size,            # Symbol的总类别数 (包括未知符号)
    symbol_embedding_dim,         # Symbol嵌入向量的维度
    hour_vocab_size,              # 小时特征的类别数 (e.g., 24 for 0-23)
    hour_embedding_dim,           # 小时嵌入向量的维度
    day_vocab_size,               # 日期特征的类别数 (e.g., 31 for 0-30 for day_of_month)
    day_embedding_dim,            # 日期嵌入向量的维度
    month_vocab_size,             # 月份特征的类别数 (e.g., 12 for 0-11 for month)
    month_embedding_dim,          # 月份嵌入向量的维度
    year_vocab_size,              # 年份特征的类别数 (e.g., 数量 of unique years in training)
    year_embedding_dim,           # 年份嵌入向量的维度
    lstm_units=64,
    ffn_dense_units=[32],         # FFN隐藏层单元数列表，可以有多个，例如[64, 32]
    dropout_rate=0.2,
    include_time_embeddings=True  # 控制是否使用时间特征的嵌入
):
    """
    构建 LSTM + FFN 模型，包含 symbol 和可选的时间特征嵌入。

    参数:
    T_lookback: 时间序列回看步长
    num_f_features: F#数值因子的数量
    symbol_vocab_size: Symbol 的词汇表大小 (最大整数编码 + 1)
    symbol_embedding_dim: Symbol 嵌入层的输出维度
    hour_vocab_size: 小时特征的词汇表大小 (例如 24)
    hour_embedding_dim: 小时嵌入层的输出维度
    day_vocab_size: 日期 (月中天) 特征的词汇表大小 (例如 31)
    day_embedding_dim: 日期嵌入层的输出维度
    month_vocab_size: 月份特征的词汇表大小 (例如 12)
    month_embedding_dim: 月份嵌入层的输出维度
    year_vocab_size: 年份特征的词汇表大小
    year_embedding_dim: 年份嵌入层的输出维度
    lstm_units: LSTM层的单元数
    ffn_dense_units: 一个整数列表，定义FFN中每个Dense隐藏层的单元数
    dropout_rate: Dropout比率
    include_time_embeddings: 布尔值，是否为时间特征（年、月、日、时）使用嵌入层。
                             如果为False，则假定这些时间特征已被处理为数值型并包含在num_f_features中。
                             (注意：这要求数据加载器相应地准备数据)

    返回:
    TensorFlow Keras Model
    """

    # 1. 定义输入层
    # symbol 输入：每个时间步都相同，但为了与时序特征对齐，也定义为(T_lookback,)
    # 实际上，对于一个序列，symbol通常是固定的。我们可以在数据准备时将symbol嵌入广播到T个时间步，
    # 或者只用一个symbol输入，然后在模型中处理。
    # 为了简单起见，我们假设数据加载器会为每个序列的T个时间步都提供symbol_encoded。
    # 如果symbol对于一个序列是固定的，更高效的做法是只输入一个symbol_id，嵌入一次，然后将嵌入向量
    # 与每个时间步的其他特征拼接，或者作为LSTM的初始状态的一部分。
    # 这里我们先按每个时间步都有特征的通用方式处理，数据加载器需要准备好对应形状的数据。

    # 假设输入的特征被分为几组：
    # - symbol_ids (T_lookback, ) -> 将在每个时间步共享或重复同一个ID
    # - time_feature_ids (T_lookback, num_time_features_to_embed) -> 例如 hour_ids, day_ids etc.
    # - numerical_features (T_lookback, num_f_features + num_time_features_not_embedded)

    # 更灵活的方式是定义一个总的输入，数据加载器负责将所有特征准备好并按顺序排列
    # 这里我们先按特征组定义输入，以便清晰展示嵌入层的使用：

    all_inputs = []
    embedded_features_list = []

    # a. Symbol 输入和嵌入
    symbol_input = Input(shape=(T_lookback,), name='symbol_input_sequence') # 每个时间步的symbol ID
    all_inputs.append(symbol_input)
    symbol_embedded = Embedding(input_dim=symbol_vocab_size, 
                                output_dim=symbol_embedding_dim, 
                                name='symbol_embedding')(symbol_input)
    embedded_features_list.append(symbol_embedded)

    # b. 时间特征输入和嵌入 (如果启用)
    if include_time_embeddings:
        hour_input = Input(shape=(T_lookback,), name='hour_input_sequence')
        day_input = Input(shape=(T_lookback,), name='day_input_sequence')
        month_input = Input(shape=(T_lookback,), name='month_input_sequence')
        year_input = Input(shape=(T_lookback,), name='year_input_sequence')
        
        all_inputs.extend([hour_input, day_input, month_input, year_input])
        
        hour_embedded = Embedding(input_dim=hour_vocab_size, output_dim=hour_embedding_dim, name='hour_embedding')(hour_input)
        day_embedded = Embedding(input_dim=day_vocab_size, output_dim=day_embedding_dim, name='day_embedding')(day_input)
        month_embedded = Embedding(input_dim=month_vocab_size, output_dim=month_embedding_dim, name='month_embedding')(month_input)
        year_embedded = Embedding(input_dim=year_vocab_size, output_dim=year_embedding_dim, name='year_embedding')(year_input)
        
        embedded_features_list.extend([hour_embedded, day_embedded, month_embedded, year_embedded])

    # c. F# 数值因子输入 (这些是已经Z-score标准化过的)
    # 如果时间特征不使用嵌入，它们应该在这里作为数值特征传入
    num_other_numerical_features = num_f_features
    if not include_time_embeddings:
        # 假设年、月、日、时会作为4个额外的数值特征传入
        num_other_numerical_features += 4 
        
    numerical_factors_input = Input(shape=(T_lookback, num_other_numerical_features), name='numerical_factors_input')
    all_inputs.append(numerical_factors_input)

    # 2. 拼接所有特征
    # embedded_features_list 中的每个元素形状是 (batch_size, T_lookback, embedding_dim_for_that_feature)
    # numerical_factors_input 形状是 (batch_size, T_lookback, num_other_numerical_features)
    
    merged_features = Concatenate(axis=-1)(embedded_features_list + [numerical_factors_input])
    # merged_features 的形状现在是 (batch_size, T_lookback, total_concatenated_feature_dim)

    # 3. LSTM 层
    # 我们只关心最后一个时间步的输出，所以 return_sequences=False
    lstm_out = LSTM(lstm_units, return_sequences=False, name='lstm_layer')(merged_features)
    lstm_out_dropout = Dropout(dropout_rate, name='lstm_dropout')(lstm_out)

    # 4. FFN (前馈神经网络) 层
    x = lstm_out_dropout
    for i, units in enumerate(ffn_dense_units):
        x = Dense(units, activation='relu', name=f'ffn_dense_{i+1}')(x)
        x = Dropout(dropout_rate, name=f'ffn_dropout_{i+1}')(x)
    
    output_layer = Dense(1, name='output_layer')(x) # 输出层，预测 target_1 (回归)

    # 5. 构建并返回模型
    model = Model(inputs=all_inputs, outputs=output_layer)
    
    return model

if __name__ == '__main__':
    # 这是一个用于测试模型构建的示例
    T = 20
    num_f_factors = 20 # 假设有20个F#因子
    
    # 假设的词汇表大小和嵌入维度 (这些值需要从您的数据预处理中动态获取)
    sym_vocab = 100  # 假设有99个已知symbol + 1个未知symbol
    sym_emb_dim = 10
    
    h_vocab = 24; h_emb_dim = 4   # 小时
    d_vocab = 31; d_emb_dim = 5   # 日
    m_vocab = 12; m_emb_dim = 4   # 月
    y_vocab = 5;  y_emb_dim = 3   # 假设有5个不同的年份

    # 构建模型 (启用时间嵌入)
    model_with_time_embeddings = build_lstm_ffn_model(
        T_lookback=T,
        num_f_features=num_f_factors,
        symbol_vocab_size=sym_vocab,
        symbol_embedding_dim=sym_emb_dim,
        hour_vocab_size=h_vocab,
        hour_embedding_dim=h_emb_dim,
        day_vocab_size=d_vocab,
        day_embedding_dim=d_emb_dim,
        month_vocab_size=m_vocab,
        month_embedding_dim=m_emb_dim,
        year_vocab_size=y_vocab,
        year_embedding_dim=y_emb_dim,
        lstm_units=64,
        ffn_dense_units=[32],
        dropout_rate=0.3,
        include_time_embeddings=True
    )
    print("\n--- 模型结构 (包含时间特征嵌入) ---")
    model_with_time_embeddings.summary()

    # 构建模型 (不启用时间嵌入，时间特征作为数值特征传入)
    # 在这种情况下，num_f_features 应该包含 F#因子 和 4个（标准化后的）时间特征
    model_without_time_embeddings = build_lstm_ffn_model(
        T_lookback=T,
        num_f_features=num_f_factors + 4, # F#因子 + 4个数值型时间特征
        symbol_vocab_size=sym_vocab,
        symbol_embedding_dim=sym_emb_dim,
        # 以下时间相关的 vocab 和 emb_dim 在 include_time_embeddings=False 时不会被使用
        hour_vocab_size=h_vocab, hour_embedding_dim=h_emb_dim,
        day_vocab_size=d_vocab, day_embedding_dim=d_emb_dim,
        month_vocab_size=m_vocab, month_embedding_dim=m_emb_dim,
        year_vocab_size=y_vocab, year_embedding_dim=y_emb_dim,
        lstm_units=64,
        ffn_dense_units=[32],
        dropout_rate=0.3,
        include_time_embeddings=False
    )
    print("\n--- 模型结构 (不包含时间特征嵌入，时间特征作为数值特征) ---")
    model_without_time_embeddings.summary()