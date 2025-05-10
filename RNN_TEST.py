# RNN_TEST.py

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pyarrow.feather as pf
import pyarrow.compute as pc 
import traceback

from preprocessing import get_or_create_preprocessing_outputs 
from model_definition import build_lstm_ffn_model

# --- 0. 配置参数 (与之前相同) ---
FEATHER_PATH = 'factor_results_4小时_spot_4h.feather' 
STATS_TABLE_CSV_PATH = 'normalization_stats.csv' 
SYMBOL_REPORT_CSV_PATH = 'symbol_mapping_report.csv'
F_PREFIX = 'F#'
T_LOOKBACK = 20        
SYMBOL_EMBEDDING_DIM = 10 
# ... (其他超参数与上一版本相同) ...
HOUR_EMBEDDING_DIM = 4; DAY_EMBEDDING_DIM = 5; MONTH_EMBEDDING_DIM = 4; YEAR_EMBEDDING_DIM = 3    
LSTM_UNITS = 64; FFN_DENSE_UNITS = [32]; DROPOUT_RATE = 0.2; LEARNING_RATE = 0.001
BATCH_SIZE = 64; EPOCHS = 50 
VALIDATION_SPLIT_FROM_TRAIN_RATIO = 0.2; TEST_SPLIT_RATIO = 0.3 
MODEL_SAVE_PATH = 'best_lstm_ffn_model.keras'; TRAINING_HISTORY_PLOT_PATH = 'training_history.png'
PREDICTION_PLOT_PATH = 'test_predictions_vs_actuals.png'

# --- 全局变量 (由 main 函数填充，preprocess_dataframe_chunk 会使用) ---
year_to_int_mapping = {}
YEAR_VOCAB_SIZE = 0

# --- 辅助函数 (create_sequences_for_symbol, prepare_model_inputs_from_concatenated_X, preprocess_dataframe_chunk 与之前相同) ---
# ... (此处省略这些函数的代码，假设它们与您上一版本相同) ...
def create_sequences_for_symbol(df_symbol_processed, T, feature_cols_ordered, target_col_name_scaled):
    X_sequences, y_targets = [], []
    actual_feature_cols = [col for col in feature_cols_ordered if col in df_symbol_processed.columns]
    if len(actual_feature_cols) != len(feature_cols_ordered) or not actual_feature_cols:
        return np.array(X_sequences), np.array(y_targets)
    data_np = df_symbol_processed[actual_feature_cols].values
    if target_col_name_scaled not in df_symbol_processed.columns:
        return np.array(X_sequences), np.array(y_targets)
    target_np = df_symbol_processed[target_col_name_scaled].values
    if len(data_np) <= T: return np.array(X_sequences), np.array(y_targets)
    for i in range(len(data_np) - T): 
        if (i + T) < len(target_np): 
            X_sequences.append(data_np[i : i + T])
            y_targets.append(target_np[i + T]) 
        else: break 
    return np.array(X_sequences), np.array(y_targets)

def prepare_model_inputs_from_concatenated_X(X_concatenated_np, num_f_features_in_model):
    inputs = {
        'symbol_input_sequence': X_concatenated_np[:, :, 0].astype(np.int32),
        'year_input_sequence': X_concatenated_np[:, :, 1].astype(np.int32),
        'month_input_sequence': X_concatenated_np[:, :, 2].astype(np.int32),
        'day_input_sequence': X_concatenated_np[:, :, 3].astype(np.int32),
        'hour_input_sequence': X_concatenated_np[:, :, 4].astype(np.int32),
        'numerical_factors_input': X_concatenated_np[:, :, 5 : 5 + num_f_features_in_model].astype(np.float32)
    }
    if inputs['numerical_factors_input'].shape[-1] != num_f_features_in_model:
        print(f"ERROR in prepare_model_inputs: Expected {num_f_features_in_model} numerical features, but got {inputs['numerical_factors_input'].shape[-1]}")
    return inputs

def preprocess_dataframe_chunk(df_chunk, norm_table, sym_report, current_f_prefix='F#'):
    global year_to_int_mapping 
    if df_chunk.empty: return pd.DataFrame(), [], "" 
    df_processed_chunk = df_chunk.copy()
    if 'dt' in df_processed_chunk.columns:
        df_processed_chunk['dt'] = pd.to_datetime(df_processed_chunk['dt'], errors='coerce')
        df_processed_chunk.dropna(subset=['dt'], inplace=True)
        if df_processed_chunk.empty: return pd.DataFrame(), [], ""
        df_processed_chunk['year_orig'] = df_processed_chunk['dt'].dt.year
        df_processed_chunk['month'] = df_processed_chunk['dt'].dt.month - 1 
        df_processed_chunk['day'] = df_processed_chunk['dt'].dt.day - 1   
        df_processed_chunk['hour'] = df_processed_chunk['dt'].dt.hour    
        if 'year_orig' in df_processed_chunk.columns and year_to_int_mapping:
             df_processed_chunk['year_encoded'] = df_processed_chunk['year_orig'].map(year_to_int_mapping).fillna(len(year_to_int_mapping)) 
        else: df_processed_chunk['year_encoded'] = 0 
    else: 
        for col in ['dt','year_orig','month', 'day', 'hour', 'year_encoded']: df_processed_chunk[col] = 0 if col != 'dt' else pd.NaT
    future_unknown_code = sym_report.get("__Schema_for_Future_Unknown_and_NaN_Symbol__", -1)
    if future_unknown_code == -1 : print("错误: sym_report中未知符号定义缺失!");
    def _encode_symbol(symbol_val):
        if pd.isna(symbol_val): return future_unknown_code
        return sym_report.get(str(symbol_val), future_unknown_code)
    if 'symbol' in df_processed_chunk.columns:
        df_processed_chunk['symbol_encoded'] = df_processed_chunk['symbol'].apply(_encode_symbol)
    else: df_processed_chunk['symbol_encoded'] = future_unknown_code
    f_factor_cols = [col for col in df_processed_chunk.columns if col.startswith(current_f_prefix)]
    scaled_f_cols_list = [] 
    cols_to_scale = f_factor_cols + (['target_1'] if 'target_1' in df_processed_chunk.columns else [])
    for col in cols_to_scale:
        if col in norm_table.columns and col in df_processed_chunk.columns:
            mean = norm_table.loc['mean', col]; std = norm_table.loc['std_dev', col]
            scaled_col_name = col + '_scaled'
            if pd.notna(mean) and pd.notna(std) and std != 0: df_processed_chunk[scaled_col_name] = (df_processed_chunk[col] - mean) / std
            elif std == 0: df_processed_chunk[scaled_col_name] = 0 
            else: df_processed_chunk[scaled_col_name] = df_processed_chunk[col] 
            if col.startswith(current_f_prefix): scaled_f_cols_list.append(scaled_col_name)
        elif col in df_processed_chunk.columns:
             print(f"警告：列 '{col}' 在统计表中未找到，无法标准化。将使用原始值。")
             df_processed_chunk[col + '_scaled'] = df_processed_chunk[col]
             if col.startswith(current_f_prefix): scaled_f_cols_list.append(col + '_scaled')
    final_scaled_target_col_name = 'target_1_scaled'
    if final_scaled_target_col_name not in df_processed_chunk.columns and 'target_1' in df_processed_chunk.columns:
        if 'target_1' in df_processed_chunk: df_processed_chunk[final_scaled_target_col_name] = df_processed_chunk['target_1']
    return df_processed_chunk, scaled_f_cols_list, final_scaled_target_col_name

def main():
    global year_to_int_mapping, YEAR_VOCAB_SIZE 

    print(f"--- 主程序 RNN_TEST.py 开始 ---")
    
    # 从 preprocessing.py 获取预处理结果
    norm_table, sym_report, min_time, max_time, original_feather_rows, all_feather_cols_from_preprocessing = \
        get_or_create_preprocessing_outputs(
            FEATHER_PATH, STATS_TABLE_CSV_PATH, SYMBOL_REPORT_CSV_PATH,
            f_prefix=F_PREFIX, verbose=True
        )

    # 关键信息检查 (与之前版本相同)
    if "__Error__" in sym_report or pd.isna(min_time) or pd.isna(max_time) or not all_feather_cols_from_preprocessing:
        print("错误：预处理模块未能成功加载或生成关键信息。程序终止。")
        # ... (打印错误详情) ...
        return
    print(f"预处理完成。数据时间范围: {min_time} 至 {max_time}。原始文件共 {original_feather_rows} 行, {len(all_feather_cols_from_preprocessing)} 列。")

    # --- 步骤 0.1: 定义 SYMBOL_VOCAB_SIZE (与之前版本相同) ---
    # ... (future_unknown_code 和 SYMBOL_VOCAB_SIZE 的计算逻辑) ...
    future_unknown_code = sym_report.get("__Schema_for_Future_Unknown_and_NaN_Symbol__")
    if future_unknown_code is None: 
        # ... (错误处理或回退逻辑) ...
        valid_codes = [v for k, v in sym_report.items() if not k.startswith("__") and isinstance(v, (int, np.integer))]
        if not valid_codes: print("错误: sym_report 中无有效编码。"); return
        future_unknown_code = max(valid_codes) + 1 
        print(f"警告: __Schema_for_Future_Unknown_and_NaN_Symbol__ 未定义，临时设为 {future_unknown_code}")
    SYMBOL_VOCAB_SIZE = future_unknown_code + 1
    print(f"根据 sym_report 确定 SYMBOL_VOCAB_SIZE: {SYMBOL_VOCAB_SIZE}")


    # --- 步骤 0.2: 建立全局年份映射和 YEAR_VOCAB_SIZE (与之前版本相同) ---
    # ... (year_to_int_mapping 和 YEAR_VOCAB_SIZE 的计算逻辑) ...
    print("\n--- 步骤 0.2: 建立全局年份映射 ---")
    try:
        # ... (代码与上一版本相同) ...
        if 'dt' not in all_feather_cols_from_preprocessing:
            print(f"错误: 'dt'列不在Feather文件的列名列表 ({all_feather_cols_from_preprocessing})中，无法进行年份映射。"); return
        dt_year_table = pf.read_table(FEATHER_PATH, columns=['dt'])
        dt_series_for_year_pd = dt_year_table.column('dt').to_pandas(strings_to_categorical=False, zero_copy_only=False, self_destruct=True)
        del dt_year_table
        years_for_mapping = pd.to_datetime(dt_series_for_year_pd, errors='coerce').dt.year.dropna().unique()
        if len(years_for_mapping) > 0:
            unique_years_in_data_sorted = sorted(years_for_mapping)
            year_to_int_mapping.update({year: i for i, year in enumerate(unique_years_in_data_sorted)})
            YEAR_VOCAB_SIZE = len(year_to_int_mapping)
            print(f"全局年份映射已创建，YEAR_VOCAB_SIZE: {YEAR_VOCAB_SIZE}")
        else: YEAR_VOCAB_SIZE = 1; print("警告：无法从'dt'列提取有效年份。YEAR_VOCAB_SIZE 设为1。")
        del dt_series_for_year_pd, years_for_mapping
    except Exception as e: YEAR_VOCAB_SIZE = 1; print(f"错误：预读取'dt'列以建立年份映射失败: {e}。YEAR_VOCAB_SIZE 设为1。")


    # --- 步骤 1: 准备按 Symbol 处理数据 ---
    print(f"\n--- 步骤 1: 准备按 Symbol 处理数据 (使用已获取的列名列表) ---")
    
    # =====================================================================
    # 定义 f_factor_cols_original_names 在这里!
    # =====================================================================
    f_factor_cols_original_names = [col for col in all_feather_cols_from_preprocessing if col.startswith(F_PREFIX)]
    
    # 然后用它来构建 cols_to_load_for_processing
    essential_cols = ['symbol', 'dt', 'target_1']
    cols_to_load_for_processing = list(set(essential_cols + f_factor_cols_original_names))
    cols_to_load_for_processing = [col for col in cols_to_load_for_processing if col in all_feather_cols_from_preprocessing] 
    
    if 'target_1' not in cols_to_load_for_processing :
        print(f"致命错误: 'target_1' 未被选中加载 (检查 all_feather_cols_from_preprocessing: {all_feather_cols_from_preprocessing})，无法继续!"); return
    if not any(col.startswith(F_PREFIX) for col in cols_to_load_for_processing):
        print(f"警告: 没有F#因子列被选中加载。检查 F_PREFIX ('{F_PREFIX}') 和 all_feather_cols_from_preprocessing。")

    print(f"将从Feather文件读取以下核心列进行处理: {cols_to_load_for_processing}")
    arrow_table_main_data = None
    try:
        arrow_table_main_data = pf.read_table(FEATHER_PATH, columns=cols_to_load_for_processing)
        print(f"核心列数据已加载到 Arrow Table，包含 {arrow_table_main_data.num_rows} 行。")
    except Exception as e:
        print(f"错误: 加载核心列数据到 Arrow Table 失败: {e}"); return

    # ... (获取 unique_symbols_from_report 的逻辑) ...
    unique_symbols_from_report = [s for s in sym_report.keys() if not s.startswith("__")]
    if not unique_symbols_from_report: # Fallback
        # ... (与上一版本相同的 fallback 逻辑) ...
        print("警告：无法从 sym_report 获取唯一 symbol 列表。尝试从原始数据读取...")
        try:
            if 'symbol' not in all_feather_cols_from_preprocessing: print("错误: 'symbol' 列不在 Feather 文件中。"); return
            symbol_table_raw = pf.read_table(FEATHER_PATH, columns=['symbol'])
            unique_symbols_from_report = symbol_table_raw['symbol'].to_pandas(strings_to_categorical=False).unique().tolist()
            unique_symbols_from_report = [s for s in unique_symbols_from_report if pd.notna(s)] 
            del symbol_table_raw
            if not unique_symbols_from_report: print("错误：原始数据中也未能找到有效符号。"); return
            print(f"从原始数据中动态获取到 {len(unique_symbols_from_report)} 个符号。")
        except Exception as e_sym: print(f"错误：尝试从原始数据读取 'symbol' 列失败: {e_sym}。"); return

    # ... (初始化 all_X_train_list 等列表) ...
    all_X_train_list, all_y_train_list = [], []
    all_X_val_list, all_y_val_list = [], []
    all_X_test_list, all_y_test_list = [], []
    df_test_info_list = []

    # ... (定义时间划分点 min_time_dt, max_time_dt, train_val_cutoff_time) ...
    min_time_dt = pd.to_datetime(min_time); max_time_dt = pd.to_datetime(max_time)
    if pd.isna(min_time_dt) or pd.isna(max_time_dt) : print("错误: min_time 或 max_time 无效!"); return
    total_duration_seconds = (max_time_dt - min_time_dt).total_seconds()
    if total_duration_seconds <=0: print("错误: max_time不大于min_time!"); return
    train_val_cutoff_time = min_time_dt + pd.to_timedelta(total_duration_seconds * (1 - TEST_SPLIT_RATIO), unit='s')


    # --- 按 Symbol 循环处理数据 ---
    for current_sym_name in unique_symbols_from_report:
        # ... (内部逻辑与上一版本相同，使用 arrow_table_main_data, preprocess_dataframe_chunk, create_sequences_for_symbol) ...
        # print(f"\n  正在处理 Symbol: {current_sym_name} ...") # 可以取消注释以跟踪进度
        try:
            condition = pc.equal(arrow_table_main_data.column('symbol'), current_sym_name)
            arrow_table_symbol_specific = arrow_table_main_data.filter(condition)
            if arrow_table_symbol_specific.num_rows == 0: del arrow_table_symbol_specific; continue
            df_symbol_chunk = arrow_table_symbol_specific.to_pandas(strings_to_categorical=False, zero_copy_only=False, self_destruct=True)
            del arrow_table_symbol_specific
            if df_symbol_chunk.empty: continue

            df_sym_proc, current_scaled_f_cols, current_scaled_target_col = \
                preprocess_dataframe_chunk(df_symbol_chunk, norm_table, sym_report, F_PREFIX)

            if df_sym_proc.empty or not current_scaled_f_cols or not current_scaled_target_col or current_scaled_target_col not in df_sym_proc: continue
            df_sym_proc = df_sym_proc.sort_values(by='dt')

            s_test_df = df_sym_proc[df_sym_proc['dt'] > train_val_cutoff_time] 
            s_train_val_df = df_sym_proc[df_sym_proc['dt'] <= train_val_cutoff_time]
            s_train_df, s_val_df = pd.DataFrame(), pd.DataFrame()
            if not s_train_val_df.empty:
                s_min_dt_tv = s_train_val_df['dt'].min(); s_max_dt_tv = s_train_val_df['dt'].max()
                if pd.notna(s_min_dt_tv) and pd.notna(s_max_dt_tv) and s_max_dt_tv > s_min_dt_tv:
                    s_tv_duration_seconds = (s_max_dt_tv - s_min_dt_tv).total_seconds()
                    s_train_cutoff = s_min_dt_tv + pd.to_timedelta(s_tv_duration_seconds * (1 - VALIDATION_SPLIT_FROM_TRAIN_RATIO), unit='s')
                    s_val_df = s_train_val_df[s_train_val_df['dt'] > s_train_cutoff]
                    s_train_df = s_train_val_df[s_train_val_df['dt'] <= s_train_cutoff]
                else: s_train_df = s_train_val_df
            
            _feat_cols_seq = ['symbol_encoded', 'year_encoded', 'month', 'day', 'hour'] + current_scaled_f_cols
            
            valid_seq_cols = True
            for col_chk in _feat_cols_seq + [current_scaled_target_col]:
                if col_chk not in df_sym_proc.columns: valid_seq_cols = False; break
            if not valid_seq_cols: print(f"警告: Symbol {current_sym_name} 的处理数据中缺少关键列，跳过序列生成。"); continue

            if not s_train_df.empty:
                Xs_tr, ys_tr = create_sequences_for_symbol(s_train_df, T_LOOKBACK, _feat_cols_seq, current_scaled_target_col)
                if Xs_tr.size > 0: all_X_train_list.append(Xs_tr); all_y_train_list.append(ys_tr)
            if not s_val_df.empty:
                Xs_v, ys_v = create_sequences_for_symbol(s_val_df, T_LOOKBACK, _feat_cols_seq, current_scaled_target_col)
                if Xs_v.size > 0: all_X_val_list.append(Xs_v); all_y_val_list.append(ys_v)
            if not s_test_df.empty:
                Xs_te, ys_te = create_sequences_for_symbol(s_test_df, T_LOOKBACK, _feat_cols_seq, current_scaled_target_col)
                if Xs_te.size > 0:
                    all_X_test_list.append(Xs_te); all_y_test_list.append(ys_te)
                    num_seq = len(ys_te)
                    if num_seq > 0 and 'target_1' in s_test_df.columns and 'dt' in s_test_df.columns:
                         df_test_info_list.append(pd.DataFrame({
                            'symbols': [current_sym_name] * num_seq,
                            'dates': s_test_df['dt'].iloc[T_LOOKBACK : T_LOOKBACK + num_seq].tolist(),
                            'original_target_1': s_test_df['target_1'].iloc[T_LOOKBACK : T_LOOKBACK + num_seq].tolist()}))
        except Exception as e_sym_proc_loop:
            print(f"    处理 Symbol {current_sym_name} 循环时发生错误: {e_sym_proc_loop}"); traceback.print_exc(); continue

    if arrow_table_main_data is not None: del arrow_table_main_data

    # --- 整合所有 symbol 的序列数据 (与之前相同) ---
    if not all_X_train_list : print("错误：训练集序列为空，无法继续。"); return
    X_train = np.concatenate(all_X_train_list, axis=0); y_train = np.concatenate(all_y_train_list, axis=0)
    # ... (其他 X_val, y_val, X_test, y_test, df_test_info 的整合)
    X_val = np.concatenate(all_X_val_list, axis=0) if all_X_val_list else np.array([])
    y_val = np.concatenate(all_y_val_list, axis=0) if all_y_val_list else np.array([])
    X_test = np.concatenate(all_X_test_list, axis=0) if all_X_test_list else np.array([])
    y_test = np.concatenate(all_y_test_list, axis=0) if all_y_test_list else np.array([])
    df_test_info = pd.concat(df_test_info_list, ignore_index=True) if df_test_info_list else pd.DataFrame()

    print(f"最终训练序列形状: X={X_train.shape}, y={y_train.shape}")
    print(f"最终验证序列形状: X={X_val.shape}, y={y_val.shape}")
    print(f"最终测试序列形状: X={X_test.shape}, y={y_test.shape}")

    if X_train.shape[0] == 0 : print("错误：没有有效的训练数据序列。"); return
    
    # =====================================================================
    # 定义 num_actual_f_features 在这里! (基于之前定义的 f_factor_cols_original_names)
    # =====================================================================
    num_actual_f_features = len(f_factor_cols_original_names) 
    print(f"模型将使用 {num_actual_f_features} 个 F# 数值特征。")

    train_model_inputs = prepare_model_inputs_from_concatenated_X(X_train, num_actual_f_features)
    # ... (val_model_inputs, test_model_inputs 的准备)
    val_model_inputs = prepare_model_inputs_from_concatenated_X(X_val, num_actual_f_features) if X_val.size > 0 else None
    test_model_inputs = prepare_model_inputs_from_concatenated_X(X_test, num_actual_f_features) if X_test.size > 0 else None
        
    # --- 5. 构建并编译模型 ---
    print(f"\n--- 步骤 5: 构建并编译 LSTM 模型 ---")
    HOUR_VOCAB_SIZE = 24; DAY_VOCAB_SIZE = 31; MONTH_VOCAB_SIZE = 12 
    if YEAR_VOCAB_SIZE == 0: YEAR_VOCAB_SIZE = 1 # 确保不为0
    if SYMBOL_VOCAB_SIZE == 0: SYMBOL_VOCAB_SIZE = 1 # 确保不为0

    model = build_lstm_ffn_model( 
        T_lookback=T_LOOKBACK, num_f_features=num_actual_f_features, 
        symbol_vocab_size=SYMBOL_VOCAB_SIZE, symbol_embedding_dim=SYMBOL_EMBEDDING_DIM,
        hour_vocab_size=HOUR_VOCAB_SIZE, hour_embedding_dim=HOUR_EMBEDDING_DIM,
        day_vocab_size=DAY_VOCAB_SIZE, day_embedding_dim=DAY_EMBEDDING_DIM,
        month_vocab_size=MONTH_VOCAB_SIZE, month_embedding_dim=MONTH_EMBEDDING_DIM,
        year_vocab_size=YEAR_VOCAB_SIZE, year_embedding_dim=YEAR_EMBEDDING_DIM,
        lstm_units=LSTM_UNITS, ffn_dense_units=FFN_DENSE_UNITS, dropout_rate=DROPOUT_RATE,
        include_time_embeddings=True 
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary(line_length=120)

    # --- 6. 训练模型 (与之前相同) ---
    # ...
    print(f"\n--- 步骤 6: 训练模型 ---")
    callbacks_list = [ EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1) ]
    validation_data_for_fit = (val_model_inputs, y_val) if val_model_inputs is not None and y_val.size > 0 else None
    if validation_data_for_fit is None and X_val.size > 0 : print("警告: 验证特征不为空但标签为空，或反之。不使用验证集。") 

    history = model.fit( train_model_inputs, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=validation_data_for_fit, callbacks=callbacks_list, verbose=1 )


    print(f"\n--- 步骤 7: 加载最佳模型并评估 ---")
    best_model = model 
    if os.path.exists(MODEL_SAVE_PATH):
        try: best_model = tf.keras.models.load_model(MODEL_SAVE_PATH); print(f"最佳模型已加载。")
        except Exception as e: print(f"加载最佳模型失败: {e}. 使用当前模型。")
    else: print(f"模型检查点 '{MODEL_SAVE_PATH}' 未找到。使用当前模型。")

    if test_model_inputs is None or y_test.size == 0: print("测试数据为空，跳过评估。")
    else:
        test_loss_scaled, test_mae_scaled = best_model.evaluate(test_model_inputs, y_test, verbose=0)
        print(f"测试集 (缩放尺度) - Loss (MSE): {test_loss_scaled:.6f}, MAE: {test_mae_scaled:.6f}")
        y_pred_scaled = best_model.predict(test_model_inputs).flatten()

        target_mean, target_std = np.nan, np.nan
        if 'target_1' in norm_table.columns:
            target_mean = norm_table.loc['mean', 'target_1']
            target_std = norm_table.loc['std_dev', 'target_1']
        
        y_pred_original = y_pred_scaled 
        y_test_original = df_test_info['original_target_1'].values if not df_test_info.empty else np.array([])

        if pd.notna(target_mean) and pd.notna(target_std) and target_std != 0:
            y_pred_original = (y_pred_scaled * target_std) + target_mean
            if y_test_original.size == y_pred_original.size and y_test_original.size > 0:
                r2_orig = r2_score(y_test_original, y_pred_original)
                mse_orig = mean_squared_error(y_test_original, y_pred_original)
                mae_orig = mean_absolute_error(y_test_original, y_pred_original)
                print(f"测试集 (原始尺度) - R-squared: {r2_orig:.4f}, MSE: {mse_orig:.6f}, MAE: {mae_orig:.6f}")
            else: print("警告: 原始测试目标和预测长度不匹配或为空 (原始尺度)。")
        else:
            print("警告：无法获取 target_1 的有效均值/标准差进行反向转换。")
            if y_test.size == y_pred_scaled.size and y_test.size > 0: # y_test is scaled
                r2_scaled = r2_score(y_test, y_pred_scaled) 
                print(f"测试集 (缩放尺度) - R-squared: {r2_scaled:.4f}")
        
        print(f"\n--- 步骤 9: 可视化结果 ---")
        plt.figure(figsize=(12, 6)) # 训练历史图
        plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss'); 
        if 'val_loss' in history.history: plt.plot(history.history['val_loss'], label='Val Loss');
        plt.title('Model Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.subplot(1, 2, 2); plt.plot(history.history['mae'], label='Train MAE'); 
        if 'val_mae' in history.history: plt.plot(history.history['val_mae'], label='Val MAE');
        plt.title('Model MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()
        plt.tight_layout(); plt.savefig(TRAINING_HISTORY_PLOT_PATH); print(f"训练历史图已保存到: {TRAINING_HISTORY_PLOT_PATH}")

        if y_test_original.size > 0 and y_pred_original.size > 0 and not df_test_info.empty:
            if len(y_pred_original) == len(df_test_info): # 确保长度一致
                df_test_info['predicted_target_1_original'] = y_pred_original
                plt.figure(figsize=(18, 8)) # 测试集对比图
                num_symbols_to_plot = min(5, df_test_info['symbols'].nunique())
                plot_idx = 1
                if num_symbols_to_plot > 0:
                    for sym_plot, group_data_plot in df_test_info.groupby('symbols'):
                        if plot_idx > num_symbols_to_plot: break
                        plt.subplot(num_symbols_to_plot, 1, plot_idx)
                        plt.plot(pd.to_datetime(group_data_plot['dates']), group_data_plot['original_target_1'], label=f'Actual ({sym_plot})', alpha=0.8)
                        plt.plot(pd.to_datetime(group_data_plot['dates']), group_data_plot['predicted_target_1_original'], label=f'Predicted ({sym_plot})', linestyle='--', alpha=0.8)
                        plt.title(f'Test Set: Actual vs. Predicted for {sym_plot} (Original Scale)')
                        plt.ylabel('Target_1 Value'); plt.legend(); plt.xticks(rotation=30)
                        plot_idx += 1
                    plt.tight_layout(); plt.savefig(PREDICTION_PLOT_PATH); print(f"预测对比图已保存到: {PREDICTION_PLOT_PATH}")
                else: print("没有足够的唯一符号来绘制对比图。")    
            else: print("预测结果与测试信息长度不匹配，无法绘制符号对比图。")
        else: print("测试集数据或预测为空，无法绘制预测对比图。")

    print(f"--- 主程序 RNN_TEST.py 结束 ---")

if __name__ == '__main__':
    main()