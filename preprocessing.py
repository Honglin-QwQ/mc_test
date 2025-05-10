import pandas as pd
import pyarrow.feather as pf
import numpy as np
import os
import traceback

# --- 配置参数 ---
KNOWN_SYMBOLS = ["BNB/USDT", "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
PREDEFINED_SYMBOL_TO_INT_MAPPING = {symbol: i for i, symbol in enumerate(KNOWN_SYMBOLS)}
# NEXT_AVAILABLE_CODE_START = len(KNOWN_SYMBOLS) # calculate_columnar_stats 会动态处理

# --- 辅助函数 ---
def get_symbol_encoding_map_report_from_series(symbol_series_pandas):
    """
    根据 Pandas Series 生成 symbol 编码的详细报告字典。
    这个函数主要用于生成报告，实际编码在 calculate_columnar_stats 中处理。
    """
    final_symbol_map_for_report = {}
    # 先加入已知的 symbols 及其预定义编码
    for symbol, code in PREDEFINED_SYMBOL_TO_INT_MAPPING.items():
        final_symbol_map_for_report[symbol] = code
    
    if not symbol_series_pandas.empty:
        unique_symbols_in_data = symbol_series_pandas.unique()
        for symbol_val in unique_symbols_in_data:
            if pd.isna(symbol_val):
                # 在报告中可以标记NaN，实际编码由主映射决定
                final_symbol_map_for_report[f"NaN_symbol_value_in_data (actual mapping depends on main logic)"] = "See __Schema_for_Future_Unknown_and_NaN_Symbol__"
            elif symbol_val not in PREDEFINED_SYMBOL_TO_INT_MAPPING:
                final_symbol_map_for_report[f"{symbol_val} (Unseen in PREDEFINED, will get new code)"] = "See main map"
            
    # __Schema_for_Future_Unknown_and_NaN_Symbol__ 会在主映射中添加
    return final_symbol_map_for_report

def _encode_series_with_learned_map(s_series, learned_map, fallback_code):
    """使用学习到的映射和回退码来编码序列"""
    def get_code(x):
        if pd.isna(x):
            return fallback_code
        return learned_map.get(str(x), fallback_code) # 确保x是字符串进行查找
    return s_series.map(get_code)

def get_datetime_features_stats_and_range(feather_file_path, stats_dict, all_column_names_from_file):
    """
    只读取 dt 列，拆分并计算年、月、日、小时的统计量，并返回该列的最小和最大日期时间。
    将统计结果更新到传入的 stats_dict。
    返回: (min_datetime, max_datetime)
    """
    min_dt, max_dt = pd.NaT, pd.NaT
    if 'dt' not in all_column_names_from_file:
        print("警告：(辅助函数) 主数据文件中未找到 'dt' 列。")
        for col_name_dt_err in ['year', 'month', 'day', 'hour']:
            stats_dict[col_name_dt_err] = {'mean': np.nan, 'std_dev': np.nan}
        return min_dt, max_dt 

    # print("  (辅助函数) 处理 'dt' 列以计算日期时间特征统计量和范围...") #减少打印
    try:
        arrow_table_dt = pf.read_table(feather_file_path, columns=['dt'])
        # 确保 dt_series_str 是 Series
        dt_column_data = arrow_table_dt['dt'].to_pandas(strings_to_categorical=False, zero_copy_only=False, self_destruct=True)
        del arrow_table_dt
        
        if isinstance(dt_column_data, pd.DataFrame): # 如果返回的是DataFrame，取第一列
            if not dt_column_data.empty:
                dt_series_str = dt_column_data.iloc[:, 0]
            else:
                dt_series_str = pd.Series(dtype='object')
        else: # 已经是Series
            dt_series_str = dt_column_data

        datetime_series = pd.to_datetime(dt_series_str, errors='coerce')
        
        datetime_series_cleaned_for_range = datetime_series.dropna()
        if not datetime_series_cleaned_for_range.empty:
            min_dt = datetime_series_cleaned_for_range.min()
            max_dt = datetime_series_cleaned_for_range.max()
        
        dt_derived_features_data = {
            'year': datetime_series.dt.year, 'month': datetime_series.dt.month,
            'day': datetime_series.dt.day, 'hour': datetime_series.dt.hour
        }
        for col_name, series_data in dt_derived_features_data.items():
            series_data_cleaned_for_stats = series_data.dropna().astype(float) # dropna后确保是float
            if not series_data_cleaned_for_stats.empty:
                stats_dict[col_name] = {'mean': np.nanmean(series_data_cleaned_for_stats), 'std_dev': np.nanstd(series_data_cleaned_for_stats)}
            else:
                stats_dict[col_name] = {'mean': np.nan, 'std_dev': np.nan}
        # print("  (辅助函数) 'dt' 列派生特征统计量和范围计算完毕。")
    except Exception as e:
        print(f"  (辅助函数) 处理 'dt' 列时发生错误: {e}")
        for col_name_dt_err in ['year', 'month', 'day', 'hour']:
            stats_dict[col_name_dt_err] = {'mean': np.nan, 'std_dev': np.nan}
    return min_dt, max_dt

def get_single_column_stats(feather_file_path, column_name, stats_dict, all_column_names_from_file):
    if column_name not in all_column_names_from_file:
        stats_dict[column_name] = {'mean': np.nan, 'std_dev': np.nan} # 确保有占位符
        print(f"警告：(辅助函数) 列 '{column_name}' 在元数据中未找到，跳过统计。")
        return
    # print(f"    (辅助函数) 处理列 '{column_name}'...") # 减少打印
    try:
        arrow_table_col = pf.read_table(feather_file_path, columns=[column_name])
        series_data_raw = arrow_table_col[column_name].to_pandas(strings_to_categorical=False, zero_copy_only=False, self_destruct=True)
        del arrow_table_col

        if isinstance(series_data_raw, pd.DataFrame): #如果返回的是DataFrame，取第一列
            if not series_data_raw.empty:
                series_data = series_data_raw.iloc[:,0]
            else:
                series_data = pd.Series(dtype='object')
        else:
            series_data = series_data_raw
            
        series_data_cleaned = series_data.dropna()

        if not series_data_cleaned.empty and pd.api.types.is_numeric_dtype(series_data_cleaned):
            stats_dict[column_name] = {'mean': np.nanmean(series_data_cleaned.astype(float)), 
                                       'std_dev': np.nanstd(series_data_cleaned.astype(float))}
        elif series_data_cleaned.empty: # 清理后为空
            stats_dict[column_name] = {'mean': np.nan, 'std_dev': np.nan}
            # print(f"  (辅助函数) 警告：列 '{column_name}' 清理后为空或全是NaN。")
        else: # 非数值
            stats_dict[column_name] = {'mean': 'N/A (non-numeric)', 'std_dev': 'N/A (non-numeric)'}
            # print(f"  (辅助函数) 警告：列 '{column_name}' 不是数值类型。")
    except Exception as e:
        print(f"  (辅助函数) 处理列 '{column_name}' 时发生错误: {e}")
        stats_dict[column_name] = {'mean': np.nan, 'std_dev': np.nan}


def calculate_columnar_stats(feather_file_path, f_prefix='F#'):
    normalization_stats_data = {} 
    final_learned_symbol_map = {} 
    min_dt_overall, max_dt_overall = pd.NaT, pd.NaT
    num_rows_original, num_cols_original = 0, 0
    all_column_names = []

    try:
        print(f"  (calculate_columnar_stats) 开始从 '{feather_file_path}' 计算统计量并获取元数据...")
        # 使用 try-except 块来确保即使这里出错，函数也能返回预期的4个值
        try:
            table_for_metadata = pf.read_table(feather_file_path, memory_map=True)
            all_column_names = table_for_metadata.schema.names
            num_rows_original = table_for_metadata.num_rows
            num_cols_original = table_for_metadata.num_columns
            del table_for_metadata 
            print(f"  (calculate_columnar_stats) 文件元数据: {num_rows_original} 行, {num_cols_original} 列。")
        except Exception as e_meta:
            print(f"  (calculate_columnar_stats) 错误：获取文件元数据失败: {e_meta}")
            # 即使元数据获取失败，仍然尝试返回正确的结构，让上层函数处理错误
            return pd.DataFrame(), {"__Error__": f"Failed to read metadata: {e_meta}"}, pd.NaT, pd.NaT

        if num_rows_original == 0: 
            print("  (calculate_columnar_stats) 错误：文件为空。")
            return pd.DataFrame(), {"__Error__": "Source Feather file is empty."}, pd.NaT, pd.NaT
        
        # --- Symbol处理 ---
        if 'symbol' in all_column_names:
            print("  (calculate_columnar_stats) 处理 'symbol' 列...")
            arrow_table_symbol = pf.read_table(feather_file_path, columns=['symbol'])
            symbol_series_pandas_raw = arrow_table_symbol['symbol'].to_pandas(strings_to_categorical=False, zero_copy_only=False, self_destruct=True)
            del arrow_table_symbol

            if isinstance(symbol_series_pandas_raw, pd.DataFrame):
                symbol_series_pandas = symbol_series_pandas_raw.iloc[:,0] if not symbol_series_pandas_raw.empty else pd.Series(dtype='object')
            else:
                symbol_series_pandas = symbol_series_pandas_raw

            current_max_code = -1
            for symbol, code in PREDEFINED_SYMBOL_TO_INT_MAPPING.items():
                final_learned_symbol_map[symbol] = code
                current_max_code = max(current_max_code, code)
            next_available_code_for_new = current_max_code + 1
            if not symbol_series_pandas.empty:
                unique_symbols_in_data = symbol_series_pandas.unique()
                for symbol_val in unique_symbols_in_data:
                    if pd.notna(symbol_val) and str(symbol_val) not in final_learned_symbol_map: # 确保用 str 比较
                        final_learned_symbol_map[str(symbol_val)] = next_available_code_for_new
                        next_available_code_for_new += 1
            future_unknown_and_nan_code = next_available_code_for_new
            final_learned_symbol_map["__Schema_for_Future_Unknown_and_NaN_Symbol__"] = future_unknown_and_nan_code
            if symbol_series_pandas.isna().any():
                 final_learned_symbol_map["__Note_NaN_Symbols_In_Training_Data_Mapped_To__"] = future_unknown_and_nan_code
            # print(f"  (calculate_columnar_stats) Symbol 映射已动态生成。") # 减少打印

            symbol_encoded_series = _encode_series_with_learned_map(
                symbol_series_pandas, final_learned_symbol_map, future_unknown_and_nan_code
            )
            if not symbol_encoded_series.empty:
                normalization_stats_data['symbol_encoded'] = {
                    'mean': np.nanmean(symbol_encoded_series.astype(float)),
                    'std_dev': np.nanstd(symbol_encoded_series.astype(float))
                }
            else:
                normalization_stats_data['symbol_encoded'] = {'mean': np.nan, 'std_dev': np.nan}
            del symbol_series_pandas, symbol_encoded_series
        else: 
            print("  (calculate_columnar_stats) 警告：主数据文件中未找到 'symbol' 列。")
            normalization_stats_data['symbol_encoded'] = {'mean': np.nan, 'std_dev': np.nan}
            final_learned_symbol_map = {key: val for key, val in PREDEFINED_SYMBOL_TO_INT_MAPPING.items()}
            final_learned_symbol_map["__Schema_for_Future_Unknown_and_NaN_Symbol__"] = len(PREDEFINED_SYMBOL_TO_INT_MAPPING)
            final_learned_symbol_map["__Error__"] = "Symbol column not found in main data file"

        # --- DateTime处理 ---
        min_dt_overall, max_dt_overall = get_datetime_features_stats_and_range(
            feather_file_path, normalization_stats_data, all_column_names
        )

        # --- F#列处理 ---
        f_cols_to_process = [col for col in all_column_names if col.startswith(f_prefix)]
        if f_cols_to_process:
            print(f"  (calculate_columnar_stats) 开始处理 {len(f_cols_to_process)} 个 F# 开头的列...")
            for col_name_f in f_cols_to_process:
                get_single_column_stats(feather_file_path, col_name_f, normalization_stats_data, all_column_names)
            # print("  (calculate_columnar_stats) F# 开头的列统计量计算完毕。") # 减少打印
        else:
            print("  (calculate_columnar_stats) 警告：未找到 F# 开头的列。")

        # --- target_1列处理 ---
        if 'target_1' in all_column_names:
            # print("  (calculate_columnar_stats) 开始处理 'target_1' 列...") # 减少打印
            get_single_column_stats(feather_file_path, 'target_1', normalization_stats_data, all_column_names)
        else:
            print("  (calculate_columnar_stats) 警告：主数据文件中未找到 'target_1' 列。")
            normalization_stats_data['target_1'] = {'mean': np.nan, 'std_dev': np.nan}
        
        stats_df = pd.DataFrame(normalization_stats_data) if normalization_stats_data else pd.DataFrame()
        # print("  (calculate_columnar_stats) 所有统计量计算完成。") # 减少打印
        return stats_df, final_learned_symbol_map, min_dt_overall, max_dt_overall

    except FileNotFoundError:
        print(f"  (calculate_columnar_stats) 错误：文件 '{feather_file_path}' 未找到。")
        return pd.DataFrame(), {"__Error__": "File not found"}, pd.NaT, pd.NaT
    except ImportError: 
        print("  (calculate_columnar_stats) 错误: pyarrow 库未找到。")
        return pd.DataFrame(), {"__Error__": "pyarrow not installed"}, pd.NaT, pd.NaT
    except Exception as e:
        print(f"  (calculate_columnar_stats) 计算统计量过程中发生严重错误：{e}")
        traceback.print_exc()
        return pd.DataFrame(), {"__Error__": str(e)}, pd.NaT, pd.NaT # 确保返回4个值


def get_or_create_preprocessing_outputs(feather_path, stats_csv_path, symbol_csv_path, f_prefix='F#', verbose=True):
    print(f"--- (preprocessing.py) 开始执行 get_or_create_preprocessing_outputs ---")
    normalization_table = pd.DataFrame()
    symbol_report = {} 
    min_overall_dt, max_overall_dt = pd.NaT, pd.NaT
    original_feather_rows = 0 # 初始化
    all_feather_column_names = [] # 初始化

    if not os.path.exists(feather_path):
        # ... (错误处理) ...
        return normalization_table, {"__Error__": f"Source Feather file '{feather_path}' not found."}, min_overall_dt, max_overall_dt, original_feather_rows, all_feather_column_names

    try:
        # 步骤 0.1: 获取所有列名和原始行数 (使用推荐的 read_table().schema)
        print(f"  (preprocessing.py) 尝试通过 pf.read_table().schema 获取元数据: {feather_path}")
        table_for_metadata = pf.read_table(feather_path, memory_map=True)
        all_feather_column_names = table_for_metadata.schema.names
        original_feather_rows = table_for_metadata.num_rows 
        del table_for_metadata
        if verbose: print(f"  (preprocessing.py) 成功获取文件元数据: 共 {len(all_feather_column_names)} 列, {original_feather_rows} 行。")
        
        if original_feather_rows == 0 and len(all_feather_column_names) > 0 : # 有列名但行数为0
             print(f"  (preprocessing.py) 警告: 原始 Feather 文件 '{feather_path}' 为空 (0行数据)。")
             # 即使文件为空，也返回获取到的信息，主程序可以决定如何处理
        elif original_feather_rows == 0 and not all_feather_column_names: # 列名也获取不到
             error_msg = "无法获取列名且文件报告0行，可能文件无效或为空。"
             print(f"  (preprocessing.py) 错误: {error_msg}")
             if verbose: print_verbose_reports(None, {"__Error__": error_msg}, pd.NaT, pd.NaT, 0, feather_path, "File Empty/Invalid Check")
             return pd.DataFrame(), {"__Error__": error_msg}, pd.NaT, pd.NaT, 0, []


    except Exception as e_meta:
        error_msg = f"获取 Feather 文件元数据时出错: {e_meta}"
        print(f"  (preprocessing.py) {error_msg}")
        if verbose: print_verbose_reports(None, {"__Error__": error_msg}, pd.NaT, pd.NaT, 0, feather_path, "Metadata Read Error")
        return pd.DataFrame(), {"__Error__": error_msg}, min_overall_dt, max_overall_dt, 0, [] # 返回0和空列表
    
    # 步骤 0.2: 获取全局时间范围 (现在 all_feather_column_names 已定义)
    # get_datetime_features_stats_and_range 会读取 'dt' 列
    temp_stats_for_dt_range = {}
    min_overall_dt, max_overall_dt = get_datetime_features_stats_and_range(
        feather_path, temp_stats_for_dt_range, all_feather_column_names
    )
    if pd.isna(min_overall_dt) or pd.isna(max_overall_dt):
         if verbose and 'dt' in all_feather_column_names:
             print(f"  (preprocessing.py) 警告: 从'{feather_path}'的'dt'列未能确定有效时间范围。")
    elif verbose:
        print(f"  (preprocessing.py) 原始数据时间范围: {min_overall_dt} 至 {max_overall_dt}")


    # 步骤 1: 尝试加载已存在的统计文件
    files_exist = os.path.exists(stats_csv_path) and os.path.exists(symbol_csv_path)
    attempt_load_from_file = files_exist
    
    if attempt_load_from_file:
        if verbose: print(f"  (preprocessing.py) 发现已存在的统计文件。正在加载...")
        try:
            normalization_table = pd.read_csv(stats_csv_path, index_col=0)
            symbol_report_df = pd.read_csv(symbol_csv_path)
            symbol_report = dict(zip(symbol_report_df['Symbol_Or_SchemaKey'], symbol_report_df['Encoded_Value']))
            if verbose: print("  (preprocessing.py) 统计文件加载成功。")
            if normalization_table.empty or not symbol_report:
                if verbose: print("  (preprocessing.py) 警告：加载的统计文件内容为空，将重新生成。")
                attempt_load_from_file = False 
        except Exception as e:
            if verbose: print(f"  (preprocessing.py) 加载已存在的统计文件时发生错误: {e}。将重新预处理。")
            attempt_load_from_file = False 
            normalization_table = pd.DataFrame() 
            symbol_report = {}
    
    # 步骤 2: 如果需要，执行预处理计算
    if not attempt_load_from_file: 
        if verbose: print("  (preprocessing.py) 执行预处理步骤以生成/重新生成统计文件...")
        
        # calculate_columnar_stats 返回: stats_df, symbol_map, min_dt, max_dt
        temp_norm_table, temp_symbol_report, calc_min_dt, calc_max_dt = calculate_columnar_stats(feather_path, f_prefix)
        
        error_in_calc_stats = temp_symbol_report.get("__Error__") or \
                              (temp_norm_table.empty and not (temp_symbol_report and temp_symbol_report.get("__Error__")) and \
                               temp_symbol_report.get("__Info__") != "Source Feather file is empty.")
        
        if error_in_calc_stats:
            if verbose: print("  (preprocessing.py) 预处理步骤未能成功生成有效数据。")
            symbol_report = temp_symbol_report if isinstance(temp_symbol_report, dict) else {"__Error__": "Preprocessing failed (symbol_report format error)"}
            normalization_table = temp_norm_table if isinstance(temp_norm_table, pd.DataFrame) else pd.DataFrame()
        else:
            normalization_table = temp_norm_table
            symbol_report = temp_symbol_report
            if verbose: print("  (preprocessing.py) 预处理数据生成成功。")
        
        # 使用在 calculate_columnar_stats 中计算得到的 min/max dt，因为那是基于完整数据处理的
        min_overall_dt, max_overall_dt = calc_min_dt, calc_max_dt 
            
        if not normalization_table.empty:
            try:
                normalization_table.to_csv(stats_csv_path, index=True)
                if verbose: print(f"  (preprocessing.py) 统计量表已保存到: {stats_csv_path}")
            except Exception as e_save_norm:
                if verbose: print(f"  (preprocessing.py) 保存统计量表到 CSV 时发生错误: {e_save_norm}")
        
        if symbol_report and "__Error__" not in symbol_report:
            try:
                symbol_report_df = pd.DataFrame(list(symbol_report.items()), columns=['Symbol_Or_SchemaKey', 'Encoded_Value'])
                symbol_report_df.to_csv(symbol_csv_path, index=False)
                if verbose: print(f"  (preprocessing.py) Symbol 编码映射报告已成功保存到: {symbol_csv_path}")
            except Exception as e_save_sym:
                if verbose: print(f"  (preprocessing.py) 保存 Symbol 映射报告到 CSV 时发生错误: {e_save_sym}")

    if verbose:
        # print_verbose_reports 现在需要 all_feather_column_names (虽然它内部没直接用，但可以调整)
        # 但它主要用 original_feather_rows
        print_verbose_reports(normalization_table, symbol_report, min_overall_dt, max_overall_dt, original_feather_rows, feather_path, "Final report")
        
    print(f"--- (preprocessing.py) 结束执行 get_or_create_preprocessing_outputs ---")
    # 返回这6个值
    return normalization_table, symbol_report, min_overall_dt, max_overall_dt, original_feather_rows, all_feather_column_names


def print_verbose_reports(norm_table, sym_report, min_dt, max_dt, feather_rows, feather_path, context_msg=""): # 定义与调用一致
    """Helper function to print reports if verbose."""
    print(f"\n\n--- (preprocessing.py) {context_msg}: 归一化/标准化所需统计量表 ---")
    if norm_table is not None and not norm_table.empty:
        print(norm_table)
    else:
        print("未能生成或加载统计量表。")

    print(f"\n\n--- (preprocessing.py) {context_msg}: Symbol 编码映射报告 ---")
    if sym_report and isinstance(sym_report, dict) and "__Error__" not in sym_report:
        for symbol_key, code_val in sym_report.items():
            print(f"  '{symbol_key}': {code_val}")
        unknown_val_from_report = sym_report.get("__Schema_for_Future_Unknown_and_NaN_Symbol__", "未定义")
        print(f"\n  (注意: 不在预定义列表 {KNOWN_SYMBOLS} 中的新 symbol，")
        print(f"   或原始数据中的 NaN symbol, 其编码会被处理为 {unknown_val_from_report}，具体见映射报告)")
    elif sym_report and isinstance(sym_report, dict) and "__Error__" in sym_report:
        print(f"未能生成或加载 Symbol 映射报告: {sym_report['__Error__']}")
    else:
        print("未能生成或加载 Symbol 映射报告（可能是空或格式不正确）。")
    
    print(f"\n--- (preprocessing.py) {context_msg}: 关于目标函数 (target_1) 的说明 ---")
    target_mean_valid = False
    if norm_table is not None and not norm_table.empty and 'target_1' in norm_table.columns:
        if 'mean' in norm_table.index and not pd.isna(norm_table.loc['mean', 'target_1']):
            target_mean_valid = True
    if target_mean_valid: 
        print("目标函数 'target_1' 的均值和标准差已包含在统计量表中。")
        print("对于 RNN 回归任务，通常建议对目标变量也进行 Z-score 标准化。")
        print("如果进行了标准化，模型预测的结果将是标准化后的值，需要使用此表中的均值和标准差进行反向转换，以得到原始尺度的预测值。")
    else: 
        print("目标函数 'target_1' 的统计信息未在表中找到/生成或值无效。")

    print(f"\n\n--- (preprocessing.py) {context_msg}: 原始 Feather 文件信息 ---")
    if os.path.exists(feather_path):
        if feather_rows > 0: 
            print(f"原始 Feather 文件 ('{feather_path}') 包含数据行数: {feather_rows}")
        else: 
            print(f"原始 Feather 文件 ('{feather_path}') 可能为空或未能读取行数 (报告行数: {feather_rows})。")
        
        if pd.notna(min_dt) and pd.notna(max_dt):
            print(f"  'dt' 列时间范围： {min_dt} 至 {max_dt}")
        else:
            print(f"  未能从 'dt' 列确定有效的时间范围。")
    else: 
        print(f"原始 Feather 文件 ('{feather_path}') 未找到。")