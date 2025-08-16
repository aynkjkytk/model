# import pandas as pd

# # 读取 Excel 文件
# excel_path = 'data/raw_data.xlsx'  # 文件名
# xlsx = pd.ExcelFile(excel_path)

# # 遍历所有 sheet
# for sheet_name in xlsx.sheet_names:
#     df = pd.read_excel(xlsx, sheet_name=sheet_name)
#     csv_name = f'{sheet_name}.csv'
#     df.to_csv(csv_name, index=False, encoding='utf-8-sig')
#     print(f'✅ 已保存 {csv_name}（{len(df)} 行）')

# import pandas as pd

# # 替换为你的实际 CSV 路径
# csv_path = 'Sheet1.csv'

# # 读取数据
# df = pd.read_csv(csv_path)

# # 检查每一列的缺失值数量和占比
# missing_report = df.isnull().sum().to_frame(name='缺失数量')
# missing_report['总行数'] = len(df)
# missing_report['缺失比例(%)'] = (missing_report['缺失数量'] / missing_report['总行数']) * 100
# missing_report = missing_report.sort_values(by='缺失比例(%)', ascending=False)

# # 打印或保存结果
# print(missing_report)
# missing_report.to_csv("missing_report.csv", index=True, encoding='utf-8-sig')

# import pandas as pd

# # 读取原始数据
# df = pd.read_csv('Sheet1.csv')

# # 要删除的列
# columns_to_drop = [
#     "APACHEII", "糖化血红蛋白_day14", "糖化血红蛋白_day7", "全血剩余碱_day14", "吸氧浓度", "全血剩余碱_day7", "糖化血红蛋白_day4", "全血剩余碱_day4",
#     "肝素结合蛋白_day7", "极低脂肪距离入院天数", "氧分压_day14(mmHg)", "二氧化碳分压_day14", "酸碱度_day14", "高敏肌钙蛋白I_day14",
#     "全血剩余碱", "氧分压_day7(mmHg)", "二氧化碳分压_day7", "酸碱度_day7", "氨基末端B型利钠肽前体_day14", "乳酸_day14", "肌红蛋白定量_day14",
#     "CK-MB质量_day14", "普食距离入院天数", "高敏肌钙蛋白I_day7", "低脂肪距离入院天数", "降钙素原_day14", "胃肠减压", "高敏肌钙蛋白I_day4",
#     "氧分压_day4(mmHg)", "二氧化碳分压_day4", "酸碱度_day4", "前白蛋白d0", "C反应蛋白_day14", "温开水（ml）", "温开水启动入院第几天",
#     "肌酐d0", "糖化血红蛋白", "乳酸_day7", "白细胞计数D0", "葡萄糖_day14", "氨基末端B型利钠肽前体_day7", "D-二聚体定量_day14",
#     "PT_day14", "INR_day14", "APTT_day14", "纤维蛋白降解产物_day14", "Fg_day14", "CK-MB质量_day7", "肌红蛋白定量_day7", "尿素_day14",
#     "尿酸_day14", "肌酐_day14", "前白蛋白_day14", "总胆红素_day14", "丙氨酸氨基转移酶_day14", "白蛋白_day14", "钠_day14", "钙_day14", "钾_day14",
#     "降钙素原_day7", "前白蛋白d2", "磷_day14", "中性粒细胞_day14", "红细胞比容_day14", "血红蛋白_day14", "白细胞计数_day14", "血小板计数_day14",
#     "前白蛋白d3", "高敏肌钙蛋白I", "乳酸_day4", "肌酐d3", "肌酐d2", "肠内营养1500大卡距离入院时间", "肠内营养第1天剂量",
#     "氨基末端B型利钠肽前体_day4", "二氧化碳分压", "氧分压mmHg", "酸碱度", "CK-MB质量_day4", "肌红蛋白定量_day4",
#     "降钙素原_day4", "白细胞计数D2", "少量饮水距离入院天数", "白细胞计数D3", "C反应蛋白_day7", "营养启动时间（入院第几天）",
#     "APTT_day7", "INR_day7", "PT_day7", "Fg_day7", "纤维蛋白降解产物_day7", "D-二聚体定量_day7", "纯碳水距离入院天数", "乳酸(mmol/L)",
#     "葡萄糖_day7", "CTSI", "C反应蛋白_day4", "前白蛋白_day7", "尿酸_day7", "尿素_day7", "肌酐_day7", "INR_day4",
#     "纤维蛋白降解产物_day4", "APTT_day4", "Fg_day4", "PT_day4", "D-二聚体定量_day4", "丙氨酸氨基转移酶_day7", "白蛋白_day7", "总胆红素_day7",
#     "钾_day7", "钠_day7", "钙_day7", "中性粒细胞_day7", "血红蛋白_day7", "红细胞比容_day7", "血小板计数_day7", "白细胞计数_day7",
#     "磷_day7", "C反应蛋白", "葡萄糖_day4", "CK-MB质量", "肌红蛋白", "氨基末端B型利钠肽前体", "前白蛋白d1", "经口饮食距离入院天数",
#     "经口饮食（0 少量饮水，1纯碳水化合物(无脂)、米汤 、 白粥 3 极低脂肪(脂肪≤20g 4 低脂肪(脂肪20 - 50g) 2普食）",
#     "肌酐d1", "降钙素原", "尿酸_day4", "尿素_day4", "肌酐_day4", "前白蛋白_day4", "白蛋白_day4", "丙氨酸氨基转移酶_day4", "总胆红素_day4",
#     "白细胞计数D1", "钠_day4", "钙_day4", "钾_day4"
# ]

# # 执行删除
# df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# # 保存新文件
# df_cleaned.to_csv('Sheet1(cleaned).csv', index=False, encoding='utf-8-sig')

# print("✅ 已完成列删除，生成 Sheet1(cleaned).csv")

# import pandas as pd

# # 读取上一步处理过的 CSV 文件
# df = pd.read_csv('Sheet1(ultra).csv')

# # 要处理的列
# target_cols = ["主要诊断", "其他诊断1", "其他诊断2", "其他诊断3", "其他诊断4", "其他诊断5"]

# # 遍历填充缺失值
# for col in target_cols:
#     if col in df.columns:
#         df[col] = df[col].fillna("无")

# # 保存覆盖或另存
# df.to_csv('Sheet1(ultra).csv', index=False, encoding='utf-8-sig')

# print("✅ 已处理诊断列的缺失值（替换为“无”）")

# import pandas as pd

# # 读取已清洗的CSV
# df = pd.read_csv("Sheet1(ultra).csv", encoding='utf-8')

# # 保留列（略，和你上面的一致）
# keep_columns = [
#     # 基本信息
#     "年龄", "住院天数", "住院费用", "结局(1死 0活(治愈/好转)) 2未愈 3其他", "身高", "体重", "BMI",
#     "主要诊断", "其他诊断1", "其他诊断2", "其他诊断3", "其他诊断4", "其他诊断5",
#     # 明确术前指标
#     "mMarshall", "SBP mmHg", "DAP mmHg", "MAPmmHg", "呼吸频率/min", "心率",
#     "体温max", "体温min", "白细胞计数", "中性粒细胞计数", "血红蛋白", "血小板计数", "红细胞比容",
#     "丙氨酸氨基转移酶", "前白蛋白", "白蛋白", "总胆红素", "肌酐", "尿素", "尿酸",
#     "钾", "钠", "钙", "磷", "葡萄糖",
#     "PT", "APTT", "Fg", "D-二聚体定量(mg/L)", "纤维蛋白降解产物(mg/L)", "INR",
#     # 既往病史类
#     "心肌梗死（1是0否）", "心衰(1是0否)", "脑血管疾病(1是0否)", "COPD (1是0否)",
#     "消化性溃疡(1是0否)", "慢性肝病(1是0否)", "糖尿病(1是0否)", "高血压(1是0否)",
#     "慢性肾功能不全(1是0否)", "实体肿瘤(1是0否)",
#     # 可保留的术后指标
#     "呼吸机（否 0 是 1）", "血液净化（否 0 是 1）", "血脂分离（1是0否）", "是否使用抗生素（1是0否）",
#     "入院2周营养类型（0无；1肠内；2肠外）", "腹腔穿刺（1是，0否）",
#     "深静脉（0否 是 1）", "深静脉导管感染（0否 是 1）", "血流感染",
#     "腹腔感染（1是0否）", "腹腔出血（1是0否）", "胰腺坏死（1是0否）",
#     "假性囊肿（1是0否）", "CT诊断胰腺坏死（1是，0否）"
# ]

# # 检查并过滤缺失列
# existing_cols = [col for col in keep_columns if col in df.columns]
# missing_cols = set(keep_columns) - set(existing_cols)
# if missing_cols:
#     print("⚠️ 警告：以下列在原始数据中找不到，将被跳过：")
#     for col in missing_cols:
#         print(f" - {col}")

# # 保存为 UTF-8 带 BOM 编码，确保 Excel 不乱码
# df[existing_cols].to_csv("data.csv", index=False, encoding='utf-8-sig')

# print(f"✅ 已成功生成 data.csv（带 BOM 编码），包含 {len(existing_cols)} 列。")

# import pandas as pd

# # 读取 CSV 文件（带 BOM 编码以兼容 Excel）
# df = pd.read_csv("data.csv", encoding="utf-8-sig")

# # 删除含有任意空值的行
# cleaned_df = df.dropna()

# # 保存为新文件（仍使用带 BOM 的 UTF-8 编码）
# cleaned_df.to_csv("data(cleaned).csv", index=False, encoding="utf-8-sig")

# print(f"✅ 清洗完成，共删除 {len(df) - len(cleaned_df)} 行缺失记录。")

# import pandas as pd

# # 读取清洗后的数据
# df = pd.read_csv("data(cleaned).csv", encoding="utf-8-sig")

# # 诊断相关列
# diag_cols = [
#     "主要诊断",
#     "其他诊断1",
#     "其他诊断2",
#     "其他诊断3",
#     "其他诊断4",
#     "其他诊断5"
# ]

# # 合并这几列为一个 Series，丢弃空值，再统计频次
# all_diagnoses = pd.concat([df[col] for col in diag_cols], ignore_index=True)
# diagnosis_counts = all_diagnoses.dropna().value_counts().reset_index()
# diagnosis_counts.columns = ['诊断名称', '出现次数']

# # 保存为 CSV 文件
# diagnosis_counts.to_csv("diagnosis_stats.csv", index=False, encoding="utf-8-sig")

# print("✅ 已生成 diagnosis_stats.csv，包含所有诊断名称及其出现次数。")

# import pandas as pd

# # 读取诊断统计表
# df = pd.read_csv("diagnosis_stats.csv")

# # 设置编号起始值
# prefix = "D"
# code_counter = 1

# # 结果列表
# mapping = []

# for _, row in df.iterrows():
#     diagnosis = str(row['诊断名称']).strip()
#     count = int(row['出现次数'])

#     if count >= 5:
#         code = f"{prefix}{code_counter:04d}"
#         code_counter += 1
#     else:
#         code = "D9999"  # 表示“其他诊断”

#     mapping.append({
#         '诊断名称': diagnosis,
#         '诊断编码': code
#     })

# # 转换为 DataFrame 并保存
# mapping_df = pd.DataFrame(mapping)
# mapping_df.to_csv("diagnosis_mapping.csv", index=False, encoding='utf-8-sig')

# print(f"✅ 映射表已生成，共编码诊断 {len(mapping_df)} 条（其中高频 {code_counter - 1} 个，低频统一为 D9999）")

# import pandas as pd

# # 读取原始数据和映射表
# data_file = "data(cleaned).csv"
# mapping_file = "diagnosis_mapping.csv"

# df = pd.read_csv(data_file)
# mapping_df = pd.read_csv(mapping_file)

# # 构建映射字典
# mapping_dict = dict(zip(mapping_df['诊断名称'].astype(str).str.strip(), mapping_df['诊断编码']))

# # 需要替换的诊断列
# diagnosis_cols = ["主要诊断", "其他诊断1", "其他诊断2", "其他诊断3", "其他诊断4", "其他诊断5"]

# # 记录未匹配上的项
# unmapped = set()

# # 执行替换
# for col in diagnosis_cols:
#     def map_func(val):
#         val = str(val).strip()
#         if val in mapping_dict:
#             return mapping_dict[val]
#         else:
#             unmapped.add(val)
#             return "D000"  # 用 D000 表示未识别项

#     df[col] = df[col].apply(map_func)

# # 保存新文件
# df.to_csv("data(encoded).csv", index=False, encoding="utf-8-sig")
# print("✅ 已生成 data(encoded).csv，诊断字段已替换为编码")

# # 提示未匹配项
# if unmapped:
#     print("⚠️ 以下诊断未出现在映射表中（已统一替换为 D000）：")
#     for item in sorted(unmapped):
#         print(" -", item)
# else:
#     print("✅ 所有诊断都成功匹配映射")

# import pandas as pd

# # 加载数据
# df = pd.read_csv("data(encoded).csv")

# # 这些列应该全是数字
# numeric_cols = [
#     "mMarshall", "SBP mmHg", "DAP mmHg", "MAPmmHg", "呼吸频率/min", "心率", "体温max", "体温min",
#     "白细胞计数", "中性粒细胞计数", "血红蛋白", "血小板计数", "红细胞比容", "丙氨酸氨基转移酶", "前白蛋白", "白蛋白",
#     "总胆红素", "肌酐", "尿素", "尿酸", "钾", "钠", "钙", "磷", "葡萄糖",
#     "PT", "APTT", "Fg", "D-二聚体定量(mg/L)", "纤维蛋白降解产物(mg/L)", "INR",
#     "心肌梗死（1是0否）", "心衰(1是0否)", "脑血管疾病(1是0否)", "COPD (1是0否)", "消化性溃疡(1是0否)",
#     "慢性肝病(1是0否)", "糖尿病(1是0否)", "高血压(1是0否)", "慢性肾功能不全(1是0否)", "实体肿瘤(1是0否)",
#     "呼吸机（否 0 是 1）", "血液净化（否 0 是 1）", "血脂分离（1是0否）", "是否使用抗生素（1是0否）",
#     "入院2周营养类型（0无；1肠内；2肠外）", "腹腔穿刺（1是，0否）", "深静脉（0否 是 1）", "深静脉导管感染（0否 是 1）",
#     "血流感染", "腹腔感染（1是0否）", "腹腔出血（1是0否）", "胰腺坏死（1是0否）", "假性囊肿（1是0否）", "CT诊断胰腺坏死（1是，0否）"
# ]

# # 强制转换为数值，如果失败则变为 NaN
# df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# # 删除含有 NaN 的行（即非数字原始值）
# cleaned_df = df.dropna(subset=numeric_cols)

# # 保存为干净文件
# cleaned_df.to_csv("data(final).csv", index=False, encoding="utf-8-sig")
# print(f"✅ 清洗完毕，已删除非数字数据行。保留 {cleaned_df.shape[0]} 条记录。")
