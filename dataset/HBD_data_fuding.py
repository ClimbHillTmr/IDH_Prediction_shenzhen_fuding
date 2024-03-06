import pandas as pd
import numpy as np

from data_process import (
    calculate_column,
    calculate_time_points,
    convert_to_float_list,
    replace_extremes_with_percentiles,
    diagnose_hypertension,
)

dataset = pd.read_csv("Dataset/updated_dataset_fuding.csv")

print(dataset.info())


def fuding_dataset(first_pressure_sd=None):
    # Read the dataset
    dataset = pd.read_csv("Dataset/updated_dataset_fuding.csv")

    dataset["透前动脉压"] = 1 / 3 * dataset["透前收缩压"] + 2 / 3 * dataset["透前舒张压"]
    dataset["透后动脉压"] = 1 / 3 * dataset["透后收缩压"] + 2 / 3 * dataset["透后舒张压"]

    # Data filtering
    dataset = dataset[dataset["透中血压"].notnull() & (dataset["透中血压"] != "")]
    # dataset = dataset[(dataset["干体重"] > 0) & (dataset["实际透析时长"] > 0)]

    # 将逗号分隔的血压值拆分为列表
    dataset["透中血压"] = dataset["透中血压"].str.split(",")

    # 初始化两个空列表，一个用于透中舒张压，另一个用于透中收缩压
    diastolic_list, systolic_list = [], []

    # 遍历每行数据
    for index, row in dataset.iterrows():
        diastolic_row, systolic_row = [], []
        for bp in row["透中血压"]:
            if "/" in bp:
                diastolic, systolic = bp.split("/")
                if diastolic and systolic:
                    diastolic_row.append(int(diastolic))
                    systolic_row.append(int(systolic))
                else:
                    diastolic_row.append(None)
                    systolic_row.append(None)
                    print(bp)
                    print(index)
                    print(row["透中血压"])
            else:
                diastolic_row.append(None)
                systolic_row.append(None)
                print(bp)
                print(index)
                print(row["透中血压"])

        diastolic_list.append(diastolic_row)
        systolic_list.append(systolic_row)

    # 将结果添加为两列
    dataset["透析中收缩压"] = diastolic_list
    dataset["透析中舒张压"] = systolic_list
    # 删除原始透中血压列
    dataset.drop(columns=["透中血压"], inplace=True)

    # Convert specified date columns to datetime format
    date_columns = [
        "出生日期",
        "首次透析日期",
        # '本院首次透析日期',
        "终止日期",
        "透析日期",
        "瘘管置管时间",
    ]
    dataset[date_columns] = dataset[date_columns].apply(pd.to_datetime, errors="coerce")

    # 计算动脉压并添加到 DataFrame 中
    dataset["动脉压"] = dataset.apply(
        lambda row: [
            (1 / 3) * sbp + (2 / 3) * dbp
            for sbp, dbp in zip(row["透析中收缩压"], row["透析中舒张压"])
        ],
        axis=1,
    )
    dataset = dataset[dataset["动脉压"].notna()]
    # dataset = dataset[
    #     (dataset["干体重"] > 0) & dataset["动脉压"].notna() & (dataset["实际透析时长"] > 0)
    # ]

    # # 处理列表数据
    list_columns = [
        "透析中收缩压",
        "透析中舒张压",
        "透析中脉搏",
        # "平均动脉压",
        "超滤率",
        # "超滤量",
        "静脉压",
        "动脉压",
        "血流速",
        "透析液温度",
        "跨膜压",
    ]

    for col in list_columns:
        # print(col)
        # dataset[col] = dataset[col].str.strip("[]").str.split(",")
        # print(dataset[col])
        dataset[col] = dataset[col].apply(convert_to_float_list)
        # print(dataset[col])

    dataset["超滤量MAX"] = dataset["超滤量"] * 1000

    dataset["透析中收缩压"] = dataset["透析中收缩压"].apply(lambda lst: lst if lst else [])
    dataset = dataset[dataset["透析中收缩压"].apply(len) > 0]

    # 使用函数替换极值
    dataset = replace_extremes_with_percentiles(dataset, list_columns)

    # Remove rows with empty lists
    dataset = dataset[dataset["透析中收缩压"].apply(lambda lst: lst and len(lst) > 0)]
    dataset["透中数据记录时间节点"] = dataset["透中数据记录时间节点"].str.strip("[]").str.split(",")
    dataset = dataset[dataset["动脉压"].apply(lambda lst: len(lst) > 0)]

    # Calculate derived features
    dataset["透析年龄"] = (
        pd.to_datetime(dataset["透析日期"]).dt.year
        - pd.to_datetime(dataset["出生日期"]).dt.year
    )
    dataset["首次透析年龄"] = (
        pd.to_datetime(dataset["首次透析日期"]).dt.year
        - pd.to_datetime(dataset["出生日期"]).dt.year
    )
    dataset["瘘管使用时间"] = (
        pd.to_datetime(dataset["透析日期"]).dt.year
        - pd.to_datetime(dataset["瘘管置管时间"]).dt.year
    )
    dataset["高血压诊断"] = dataset["诊断"].apply(diagnose_hypertension)
    # 透析龄
    dataset["透析龄_天数"] = (dataset["透析日期"] - dataset["首次透析日期"]).dt.days
    dataset["透析龄"] = (dataset["透析日期"] - dataset["首次透析日期"]).dt.days // 365

    # Encode gender using LabelEncoder
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    dataset["性别"] = label_encoder.fit_transform(dataset["性别"])

    # 找到元素长度不一致的行的索引
    rows_to_remove = dataset[
        dataset.apply(lambda row: len(row["动脉压"]) != len(row["透中数据记录时间节点"]), axis=1)
    ].index

    # 从DataFrame中删除对应的行
    dataset.drop(rows_to_remove, inplace=True)

    dataset["透中低血压_计算"] = dataset.apply(
        lambda row: calculate_column(row=row, first_pressure=first_pressure_sd), axis=1
    )
    dataset["透中低血压_计算"].fillna(0, inplace=True)

    # Remove empty strings from '透中数据记录时间节点' column
    dataset["透中数据记录时间节点"] = dataset["透中数据记录时间节点"].apply(
        lambda lst: [item for item in lst if item != ""]
    )

    dataset["降幅时间点"] = dataset.apply(
        lambda row: calculate_time_points(row=row, first_pressure=first_pressure_sd),
        axis=1,
    )

    # Extract '透析开始时间' and '透析结束时间' from '透中数据记录时间节点'
    dataset["透析开始时间"] = dataset["透中数据记录时间节点"].apply(lambda lst: lst[0])
    dataset["透析结束时间"] = dataset["透中数据记录时间节点"].apply(lambda lst: lst[-1])

    # Convert time strings to datetime objects
    print(dataset["降幅时间点"])
    dataset["降幅时间点"] = pd.to_datetime(dataset["降幅时间点"])
    dataset["透析开始时间"] = pd.to_datetime(dataset["透析开始时间"])
    dataset["透析结束时间"] = pd.to_datetime(dataset["透析结束时间"])

    # Calculate '降幅时间点区间' and '降幅时间点差值' columns
    time_diff_minutes = (dataset["透析结束时间"] - dataset["透析开始时间"]).dt.total_seconds()
    dataset["降幅时间点比值"] = (
        dataset["降幅时间点"] - dataset["透析开始时间"]
    ).dt.total_seconds() / time_diff_minutes
    dataset["降幅时间点比值区间"] = dataset["降幅时间点比值"].apply(
        lambda x: 0
        if x < 0
        else (1 if x <= 0.25 else (2 if x <= 0.5 else (3 if x <= 0.75 else 4)))
    )
    dataset["降幅时间点差值"] = (
        (dataset["降幅时间点"] - dataset["透析开始时间"]).dt.total_seconds() / 60 / 60
    )
    dataset["降幅时间点差值区间"] = dataset["降幅时间点差值"].apply(
        lambda x: 0
        if x < 0
        else (1 if x <= 1 else (2 if x <= 2 else (3 if x <= 3 else 4)))
    )

    dataset["透前体重-干体重"] = dataset["透前体重"] - dataset["干体重"]

    # dataset.to_csv("Final_data/福鼎_based_data_" + str(first_pressure_sd) + "_HDH.csv")

    # Process mean and standard deviation of list columns
    for col in list_columns:
        mean_col = col + "_mean"
        std_col = col + "_std"
        dataset[mean_col] = dataset[col].apply(
            lambda lst: np.mean(lst) if lst else None
        )
        dataset[std_col] = dataset[col].apply(lambda lst: np.std(lst) if lst else None)

    keep_columns = [
        "患者id",
        "透析记录id",
        "年龄",
        "性别",
        "透析龄",
        "出生日期",
        "传染病",
        # "患者状态",
        "透前收缩压",
        "透前舒张压",
        "透前动脉压",
        "透后动脉压",
        "透后收缩压",
        "透后舒张压",
        "透析龄_天数",
        "透析年龄",
        "透析日期",
        "透前体重-干体重",
        # "透析器",
        "透析方式",
        # "透析前预设UFV",
        "透前体重",
        "透前呼吸频率",
        "透前体温",
        "干体重",
        "透析液钙浓度",
        "透析液电导率",
        # "透析液钠浓度",
        # "透析液钾浓度",
        # "透析液碳酸氢根浓度",
        "抗凝剂类型",
        # "抗凝剂使用总量",
        # "抗凝剂维持量",
        # "抗凝剂追加量",
        "透析后体重",
        # "透后脉搏",
        # "透后体温",
        "实际透析时长",
        "瘘管类型",
        "瘘管位置",
        # '瘘管置管时间',
        "瘘管使用时间",
        "首次透析年龄",
        "高血压诊断",
        "超滤量MAX",
        "透中低血压_计算",
        "降幅时间点",
        "透析开始时间",
        "透析结束时间",
        "降幅时间点比值",
        "降幅时间点比值区间",
        "降幅时间点差值",
        "降幅时间点差值区间",
        "透析中收缩压_mean",
        "透析中收缩压_std",
        "透析中舒张压_mean",
        "透析中舒张压_std",
        "透析中脉搏_mean",
        "透析中脉搏_std",
        # "平均动脉压_mean",
        # "平均动脉压_std",
        "超滤率_mean",
        "超滤率_std",
        # "超滤量_mean",
        # "超滤量_std",
        "静脉压_mean",
        "静脉压_std",
        "动脉压_mean",
        "动脉压_std",
        "血流速_mean",
        "血流速_std",
        "透析液温度_mean",
        "透析液温度_std",
        "跨膜压_mean",
        "跨膜压_std",
    ]
    # fuding = ["透中透中指脉氧", "透析液电导率"]
    # # Save the resulting DataFrame
    # dataset = dataset[keep_columns]
    # dataset.to_csv(
    #     "Final_data/福鼎_optimized_data_" + str(first_pressure_sd) + "_HDH.csv"
    # )
    # # del X['降幅时间点区间']
    # # del X['透中低血压_计算']

    mean_columns = [
        # "患者状态",
        # "透析前预设UFV",
        "透前体重",
        "透前呼吸频率",
        "透前体温",
        "干体重",
        "透析液钙浓度",
        "透析液电导率",
        # "透析液钠浓度",
        # "透析液钾浓度",
        # "透析液碳酸氢根浓度",
        "实际透析时长",
        "透前收缩压",
        "透前舒张压",
        "降幅时间点比值区间",
        "降幅时间点差值区间",
        "透中低血压_计算",
        "透前体重-干体重",
        "透析中收缩压_mean",
        "透析中舒张压_mean",
        "透析中脉搏_mean",
        # "平均动脉压_mean",
        "超滤率_mean",
        # "超滤量_mean",
        "静脉压_mean",
        "动脉压_mean",
        "血流速_mean",
        "透析液温度_mean",
        "跨膜压_mean",
        "透前动脉压",
        "透后动脉压",
        "超滤量MAX",
    ]
    dataset = dataset.sort_values(by=["患者id", "透析日期"])

    # 创建一个新的DataFrame来存储结果
    result_df = pd.DataFrame()

    # 遍历mean_columns中的每一列
    for col in mean_columns:
        # 创建空列表来存储历史平均指标、透析日期和患者ID
        historical_avg_values = []
        dialysis_dates = []
        patient_ids = []

        # 遍历DataFrame中的每一行
        for index, row in dataset.iterrows():
            # 获取当前行的患者id和透析日期
            patient_id = row["患者id"]
            dialysis_date = row["透析日期"]

            # 在透析日期之前，筛选出同一患者的历史记录
            historical_records = dataset[
                (dataset["患者id"] == patient_id) & (dataset["透析日期"] < dialysis_date)
            ]

            # 计算历史记录中治疗指标的平均值
            if historical_records.shape[0] > 0:
                avg_value = historical_records[col].mean()
            else:
                avg_value = 0

            # 将患者id、透析日期和平均值添加到列表中
            patient_ids.append(patient_id)
            dialysis_dates.append(dialysis_date)
            historical_avg_values.append(avg_value)

        # 将患者id、透析日期和历史平均值列表添加到结果DataFrame中
        result_df["患者id"] = patient_ids
        result_df["透析日期"] = dialysis_dates
        result_df["历史平均" + col] = historical_avg_values

    print(result_df)
    # result_df.to_csv(
    #     path_or_buf=("Final_data/福鼎_result_df " + str(first_pressure_sd) + "_HDH.csv")
    # )

    whole_data = pd.merge(
        dataset.reset_index(), result_df, on=["患者id", "透析日期"], how="inner"
    )

    # whole_data = pd.concat([dataset.reset_index(), result_df], axis=1)

    # for col in mean_columns:
    #     X['历史平均' + col] = X.groupby('患者id')[col].cumsum() / (
    #         X.groupby('患者id').cumcount() + 1
    #     )

    # Y_rate = pd.DataFrame(
    #     columns=[
    #         'history_HBP',
    #         'history_LBP_times_0',
    #         'history_LBP_times_1',
    #         'history_LBP_times_2',
    #         'history_LBP_times_3',
    #         'history_LBP_times_4',
    #     ]
    # )

    Y_rate = []

    # 计算平均值
    # 计算历史比率
    history_HBP = 0
    history_LBP_times_0 = 0
    history_LBP_times_1 = 0
    history_LBP_times_2 = 0
    history_LBP_times_3 = 0
    history_LBP_times_4 = 0
    total_times = 1
    last_p = ""
    dataset = dataset.sort_values(by=["患者id", "透析日期"])

    for i in range(len(dataset)):
        # for i in range(0, 1000):
        if i % 1000 == 0:
            print("加载数据", f"{str(round(i / len(dataset) * 100, 10))}%")

        p = dataset["患者id"].iloc[i]

        if last_p == p:
            history_HBP_rate_rate = history_HBP / total_times
            history_HBP_time_rate_0_rate = history_LBP_times_0 / total_times
            history_HBP_time_rate_1_rate = history_LBP_times_1 / total_times
            history_HBP_time_rate_2_rate = history_LBP_times_2 / total_times
            history_HBP_time_rate_3_rate = history_LBP_times_3 / total_times
            history_HBP_time_rate_4_rate = history_LBP_times_4 / total_times

            y_rate = [
                history_HBP_rate_rate,
                history_HBP_time_rate_0_rate,
                history_HBP_time_rate_1_rate,
                history_HBP_time_rate_2_rate,
                history_HBP_time_rate_3_rate,
                history_HBP_time_rate_4_rate,
            ]

            if dataset["透中低血压_计算"].iloc[i] == 0:
                # history_HBP = 0
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 4:
                history_LBP_times_4 += 1
            total_times += 1
        else:
            last_p = p
            total_times = 1

            history_HBP = 0
            history_LBP_times_0 = 0
            history_LBP_times_1 = 0
            history_LBP_times_2 = 0
            history_LBP_times_3 = 0
            history_LBP_times_4 = 0

            if dataset["透中低血压_计算"].iloc[i] == 0:
                # history_HBP = 0
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 4:
                history_LBP_times_4 += 1

            y_rate = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            # continue
            # y_rate = [
            #     0,
            #     0,
            #     0,
            #     0,
            #     0,
            #     0,
            # ]
        Y_rate.append(y_rate)

    Y_rate = pd.DataFrame(
        Y_rate,
        columns=[
            "history_HBP_rate",
            "history_LBP_times_0_rate",
            "history_LBP_times_1_rate",
            "history_LBP_times_2_rate",
            "history_LBP_times_3_rate",
            "history_LBP_times_4_rate",
        ],
    )

    final_data = pd.concat([whole_data, Y_rate], axis=1)

    Y_rate = []
    # 计算平均值
    # 计算历史比率
    history_HBP = 0
    history_LBP_times_0 = 0
    history_LBP_times_1 = 0
    history_LBP_times_2 = 0
    history_LBP_times_3 = 0
    history_LBP_times_4 = 0
    total_times = 1
    last_p = ""
    dataset = dataset.sort_values(by=["患者id", "透析日期"])

    for i in range(len(dataset)):
        # for i in range(0, 1000):
        if i % 1000 == 0:
            print("加载数据", f"{str(round(i / len(dataset) * 100, 10))}%")

        p = dataset["患者id"].iloc[i]

        if last_p == p:
            history_HBP_rate = history_HBP / total_times
            history_HBP_time_rate_0 = history_LBP_times_0 / total_times
            history_HBP_time_rate_1 = history_LBP_times_1 / total_times
            history_HBP_time_rate_2 = history_LBP_times_2 / total_times
            history_HBP_time_rate_3 = history_LBP_times_3 / total_times
            history_HBP_time_rate_4 = history_LBP_times_4 / total_times

            y_rate = [
                history_HBP_rate,
                history_HBP_time_rate_0,
                history_HBP_time_rate_1,
                history_HBP_time_rate_2,
                history_HBP_time_rate_3,
                history_HBP_time_rate_4,
            ]

            if dataset["透中低血压_计算"].iloc[i] == 0:
                # history_HBP = 0
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 4:
                history_LBP_times_4 += 1
            total_times += 1
        else:
            last_p = p
            total_times = 1

            history_HBP = 0
            history_LBP_times_0 = 0
            history_LBP_times_1 = 0
            history_LBP_times_2 = 0
            history_LBP_times_3 = 0
            history_LBP_times_4 = 0

            if dataset["透中低血压_计算"].iloc[i] == 0:
                # history_HBP = 0
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 4:
                history_LBP_times_4 += 1

            y_rate = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            # continue
            # y_rate = [
            #     0,
            #     0,
            #     0,
            #     0,
            #     0,
            #     0,
            # ]
        Y_rate.append(y_rate)

    Y_rate = pd.DataFrame(
        Y_rate,
        columns=[
            "history_HBP",
            "history_LBP_times_0",
            "history_LBP_times_1",
            "history_LBP_times_2",
            "history_LBP_times_3",
            "history_LBP_times_4",
        ],
    )

    final_data = pd.concat([final_data, Y_rate], axis=1)

    # final_data = pd.concat([whole_data, Y_rate], axis=1)

    final_data.to_csv(path_or_buf=("../Recognize_data/福鼎_IDH_data.csv"))


# for i in ["透前动脉压"]:
fuding_dataset()
