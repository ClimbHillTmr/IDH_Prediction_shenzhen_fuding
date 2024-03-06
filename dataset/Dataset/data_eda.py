import pandas as pd


def process_and_print_stats(dataset):
    # 初始数据集行数和[id]列唯一值数量
    initial_rows = len(dataset)
    initial_unique_id_count = dataset["患者id"].nunique()

    # 输出初始数据集信息
    print(f"初始数据集行数: {initial_rows}")
    print(f"[id]列的唯一值数量: {initial_unique_id_count}")


dataset = pd.read_csv(
    "/Users/cht/Documents/GitHub/Dialysis_ML/Dataset/basic_dataset_shenyi.csv"
)

process_and_print_stats(dataset)

dataset = pd.read_csv(
    "/Users/cht/Documents/GitHub/Dialysis_ML/Dataset/updated_dataset_shenyi.csv"
)

process_and_print_stats(dataset)

dataset = pd.read_csv(
    "/Users/cht/Documents/GitHub/Dialysis_ML/透前预测_透中高血压/Final_data/深医_final_data 透前动脉压_HDH.csv"
)

process_and_print_stats(dataset)


dataset = pd.read_csv(
    "/Users/cht/Documents/GitHub/Dialysis_ML/Dataset/basic_dataset_fuding.csv",
    low_memory=False
)

process_and_print_stats(dataset)

dataset = pd.read_csv(
    "/Users/cht/Documents/GitHub/Dialysis_ML/Dataset/updated_dataset_fuding.csv"
)

process_and_print_stats(dataset)

dataset = pd.read_csv(
    "/Users/cht/Documents/GitHub/Dialysis_ML/透前预测_透中高血压/Final_data/福鼎_final_data 透前动脉压_HDH.csv"
)

process_and_print_stats(dataset)
