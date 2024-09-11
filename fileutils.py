import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

variety_to_crop = {
    "BR017-028": "Winter_OSR", "BR017-040": "Winter_OSR", "BR017-055": "Winter_OSR",
    "BR017-090": "Winter_OSR", "BR017-093": "Winter_OSR", "BR017-096": "Winter_OSR",
    "BR017-097": "Winter_OSR", "BR017-098": "Winter_OSR", "BR017-101": "Winter_OSR",
    "BR017-102": "Winter_OSR", "BR017-105": "Winter_OSR", "BR017-106": "Winter_OSR",
    "BR017-107": "Winter_OSR", "BR017-108": "Winter_OSR", "BR017-109": "Winter_OSR",
    "BR017-113": "Winter_OSR", "BR017-133": "Winter_OSR", "BR017-137": "Winter_OSR",
    "BR017-139": "Winter_OSR", "BR017-150": "Winter_OSR", "BR017-160": "Winter_OSR",
    "BR017-168": "Winter_OSR", "BR017-172": "Winter_OSR", "BR017-185": "Winter_fodder",
    "BR017-193": "Winter_fodder", "BR017-203": "Leafy_vegetable", "BR017-206": "Leafy_vegetable",
    "BR017-207": "Leafy_vegetable", "BR017-208": "Leafy_vegetable", "BR017-209": "Leafy_vegetable",
    "BR017-212": "Winter_OSR", "BR017-213": "Winter_OSR", "BR017-218": "Leafy_vegetable",
    "BR017-221": "Spring_fodder", "BR017-229": "Semiwinter_OSR", "BR017-230": "Semiwinter_OSR",
    "BR017-237": "Semiwinter_OSR", "BR017-239": "Spring_OSR", "BR017-240": "Spring_OSR",
    "BR017-257": "Spring_OSR", "BR017-258": "Spring_OSR", "BR017-263": "Spring_OSR",
    "BR017-271": "Spring_fodder", "BR017-273": "Spring_fodder", "BR017-274": "Spring_OSR",
    "BR017-275": "Spring_OSR", "BR017-283": "Spring_OSR", "BR017-410": "Swede",
    "BR017-448": "Swede", "BR017-509": "Winter_OSR", "BR017-510": "Winter_OSR",
    "BR017-511": "Winter_OSR", "BR017-512": "Winter_OSR", "BR017-513": "Semiwinter_OSR",
    "BR017-514": "Winter_OSR", "BR017-515": "Winter_OSR", "BR017-516": "Winter_OSR",
    "BR017-517": "Winter_OSR", "BR017-518": "Winter_OSR", "BR017-520": "Semiwinter_OSR",
    "BR017-521": "Winter_OSR", "BR017-522": "Winter_OSR", "BR017-523": "Winter_OSR",
    "BR017-524": "Winter_OSR", "BR017-526": "Semiwinter_OSR", "BR017-527": "Winter_OSR",
    "BR017-528": "Winter_OSR", "BR017-529": "Semiwinter_OSR", "BR017-530": "Semiwinter_OSR",
    "BR017-531": "Winter_OSR", "BR017-532": "Winter_OSR"
}

@dataclass
class ImageMetadata:
    variety_type: str  # 品种类型编号
    location: int  # 位置 (1,2: 有效，8: 无效)
    treatment: int  # 处理条件 (1: 5摄氏度, 2: 10摄氏度)
    rep: int  # 植物编号
    date: datetime  # 日期
    view_type: str  # 视图类型 (tv: 顶视图, sv: 侧视图)
    angle: str  # 角度 (000, 045, 090)，保留前导零
    crop_type: str  # 从品种类型对应得到作物类型
    result: Dict[str, int] = field(default_factory=dict)  # 识别结果

def parse_filename(filename: str, result: Dict[str, int]):
    # 使用正则表达式提取文件名中的信息
    pattern = r"(?P<variety_type>BR\d+-\d{3})(?P<location>\d)(?P<treatment>\d)(?P<rep>\d)_(?P<date>\d{4}-\d{2}-\d{2})_.*_(?P<view_type>tv|sv)_(?P<angle>\d{3})-\d+-\d+-\d+\.*"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename '{filename}' does not match the expected format.")

    variety_type = match.group('variety_type')
    location = int(match.group('location'))
    treatment = int(match.group('treatment'))
    rep = int(match.group('rep'))
    date = datetime.strptime(match.group('date'), "%Y-%m-%d").date()
    view_type = match.group('view_type')
    angle = match.group('angle')

    # 从映射中获取对应的作物类型，如果没有找到则设为"Unknown"
    crop_type = variety_to_crop.get(variety_type, "Unknown")

    return ImageMetadata(
        variety_type=variety_type,
        location=location,
        treatment=treatment,
        rep=rep,
        date=date,
        view_type=view_type,
        angle=angle,
        crop_type=crop_type,  # 新增字段
        result=result
    )

import pandas as pd

def calculate_average_result(data: List[ImageMetadata], group_by: List[str]) -> pd.DataFrame:
    # 将 ImageMetadata 对象转换为字典形式，并展开 result 字段
    data_expanded = []
    for item in data:
        base_info = item.__dict__.copy()
        base_info.pop('result')  # Remove 'result' to expand it later
        for obj, count in item.result.items():
            data_expanded.append({**base_info, 'object': obj, 'count': count})

    # 转换为 DataFrame
    df = pd.DataFrame(data_expanded)
    # 分组并计算平均值
    grouped_df = df.groupby(group_by + ['object']).agg({'count': 'mean'}).reset_index()

    return grouped_df


def plot_object_count_over_time(average_results: pd.DataFrame, object_name: str):
    # 过滤出指定 object 的数据
    df_filtered = average_results[average_results['object'] == object_name]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='count', data=df_filtered, marker='o')

    # 设置图表标题和标签
    plt.title(f'Average {object_name.capitalize()} Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Count')
    plt.xticks(rotation=45)  # 旋转日期标签以提高可读性
    plt.grid(True)
    plt.tight_layout()

    # 显示图表
    plt.show()

def plot_all_objects_over_time(average_results: pd.DataFrame, view_type: str, rep: int):
    # 根据指定的变量进行数据过滤
    df_filtered = average_results[
        (average_results['view_type'] == view_type) &
        (average_results['rep'] == rep)
        ]
    # 绘制多个对象随时间变化的折线图
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='date', y='count', hue='object', data=df_filtered, marker='o')

    # 设置图表标题和标签
    plt.title(f'Average Object Count Over Time (view_type={view_type}, rep={rep})')
    plt.xlabel('Date')
    plt.ylabel('Average Count')
    plt.xticks(rotation=45)  # 旋转日期标签
    plt.grid(True)
    plt.tight_layout()

    # 显示图表
    plt.show()

def plot_filtered_objects_over_time(average_results: pd.DataFrame,
                                    variety_type: str = None,  # 品种类型
                                    location: int = None,  # 位置
                                    treatment: int = None,  # 处理条件
                                    rep: int = None,  # 植物编号
                                    view_type: str = None,  # 观测视角
                                    object_names: list = None,  # 对象名称
                                    hue: str = None,  # 色彩编码变量
                                    ):
    # 根据用户提供的变量进行数据过滤
    df_filtered = average_results.copy()

    if variety_type is not None and hue != "variety_type":
        df_filtered = df_filtered[df_filtered['variety_type'] == variety_type]
    if location is not None and hue != "location":
        df_filtered = df_filtered[df_filtered['location'] == location]
    if treatment is not None and hue != "treatment":
        df_filtered = df_filtered[df_filtered['treatment'] == treatment]
    if rep is not None:
        df_filtered = df_filtered[df_filtered['rep'] == rep]
    if view_type is not None:
        df_filtered = df_filtered[df_filtered['view_type'] == view_type]
    if object_names is not None:
        df_filtered = df_filtered[df_filtered['object'].isin(object_names)]

    # 绘制多个对象随时间变化的折线图
    plt.figure(figsize=(12, 8))
    if hue is not None:
        sns.lineplot(x='date', y='count', hue=hue, style='object', data=df_filtered, marker='o')
    else:
        sns.lineplot(x='date', y='count', hue='object', data=df_filtered, marker='o')

    # 设置图表标题和标签
    title = 'Average Object Count Over Time'
    filters = []
    if variety_type is not None and hue != "variety_type":
        filters.append(f'variety_type={variety_type}')
    if location is not None and hue != "location":
        filters.append(f'location={location}')
    if treatment is not None and hue != "treatment":
        filters.append(f'treatment={treatment}')
    if rep is not None:
        filters.append(f'rep={rep}')
    if filters:
        title += ' (' + ', '.join(filters) + ')'

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Average Count')
    plt.xticks(rotation=45)  # 旋转日期标签
    plt.grid(True)
    plt.tight_layout()

    # 显示图表
    plt.show()

# 保存平均结果到 CSV 文件
def save_average_results_to_csv(average_results: pd.DataFrame, filename: str):
    # 检查文件是否存在
    if os.path.isfile(filename):
        # 读取现有数据
        existing_data = pd.read_csv(filename)
        # 合并新数据和现有数据
        combined_data = pd.concat([existing_data, average_results])
        print(f"Combined data before dropping duplicates: {combined_data.shape[0]} rows")
        # 删除所有重复的行
        subset_columns = ['variety_type', 'location', 'treatment', 'rep', 'date', 'view_type', 'object']
        combined_data = combined_data.drop_duplicates(subset=subset_columns, keep=False)
        print(combined_data)
        print(f"Combined data after dropping duplicates: {combined_data.shape[0]} rows")
    else:
        # 如果文件不存在，则直接使用新的数据
        combined_data = average_results

    # 保存数据到文件，覆盖原文件
    combined_data.to_csv(filename, index=False)

# 从 CSV 文件读取数据
def load_average_results_from_csv(filename: str) -> pd.DataFrame:
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        print(f"File {filename} does not exist.")
        return pd.DataFrame()  # 返回空的 DataFrame
