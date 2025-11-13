import base64
import json
import os
from openai import OpenAI
import time
import concurrent.futures
import threading
import pandas as pd
API_KEY = 'sk-ZCcrX9JD8AfdGohshzZKgPgWV9guGZVJfgAOFy4go1Ctpl7Z'
BASE_URL = "https://api.chatanywhere.tech/v1"

# 设置包含图片的目录路径
IMAGE_DIRECTORY = r'dataset\NIPS2020\public_data\images'

# 设置输出JSON文件的路径
OUTPUT_JSON_FILE = r'dataset\NIPS2020\ocr_results_concurrent.json'

# --- 并发与速率控制配置 ---

# 每批处理的图片数量 (可以根据图片大小和API响应时间微调)
BATCH_SIZE = 5

# API速率限制：每分钟最多允许的请求次数
REQUESTS_PER_MINUTE = 30

# 线程池的最大工作线程数。这代表最多有多少个请求可以“同时”处于正在进行的状态。
# 建议这个值不要超过 REQUESTS_PER_MINUTE。
MAX_CONCURRENCY = 30

# --- 主逻辑部分 ---

def image_to_base64(image_path):
    """将图片文件转换为Base64编码的字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
        return None

def process_batch(client, image_paths):

    # 1. 构建API请求的消息体
    messages_content = []
    prompt_text = (
        "你是一个OCR助手。请识别我接下来提供的每一张图片中的问题文本和选项。文本严格按如下格式给出:\nquestion: xxx\nA: xx\nB: xx\n......"
        "你需要返回一个单一的JSON对象，其中键（key）是图片的文件名，值（value）是该图片对应的识别文本。"
        "不要在JSON对象之外添加任何解释、注释或多余的文字。"
        "下面是图片列表："
    )
    messages_content.append({"type": "text", "text": prompt_text})

    # 2. 添加图片数据和文件名到请求中
    filenames_in_batch = []
    for image_path in image_paths:
        base64_image = image_to_base64(image_path)
        if base64_image:
            filename = os.path.basename(image_path)
            filenames_in_batch.append(filename)
            messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            messages_content.append({"type": "text", "text": f"这是文件名: {filename}"})
    
    if not filenames_in_batch:
        return {}

    # 3. 发送API请求
    try:
        print(f"线程 {threading.get_ident()} 正在发送请求，处理文件: {filenames_in_batch}")
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": messages_content}],
            stream=False,
            temperature=0,
        )
        response_content = response.choices[0].message.content
        # 4. 解析模型返回的JSON字符串
        if response_content.startswith("```json"):
            response_content = response_content.strip("```json\n").strip("```")
        
        return json.loads(response_content)

    except json.JSONDecodeError as e:
        print(f"错误：API返回的不是有效的JSON格式。批次: {filenames_in_batch}, 错误: {e}")
        return {filename: "解析JSON失败" for filename in filenames_in_batch}
    except Exception as e:
        print(f"处理批次 {filenames_in_batch} 时发生API调用错误: {e}")
        return {filename: "API调用出错" for filename in filenames_in_batch}

def main_concurrent():
    """
    主函数，使用线程池并发处理所有图片。
    """
    # 1. 初始化客户端和准备数据
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    all_image_files = [f"{i}.jpg" for i in range(948)]
    final_ocr_results = {}
    
    # 2. 将所有图片路径按 BATCH_SIZE 分批
    image_batches = []
    for i in range(0, len(all_image_files), BATCH_SIZE):
        batch_files = all_image_files[i:i + BATCH_SIZE]
        batch_paths = [os.path.join(IMAGE_DIRECTORY, f) for f in batch_files]
        image_batches.append(batch_paths)

    # 计算提交任务的间隔时间，以满足速率限制
    # 例如：60秒 / 10次 = 每次提交后等待6秒
    delay_between_requests = 60 / REQUESTS_PER_MINUTE

    # 3. 使用ThreadPoolExecutor进行并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_batch = {}
        
        # 4. 以受控的速率提交所有任务到线程池
        print(f"准备提交 {len(image_batches)} 个批次任务，每 {delay_between_requests:.1f} 秒提交一个...")
        for batch in image_batches:
            # 提交任务，executor.submit会立即返回一个future对象
            future = executor.submit(process_batch, client, batch)
            future_to_batch[future] = batch
            # 等待指定时间，控制提交速率
            time.sleep(delay_between_requests)
        
        # 5. 收集已完成任务的结果
        print("\n所有任务已提交，等待处理完成...")
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                # 获取线程执行的结果
                batch_result = future.result()
                # 更新到最终结果字典
                final_ocr_results.update(batch_result)
                print(f"一个批次处理完成，已获取 {len(batch_result)} 条结果。")
            except Exception as exc:
                batch_info = future_to_batch[future]
                print(f"一个批次在执行中产生严重错误: {batch_info}, 错误: {exc}")

    # 6. 将所有结果写入JSON文件
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_ocr_results, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 所有OCR识别结果已成功保存到文件: {OUTPUT_JSON_FILE}")
    except Exception as e:
        print(f"❌ 写入JSON文件时发生错误: {e}")

def get_last_subject_id(subject_id_str):
    try:
        # 使用json.loads将字符串转换为列表
        subject_id_list = json.loads(subject_id_str)
        if isinstance(subject_id_list, list) and subject_id_list:
            return subject_id_list[-1]
    except (json.JSONDecodeError, TypeError):
        # 如果转换失败或数据不是列表，则返回None
        return None
    return None


def transform_subjectid():
    try:
        df = pd.read_csv("dataset/NIPS2020/public_data/metadata/question_metadata_task_3_4.csv")

        df['SubjectId'] = df['SubjectId'].apply(get_last_subject_id)

        # 删除 'SubjectId' 为None的行（即原始数据格式不正确的行）
        df.dropna(subset=['SubjectId'], inplace=True)
        
        # 将 'SubjectId' 转换为整数类型
        df['SubjectId'] = df['SubjectId'].astype(int)

        # 按 'QuestionId' 排序
        df_sorted = df.sort_values(by='QuestionId')

        # 将处理后的DataFrame转换为JSON格式字符串
        # orient='records' 会生成一个类似 [{'column': 'value'}, ... ] 的格式
        json_output = df_sorted.to_json(orient='records', indent=4)

        # 计算并输出有多少个不同的SubjectId
        unique_subject_id_count = df_sorted['SubjectId'].nunique()
        print(f"\n处理后共有 {unique_subject_id_count} 个不同的SubjectId。")
        with open(r'dataset\NIPS2020\Q_to_K.json', 'w', encoding='utf-8') as f:
            f.write(json_output)


    except FileNotFoundError:
        print("错误：文件未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")

def aggregate_student_data(base_path):

    # 1. 定义文件路径
    train_data_path = os.path.join(base_path, 'public_data', 'train_data', 'train_task_3_4.csv')
    answer_meta_path = os.path.join(base_path, 'public_data', 'metadata', 'answer_metadata_task_3_4.csv')

    print(f"正在从以下路径读取文件:")
    print(f"- {train_data_path}")
    print(f"- {answer_meta_path}")

    # 2. 读取CSV文件并处理错误
    try:
        # 读取训练数据，只选择需要的列以节省内存
        train_df = pd.read_csv(train_data_path, usecols=['UserId', 'QuestionId', 'AnswerId', 'IsCorrect'])
        
        # 读取答案元数据，只选择需要的列
        answer_df = pd.read_csv(answer_meta_path, usecols=['AnswerId', 'DateAnswered'])
    except FileNotFoundError as e:
        return f"错误：文件未找到 - {e}. 请确保路径 '{base_path}' 正确。"
    except Exception as e:
        return f"读取文件时发生错误: {e}"

    # 3. 合并数据以获取时间戳
    # 使用 'inner' 合并，确保每一条训练数据都有对应的时间戳
    print("正在合并训练数据和答案元数据...")
    merged_df = pd.merge(train_df, answer_df, on='AnswerId')

    # 4. 转换日期格式以便排序
    # errors='coerce' 会将无法解析的日期变为NaT（Not a Time）
    print("正在将日期字符串转换为datetime对象...")
    merged_df['DateAnswered'] = pd.to_datetime(merged_df['DateAnswered'], errors='coerce')

    # 删除无法解析日期的行
    merged_df.dropna(subset=['DateAnswered'], inplace=True)

    # 5. 按用户ID分组
    print("按UserId对数据进行分组...")
    grouped_by_user = merged_df.groupby('UserId')

    # 用于存储每个用户的格式化文本块
    all_user_blocks = []

    print(f"找到 {len(grouped_by_user)} 个独立用户。正在处理...")
    # 6. 遍历每个用户，生成格式化数据
    for user_id, user_data in grouped_by_user:
        # a. 按答题时间排序
        sorted_user_data = user_data.sort_values(by='DateAnswered')

        # b. 获取序列长度
        sequence_length = len(sorted_user_data)
        
        # 如果用户没有任何有效交互，则跳过
        if sequence_length == 0:
            continue

        # c. 提取QuestionId和IsCorrect序列
        question_ids = sorted_user_data['QuestionId'].astype(str).tolist()
        is_correct_values = sorted_user_data['IsCorrect'].astype(str).tolist()

        # d. 拼接成逗号分隔的字符串
        questions_str = ",".join(question_ids)
        correctness_str = ",".join(is_correct_values)

        # e. 组合成四行格式
        user_block = f"{sequence_length}\n{questions_str}\n{correctness_str}"
        all_user_blocks.append(user_block)

    print("所有用户处理完毕。")
    # 7. 将所有用户的文本块合并成一个大字符串
    with open('dataset/NIPS2020/used.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(all_user_blocks))
        
def analyze_and_sort_ocr_results(file_path):
 
    
    # --- 1. 读取和加载JSON文件 ---
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取文件: {file_path}")
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径: {file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的JSON格式。")
        return

    # --- 2. 提取所有图片编号 ---
    image_numbers = []
    malformed_keys = []
    for key in data.keys():
        # 移除'.jpg'后缀并尝试转换为整数
        try:
            # 假设格式总是 "数字.jpg"
            number_str = key.replace('.jpg', '')
            image_numbers.append(int(number_str))
        except ValueError:
            # 如果键的格式不正确（例如 "image_a.jpg"），则记录下来
            malformed_keys.append(key)
    
    if malformed_keys:
        print(f"\n警告：发现以下格式不正确的键，已忽略: {malformed_keys}")

    print(f"\n文件中总共找到 {len(image_numbers)} 个有效的图片条目。")

    # --- 3. 检查编号是否唯一 ---
    print("\n--- 分析开始 ---")
    if len(image_numbers) == len(set(image_numbers)):
        print("✅ 唯一性检查: 所有图片编号都是唯一的。")
    else:
        # 找出重复的编号
        seen = set()
        duplicates = {x for x in image_numbers if x in seen or seen.add(x)}
        print(f"❌ 唯一性检查: 发现重复的编号: {sorted(list(duplicates))}")

    # --- 4. 检查编号是否是从 0 到 947 的完整序列 ---
    expected_numbers = set(range(948)) # 应该包含0到947，共948个数字
    actual_numbers = set(image_numbers)

    if actual_numbers == expected_numbers:
        print("✅ 完整性检查: 编号序列是从 0 到 947 的完整序列。")
    else:
        print("❌ 完整性检查: 编号序列不完整。")
        missing = sorted(list(expected_numbers - actual_numbers))
        extra = sorted(list(actual_numbers - expected_numbers))
        
        if missing:
            print(f"   - 缺失的编号: {missing}")
        if extra:
            print(f"   - 额外的编号 (不在0-947范围内): {extra}")

    print("--- 分析结束 ---")

    # --- 5. 对数据进行排序并保存到新文件 ---
    # 根据提取的整数编号对原始字典的键进行排序
    sorted_keys = sorted(data.keys(), key=lambda k: int(k.replace('.jpg', '')))
    
    # 创建一个新的有序字典
    sorted_data = {key: data[key] for key in sorted_keys}

    # 定义输出文件名
    output_path = os.path.join(os.path.dirname(file_path), "ocr_results_sorted.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=4)
        print(f"\n已将排序后的结果保存到新文件: {output_path}")
    except Exception as e:
        print(f"\n保存排序文件时出错: {e}")

import pandas as pd
import json
import os

def merge_question_data(base_path, output_filename="enriched_question_data.json"):
    """
    合并问题ID、知识点ID、问题文本和知识点名称，并保存为一个新的JSON文件。

    参数:
    base_path (str): 'dataset/NIPS2020' 目录的路径。
    output_filename (str): 输出的JSON文件名。
    """
    
    # --- 1. 定义所有文件的路径 ---
    q_to_k_path = os.path.join(base_path, 'Q_to_K.json')
    ocr_results_path = os.path.join(base_path, 'ocr_results_sorted.json')
    subject_meta_path = os.path.join(base_path, 'public_data', 'metadata', 'subject_metadata.csv')
    output_path = os.path.join(base_path, output_filename)

    print("--- 开始合并数据 ---")
    print(f"主文件: {q_to_k_path}")
    print(f"问题文本来源: {ocr_results_path}")
    print(f"知识点名称来源: {subject_meta_path}")

    # --- 2. 加载所有数据源 ---
    try:
        # 加载主JSON文件 (QuestionId -> SubjectId)
        with open(q_to_k_path, 'r', encoding='utf-8') as f:
            q_to_k_data = json.load(f)

        # 加载OCR结果JSON (QuestionId -> QuestionText)
        with open(ocr_results_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        # 加载知识点元数据CSV，并创建一个快速查找的字典
        # 我们将SubjectId作为键，Name作为值
        subject_df = pd.read_csv(subject_meta_path)
        # to_dict()方法可以高效地完成这个转换
        subject_lookup = subject_df.set_index('SubjectId')['Name'].to_dict()
        
    except FileNotFoundError as e:
        print(f"\n错误：文件未找到 - {e}. 请确保所有输入文件都在正确的位置。")
        return
    except Exception as e:
        print(f"\n加载文件时发生错误: {e}")
        return

    # --- 3. 遍历主数据，并进行丰富化处理 ---
    enriched_data = []
    print(f"\n正在处理 {len(q_to_k_data)} 条问题数据...")

    for item in q_to_k_data:
        question_id = item.get("QuestionId")
        subject_id = item.get("SubjectId")

        # a. 检查数据是否有效
        if question_id is None or subject_id is None:
            continue # 如果条目缺少ID，则跳过

        # b. 查找问题文本
        # 构建OCR文件中的键，例如 QuestionId 941 -> "941.jpg"
        ocr_key = f"{question_id}.jpg"
        question_text = ocr_data.get(ocr_key, "Text not found")
        
        # 清理文本，移除可能存在的前缀 "question: "
        if question_text.lower().startswith("question:"):
            question_text = question_text[len("question:"):].strip()
            
        # c. 查找知识点名称
        # 直接在字典中查找，如果找不到则提供默认值
        subject_name = subject_lookup.get(subject_id, "Subject name not found")

        # d. 构建新的、包含所有信息的字典
        new_item = {
            "QuestionId": question_id,
            "SubjectId": subject_id,
            "QuestionText": question_text,
            "SubjectName": subject_name
        }
        enriched_data.append(new_item)

    # --- 4. 将最终结果保存到新的JSON文件 ---
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=4)
        print("\n--- 处理完成 ---")
        print(f"成功！已将合并后的数据保存到: {output_path}")

        # 打印一个条目作为预览
        if enriched_data:
            print("\n--- 数据预览 (第一条) ---")
            print(json.dumps(enriched_data[0], indent=4))
            
    except Exception as e:
        print(f"\n保存最终JSON文件时出错: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":

# 定义数据集的基础路径
# 脚本将在此路径下寻找所有需要的文件
    dataset_base_path = 'dataset/NIPS2020'

    # 执行合并功能
    merge_question_data(dataset_base_path)

       