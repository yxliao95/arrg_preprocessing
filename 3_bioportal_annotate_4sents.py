""" 
使用Bioportal，对interpret-cxr拆分后的报告句子进行标注。
"""

import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from queue import Queue

import requests
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from requests.exceptions import RequestException
from tqdm import tqdm

PROJ_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.basename(__file__)
log_dir = os.path.join(PROJ_ROOT_DIR, "outputs", "logs")
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(log_dir, f"{file_name}.log"), "w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))

LOGGER = logging.getLogger("main")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(file_handler)
LOGGER.addHandler(console_handler)


def bioportal_api_annotate(text, ontologies="RADLEX"):
    url = "http://data.bioontology.org/annotator"
    json_data = {
        "text": text,
        "ontologies": ontologies,
        "display_context": "false",
        "display_links": "false",
    }
    headers = {
        "Authorization": "apikey token=0d58a2d5-cf25-4879-b0c6-58ad0fbe1392",
    }

    response = requests.post(url, headers=headers, json=json_data, timeout=300)

    if response.status_code != 200:
        LOGGER.warning("Response status code: %s", response.status_code)
        LOGGER.warning("Request data: %s", json_data)
        LOGGER.warning("Response text: %s", response.text)
        response.raise_for_status()

    return response.text


response_queue = queue.Queue()


def annotate(batch_dict, sent_text, safe_request_conter):
    try:
        ids_str = "|".join(batch_dict["doc_sent_split_ids"])
        out_json = {
            "ids": batch_dict["doc_sent_split_ids"],
            "sents": batch_dict["split_sents"],
            "sent_text": sent_text,
            "ann": bioportal_api_annotate(sent_text, ONTOLOGIES),
        }
        response_queue.put((ids_str, out_json, None))

    except Exception as e:
        LOGGER.error("Exception: %s, trackback: %s", e, traceback.print_exc())
        response_queue.put((ids_str, None, f"Exception: {e}"))

    safe_request_conter.add_response_count(1)  # 得到 bioportal_api_annotate 响应后才需要更新。异常后也需要更新，不然会进入死循环
    return


request_queue = queue.Queue()


def start_request(safe_request_conter):
    while True:
        # 服务器端每秒处理三条请求
        # 如果发送了3个请求，但只响应了2个，那么下一秒就只发送2个请求
        num_avail_request = safe_request_conter.get_num_avail_request()
        while num_avail_request:
            request_thread = request_queue.get()
            request_thread.start()
            # LOGGER.debug("Req [%s] start at %s", num_avail_request, datetime.now())
            request_queue.task_done()
            num_avail_request -= 1
        time.sleep(1.2)


class SafeRequestCounter:
    def __init__(self, max_count=3):
        self.response_counter_lock = threading.Lock()
        self.response_counter = max_count
        self.max_num_resquest = max_count

    def add_response_count(self, value):
        with self.response_counter_lock:
            self.response_counter += value

    def get_num_avail_request(self):
        with self.response_counter_lock:
            if self.response_counter > self.max_num_resquest:
                self.response_counter -= self.max_num_resquest
                return self.max_num_resquest
            else:
                num_avail_request = self.response_counter
                self.response_counter = 0
                return num_avail_request


def read_file_generator(file_path, cast_to=None):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            out = line.strip()
            yield json.loads(out) if cast_to == "json" else out


def count_file_lines(file_path):
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)


def seconds_to_time_str(seconds):
    hours = seconds // 3600  # 1小时 = 3600秒
    minutes = (seconds % 3600) // 60  # 除去小时部分后，再计算分钟
    seconds = seconds % 60  # 剩余的秒数

    return f"{hours:.0f}h {minutes:.0f}min {seconds:.1f}s"


def process_interpret_sentences():
    """对interpret-cxr的句子进行标注，句子是通过llm拆分后的句子，输入数据的格式与interpret-cxr的原格式不同.

    An example of input data line parsed by json:
    {'doc_key': 'test#2981#findings', 'sent_idx': 3, 'original_sent': 'The cardiac silhouette is top normal .', 'sent_splits': ['The cardiac silhouette is top normal.']}
    """
    data_dir = "/home/yuxiang/liao/workspace/arrg_sentgen/outputs/interpret_cxr"

    output_dir = os.path.join(PROJ_ROOT_DIR, "outputs", "interpret_sents", "bioportal_annotated_radlex")
    output_resume_done_dir = os.path.join(output_dir, "resume_log_done")
    output_resume_err_dir = os.path.join(output_dir, "resume_log_err")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_resume_done_dir, exist_ok=True)
    os.makedirs(output_resume_err_dir, exist_ok=True)

    # 已经完成的 data id：
    # train#132#impression#1@0
    # data_split # row_id # section # raw_sent_idx @ split_sent_idx
    index_file_path = os.path.join(output_resume_done_dir, f"done_doc_sent_split_ids.txt")
    visited_doc_sent_split_ids = set()
    if os.path.exists(index_file_path):
        with open(index_file_path, "r", encoding="utf-8") as f:
            visited_doc_sent_split_ids = set([i.strip() for i in f.readlines()])
        LOGGER.info("Loaded %s visited doc_sent_split_id.", len(visited_doc_sent_split_ids))

    # 读取数据，将数据分成batch，每个batch包含多个句子，以减少api调用次数。api每秒仅允许3次请求
    """
    Example of batched_sents[0]:
    {'doc_sent_split_ids':
        ['train#0#impression#0@0',
        ...
        'train#3#findings#0@1'],
    'split_sents':
        ['Decreased bibasilar parenchymal opacities are seen.',
        ...
        'A dual lead AICD is demonstrated.']}
    """
    # Load data
    empty_count = 0
    done_count = 0

    batched_sents = []
    batch_size = 16  # at least 10 sentences in a batch
    curr_inbatch_idx = 0
    curr_batch_dict = {"doc_sent_split_ids": [], "split_sents": []}

    for partition in [1, 2, 3]:
        input_file_path = os.path.join(data_dir, f"llm_sent_splits_{partition}_of_3.json")
        output_file_path = os.path.join(output_dir, f"train_dev_test_empty.dockey")
        f = open(output_file_path, "w")

        for doc in tqdm(read_file_generator(input_file_path, cast_to="json"), desc=f"Loading data partition {partition} of 3"):

            if not doc["sent_splits"]:
                # Doc sectence is empty, skip it.
                f.write(f"{doc['doc_key']}\n")
                empty_count += 1
            else:
                for split_idx, split_sent in enumerate(doc["sent_splits"]):
                    doc_sent_split_id = f"{doc['doc_key']}#{doc['sent_idx']}@{split_idx}"

                    # 跳过已完成数据
                    if doc_sent_split_id in visited_doc_sent_split_ids:
                        done_count += 1
                        LOGGER.debug("Skip finished doc_sent_split_id: %s", doc_sent_split_id)
                        continue

                    curr_batch_dict["doc_sent_split_ids"].append(doc_sent_split_id)
                    curr_batch_dict["split_sents"].append(split_sent)
                    curr_inbatch_idx += 1

            if curr_inbatch_idx >= batch_size:
                batched_sents.append(curr_batch_dict)
                curr_batch_dict = {"doc_sent_split_ids": [], "split_sents": []}
                curr_inbatch_idx = 0

        f.close()

    # Update the last batch
    if curr_inbatch_idx > 0:
        batched_sents.append(curr_batch_dict)

    LOGGER.info("Skipped %s empty doc sections.", empty_count)
    LOGGER.info("Skipped %s finished sentences", done_count)

    # 负责启动请求线程的线程。守护线程，随主线程自动关闭
    # 服务器：每秒处理三条请求
    safe_request_conter = SafeRequestCounter(3)
    request_starter_thread = threading.Thread(target=start_request, args=(safe_request_conter,), daemon=True)
    request_starter_thread.start()
    start = time.time()

    output_file_path = os.path.join(output_dir, f"train_dev_test.jsonlines")
    error_file_path = os.path.join(output_resume_err_dir, f"err_indics.txt")

    # 遍历数据集，提交线程任务
    tol_num_requests = 0

    for idx, batch_dict in enumerate(tqdm(batched_sents, desc=f"Preparing request")):
        data_ids = batch_dict["doc_sent_split_ids"]

        """
        Example of sent_text:
        <|0|> Decreased bibasilar parenchymal opacities are seen. <|END|><|1|> The bibasilar parenchymal opacities are now minimal. <|END|><|2|> Stable small left pleural effusion. <|END|><|3|> Feeding tube is again seen. <|END|><|x|> xxxx <|END|>
        """
        sent_text = "".join([f"<|{idx}|> {sent} <|END|>" for idx, sent in enumerate(batch_dict["split_sents"])])
        assert sent_text != ""

        request_thread = threading.Thread(target=annotate, args=(batch_dict, sent_text, safe_request_conter))
        request_queue.put(request_thread)

        tol_num_requests += 1

    # 等待所有请求线程完成
    # 每个请求线程，都会产生一个response_queue item
    num_finished_requests = 0
    with open(output_file_path, "a") as f_out, open(index_file_path, "a") as f_idx, open(error_file_path, "a") as f_err, tqdm(total=tol_num_requests, desc=f"Processing request") as pbar:
        if tol_num_requests > 0:
            while True:
                ids_str, out_json, err_msg = response_queue.get()
                doc_sent_split_ids = ids_str.split("|")

                if out_json:
                    jsonline = json.dumps(out_json, separators=(",", ":"))
                    f_out.write(jsonline + "\n")
                    for data_id in doc_sent_split_ids:
                        f_idx.write(data_id + "\n")
                if err_msg:
                    for data_id in doc_sent_split_ids:
                        f_err.write(f"{data_id} | {err_msg}\n")

                response_queue.task_done()
                num_finished_requests += 1
                pbar.update(1)

                end = time.time()
                LOGGER.info("Finished request: %s, Time elapsed: %s", (num_finished_requests + 1), seconds_to_time_str(end - start))

                if num_finished_requests >= tol_num_requests:  # 检查是否收到结束信号
                    break

        LOGGER.info("!!! File check. %s lines in %s", count_file_lines(output_file_path), output_file_path)


if __name__ == "__main__":
    ONTOLOGIES = "RADLEX"
    process_interpret_sentences()
