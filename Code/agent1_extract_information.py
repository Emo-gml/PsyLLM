import json
import os
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ========== 配置区 ==========
# 请在运行前填写你的输入/输出路径与 API Key
INPUT_PATH = "path/to/input.json"
OUTPUT_PATH = "path/to/output.json"
API_KEYS = [
    # 示例："sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
]
BASE_URL = "https://api.deepseek.com"
MAX_WORKERS = min(32, len(API_KEYS) * 2)

# ========== 初始化 ==========
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
clients = [OpenAI(api_key=key, base_url=BASE_URL) for key in API_KEYS]
client_index = 0
lock = Lock()


def get_client():
    """轮询返回 OpenAI 客户端实例"""
    global client_index
    client = clients[client_index]
    client_index = (client_index + 1) % len(clients)
    return client


def inference_with_deepseek(prompt: str) -> str:
    """调用 DeepSeek API 获取模型输出"""
    client = get_client()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0
    )
    return response.choices[0].message.content


def extract_round_info(output: str):
    """从模型输出中提取 JSON 内容"""
    text = output.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        data = json.loads(text)
    except Exception:
        match = JSON_OBJ_RE.search(text)
        data = json.loads(match.group()) if match else {}
    return data.get("rounds"), data.get("info_by_round")


def load_processed_ids(path: str):
    """加载已处理的帖子 ID，避免重复处理"""
    processed = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed.add(json.loads(line).get('post_id'))
                except Exception:
                    continue
    return processed


def build_prompt(content: str) -> str:
    """构建心理对话分析提示词"""
    return f"""
You are a compassionate and experienced psychological counselor. 
Analyze the following user post and simulate the planning process 
of a brief therapeutic conversation (1–3 rounds).

Post content:
["{content}"]

Return your output strictly in this JSON format:
{{
  "rounds": number (1–3),
  "info_by_round": [list of strings, each describing the focus of one round]
}}
""".strip()


def main():
    posts = json.load(open(INPUT_PATH, 'r', encoding='utf-8'))
    processed_ids = load_processed_ids(OUTPUT_PATH)

    with open(OUTPUT_PATH, 'a', encoding='utf-8') as outf:
        def worker(post):
            pid = post.get('post_id')
            with lock:
                if pid in processed_ids:
                    return

            prompt = build_prompt(post.get('content', ''))
            try:
                out = inference_with_deepseek(prompt)
                rounds, info = extract_round_info(out)
                if rounds is None or info is None:
                    return
                result = {
                    "post_id": pid,
                    "file": post.get('file'),
                    "rounds": rounds,
                    "info_by_round": info
                }
                with lock:
                    outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outf.flush()
                    processed_ids.add(pid)
            except Exception as e:
                print(f"Error processing post {pid}: {e}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker, post) for post in posts]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                pass

    print(f"All done. Results saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
