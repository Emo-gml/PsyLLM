import os
import json
import threading
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== CONFIGURATION ==========
# Before releasing publicly:
# - Replace API keys with placeholders
# - Use relative paths or environment variables
clients = [
    OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", base_url="https://api.deepseek.com"),
]
MAX_WORKERS = 5  # Number of concurrent threads


# ========== MODEL INFERENCE ==========
def inference_with_deepseek_r1(text, conversation_history, client):
    """
    Generate a counselor response based on a single patient input and conversation history.

    Args:
        text (str): Current patient message.
        conversation_history (list): List of previous conversation turns (role, content).
        client (OpenAI): Initialized OpenAI API client.

    Returns:
        tuple[str, str]: (counselor reply, reasoning/thinking text)
    """
    system_prompt = (
        "You are a warm, compassionate, and emotionally attuned psychological counselor. "
        "Read the client’s message carefully and respond with empathy and emotional understanding.\n\n"
        "Guidelines:\n"
        "1. Silently refer to ICD-11 and DSM-5 as background for emotional awareness, "
        "but never include or imply any diagnostic or technical terms.\n"
        "2. Suggest one helpful therapeutic approach (e.g., cognitive behavioral, humanistic, psychodynamic, "
        "family systems, integrative) in simple and relatable language. "
        "Briefly describe how it may support the client emotionally, without sounding academic.\n"
        "3. Use a conversational, caring tone. Avoid robotic or formal responses.\n"
        "4. Keep responses concise: ≤50 words for light input; ≤150 words for emotional depth.\n"
        "Focus entirely on the client’s emotional needs, not theoretical frameworks."
    )

    # Combine conversation history and current input
    full_conversation = [{"role": "user", "content": text}] + conversation_history

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "system", "content": system_prompt}] + full_conversation,
        temperature=1.0
    )

    reply = response.choices[0].message.content
    think = response.choices[0].message.reasoning_content  # “Think” reasoning output
    return reply, think


# ========== FILE I/O ==========
def write_json_file(filename, posts):
    """Save JSON data to file in UTF-8 encoding."""
    with open(filename, "w", encoding="utf-8") as fout:
        json.dump(posts, fout, ensure_ascii=False, indent=2)


# ========== MAIN PIPELINE ==========
def main():
    """Main execution pipeline: load data, infer counselor responses, save results."""
    input_file = "path/to/input.json"
    output_file = "path/to/output.json"

    # Load input or previously saved progress
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fin:
            posts = json.load(fin)
    else:
        with open(input_file, "r", encoding="utf-8") as fin:
            posts = json.load(fin)

    # Prepare inference tasks
    tasks = []
    for i, post in enumerate(posts):
        for j, conv in enumerate(post.get("conversation", [])):
            if not conv.get("counselor_content") and conv.get("patient"):
                # Build prior conversation history (excluding reasoning)
                conversation_history = []
                for prev_conv in post["conversation"][:j]:
                    if prev_conv.get("counselor_content"):
                        conversation_history.append({"role": "assistant", "content": prev_conv["counselor_content"]})
                    if prev_conv.get("patient"):
                        conversation_history.append({"role": "user", "content": prev_conv["patient"]})

                tasks.append((i, j, conv["patient"], conversation_history))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No pending conversation rounds to process.")
        return

    # Thread synchronization
    file_lock = threading.Lock()
    client_lock = threading.Lock()
    client_index = 0

    def get_client():
        """Rotate among clients safely across threads."""
        nonlocal client_index
        with client_lock:
            c = clients[client_index]
            client_index = (client_index + 1) % len(clients)
        return c

    # Concurrent execution
    update_counter = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         tqdm(total=total_tasks, desc="Processing conversation rounds", unit="round") as pbar:

        future_to_task = {
            executor.submit(
                inference_with_deepseek_r1,
                task[2],      # patient text
                task[3],      # conversation history
                get_client()  # selected client
            ): task for task in tasks
        }

        for future in as_completed(future_to_task):
            post_idx, conv_idx, _, _ = future_to_task[future]
            try:
                reply, think = future.result()
            except Exception as e:
                print(f"Error on post {post_idx} conv {conv_idx}: {e}")
                pbar.update(1)
                continue

            # Update in-memory structure
            conv = posts[post_idx]["conversation"][conv_idx]
            conv["counselor_content"] = reply
            conv["counselor_think"] = think  # Store reasoning content separately

            # Periodic auto-save (every 10 updates)
            update_counter += 1
            if update_counter % 10 == 0:
                with file_lock:
                    write_json_file(output_file, posts)

            pbar.update(1)

    # Final save
    with file_lock:
        write_json_file(output_file, posts)
    print("All done!")


if __name__ == "__main__":
    main()
