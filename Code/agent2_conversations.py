import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial
import signal
import sys
from threading import Thread

# ========== CONFIGURATION ==========
# Before running, specify your input/output file paths and API keys.
# Do NOT commit actual API keys or local absolute paths to any public repository.
INPUT_PATH = "path/to/input.json"
OUTPUT_PATH = "path/to/output.json"
TEMP_OUTPUT_PATH = OUTPUT_PATH + ".temp"
MAX_RETRIES = 3           # Maximum retry attempts per record
PROCESSES = 4             # Number of worker processes (2–3x CPU cores recommended)
BATCH_SIZE = 25           # Batch size to prevent memory overload


# ========== CLIENT INITIALIZATION ==========
def init_client():
    """Initialize OpenAI client(s) for each process."""
    return [
        OpenAI(
            api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            base_url="https://api.deepseek.com"
        ),
    ]


# Each process will have its own client instance
clients = None


def init_process():
    """Initialize OpenAI clients for each process."""
    global clients
    clients = init_client()


# ========== CORE INFERENCE FUNCTION ==========
def inference_generate_conversation(post_id, info_by_round, retry_count=0):
    """
    Generate a realistic, multi-turn counselor-patient conversation
    based on the emotional themes extracted from the previous model output.

    Each round includes both 'patient' and 'counselor' turns.
    Retries up to MAX_RETRIES times upon API failure.
    """
    global clients
    current_client = clients[0]  # Single-client mode; can be extended to round-robin

    system_prompt = """
You are a compassionate and experienced psychological counselor.
Your task is to reconstruct a realistic, emotionally attuned multi-turn conversation (up to 3 rounds)
between a patient and a counselor, based on the emotional themes provided for each round.

Each round must include:
Patient: A natural, first-person paragraph describing their current emotional experience.
Counselor: A warm, reflective, and gentle response that demonstrates empathy and support.

Instructions:
- Ensure a coherent conversational flow across turns.
- Each counselor response should deepen the exploration naturally.
- Avoid vague or abstract metaphors; use concrete emotional language.
Return JSON with this format:
{
  "conversation": [
    {"round": 1, "patient": "...", "counselor": "..."},
    ...
  ]
}
"""

    prompt_payload = {
        "rounds": len(info_by_round),
        "info_by_round": info_by_round
    }
    user_message = f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"

    try:
        resp = current_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_message}
            ],
            temperature=1.0
        )

        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        conv = json.loads(content)
        return {
            "post_id": post_id,
            "conversation": conv["conversation"]
        }

    except Exception as e:
        if retry_count < MAX_RETRIES:
            time.sleep(1)
            return inference_generate_conversation(post_id, info_by_round, retry_count + 1)
        raise Exception(f"API request failed after {MAX_RETRIES} retries: {e}")


# ========== DATA PROCESSING HELPERS ==========
def process_record(record, progress_queue=None):
    """Process a single record and report progress to the queue."""
    post_id = record["post_id"]
    info_by_round = record["info_by_round"]
    try:
        result = inference_generate_conversation(post_id, info_by_round)
        if progress_queue:
            progress_queue.put(1)
        return result
    except Exception as e:
        print(f"\nPost {post_id} failed: {e}")
        if progress_queue:
            progress_queue.put(0)
        return None


def load_ndjson(input_path):
    """Load a line-delimited JSON (NDJSON) file into a list of records."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {line[:50]}... Error: {e}")
    return records


def load_processed_records(output_path):
    """
    Load previously saved results (supports both full JSON and NDJSON formats).
    Returns: (records, processed_ids)
    """
    temp_path = output_path + ".temp"
    if os.path.exists(temp_path):
        output_path = temp_path

    if not os.path.exists(output_path):
        return [], set()

    records = []
    processed_ids = set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            if content.startswith("["):
                try:
                    records = json.loads(content)
                    processed_ids = {rec["post_id"] for rec in records}
                    return records, processed_ids
                except json.JSONDecodeError:
                    pass

            # Fallback: read line by line
            f.seek(0)
            for line in f:
                line = line.strip().rstrip(",")
                if not line or line in ("[", "]"):
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                    processed_ids.add(record["post_id"])
                except json.JSONDecodeError as e:
                    print(f"Failed to parse line: {line[:50]}... Error: {e}")
                    continue
    except Exception as e:
        print(f"Failed to load existing output: {e}")

    return records, processed_ids


def save_progress(output_path, records):
    """Save progress to a temporary file, then atomically replace the original."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".temp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    if output_path != temp_path:
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)


# ========== PROGRESS MONITORING ==========
def update_progress(progress_queue, total, pbar):
    """Live update for tqdm progress bar across processes."""
    success_count = 0
    fail_count = 0
    while True:
        status = progress_queue.get()
        if status is None:
            break
        if status == 1:
            success_count += 1
        else:
            fail_count += 1
        pbar.set_description(f"Processing (Success: {success_count}, Fail: {fail_count})")
        pbar.update(1)


def signal_handler(sig, frame):
    """Handle interrupt signal (Ctrl+C) and exit gracefully."""
    print("\nInterrupt detected. Saving progress before exit...")
    sys.exit(0)


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    existing_records, processed_ids = load_processed_records(OUTPUT_PATH)
    print(f"Loaded {len(existing_records)} existing records")

    records = load_ndjson(INPUT_PATH)
    print(f"Loaded {len(records)} input records")

    records_to_process = [rec for rec in records if rec["post_id"] not in processed_ids]
    print(f"{len(records_to_process)} records remain to process")

    if not records_to_process:
        print("No new records to process. Exiting.")
        sys.exit(0)

    manager = Manager()
    progress_queue = manager.Queue()
    pbar = tqdm(total=len(records_to_process), desc="Generating Conversations")

    progress_thread = Thread(target=update_progress, args=(progress_queue, len(records_to_process), pbar))
    progress_thread.start()

    results = []
    try:
        with Pool(processes=PROCESSES, initializer=init_process) as pool:
            process_func = partial(process_record, progress_queue=progress_queue)

            for i in range(0, len(records_to_process), BATCH_SIZE):
                batch = records_to_process[i:i + BATCH_SIZE]
                batch_results = pool.map(process_func, batch)

                successful_results = [res for res in batch_results if res is not None]
                results.extend(successful_results)

                combined_records = existing_records + results
                save_progress(TEMP_OUTPUT_PATH, combined_records)

                print(f"\nProcessed {i + len(batch)}/{len(records_to_process)} records, "
                      f"successful: {len(successful_results)}")

    except Exception as e:
        print(f"\nError during processing: {e}")

    finally:
        progress_queue.put(None)
        progress_thread.join()
        pbar.close()

        combined_records = existing_records + results
        save_progress(OUTPUT_PATH, combined_records)

        print(f"\nProcessing completed. Success: {len(results)}, "
              f"Fail: {len(records_to_process) - len(results)}")
        print(f"Results saved to: {OUTPUT_PATH}")
