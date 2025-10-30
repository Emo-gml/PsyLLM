import json
import re
from openai import OpenAI
from tqdm import tqdm

# ========== CONFIGURATION ==========
# Initialize DeepSeek client (replace with your own API key before running)
# ⚠️ IMPORTANT: Never commit real API keys or local paths to public repositories.
clients = [
    OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", base_url="https://api.deepseek.com"),
]
current_client_index = 0


# ========== CORE INFERENCE FUNCTION ==========
def inference_with_deepseek_v3(sample_json_str: str) -> dict:
    """
    Evaluate the quality of multi-turn psychological counseling dialogue samples using DeepSeek.

    Args:
        sample_json_str (str): JSON string of a multi-turn sample with fields
                               `patient`, `counselor_think`, and `counselor_content`.

    Returns:
        dict: {
            "keep": bool,              # whether to keep the sample
            "issues": list[int],       # issue numbers (1–4)
            "reason": str              # concise explanation
        }

    Notes:
        - The model response is automatically stripped of Markdown code fences.
        - If the response is invalid or incomplete, it defaults to {"keep": False, ...}.
    """
    global current_client_index
    client = clients[current_client_index]
    current_client_index = (current_client_index + 1) % len(clients)

    system_prompt = (
        "You are a meticulous dialogue data quality reviewer for psychological counseling conversations. "
        "Each sample contains multiple turns with fields: `patient`, `counselor_think`, and `counselor_content`.\n\n"
        "Identify flawed or unusable samples based on these four criteria:\n"
        "1. counselor_think is incomplete or lacks analytical clarity.\n"
        "2. The dialogue is incoherent or contextually inconsistent.\n"
        "3. counselor_think and counselor_content are mismatched.\n"
        "4. counselor_think lacks grounding in recognized therapeutic frameworks "
        "(e.g., cognitive-behavioral, humanistic, psychodynamic) or ICD-11/DSM-5 principles.\n\n"
        "Return ONLY a valid JSON object in this exact format:\n"
        "{\n"
        "  \"keep\": true/false,\n"
        "  \"issues\": [list of issue numbers],\n"
        "  \"reason\": \"brief explanation\"\n"
        "}\n"
        "Do not include any commentary, Markdown, or additional text."
    )

    # Send request to DeepSeek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample_json_str},
        ],
        temperature=0,
    )

    result_text = response.choices[0].message.content.strip()

    # Extract JSON if wrapped in Markdown code blocks (```json ... ```)
    if result_text.startswith("```"):
        lines = result_text.splitlines()
        content_lines = []
        for line in lines[1:]:
            if line.strip().startswith("```"):
                break
            content_lines.append(line)
        result_text = "\n".join(content_lines).strip()

    try:
        result = json.loads(result_text)
        # Validate expected fields
        if not all(k in result for k in ("keep", "issues", "reason")):
            raise ValueError("Missing required keys in response JSON.")
        return result

    except Exception as e:
        # Fallback: Mark as filtered if parsing or formatting fails
        return {
            "keep": False,
            "issues": [],
            "reason": f"Invalid model response: {e} | Raw output: {result_text}",
        }


# ========== FILE UTILITIES ==========
def write_json_file(filepath: str, data: list):
    """Write a list of dictionaries to a UTF-8 JSON file with indentation."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ========== MAIN EXECUTION ==========
def main():
    """
    Main pipeline:
    1. Load multi-turn counseling samples from JSON file.
    2. Evaluate each using DeepSeek quality inference.
    3. Save results (keep flag, issues, reason, post_id) to output file.
    """
    input_file = "path/to/input.json"
    output_file = "path/to/output.json"

    # Load input JSON
    with open(input_file, "r", encoding="utf-8") as f:
        posts = json.load(f)

    results = []
    with tqdm(total=len(posts), desc="Filtering samples", unit="post") as pbar:
        for post in posts:
            post_id = post.get("post_id")
            sample_str = json.dumps(post, ensure_ascii=False)
            result = inference_with_deepseek_v3(sample_str)
            result["post_id"] = post_id
            results.append(result)
            pbar.update(1)

    # Save results
    write_json_file(output_file, results)
    print(f"✅ Filtering completed. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
