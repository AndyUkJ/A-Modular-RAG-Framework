# my_code/run_system.py
import argparse, json
from pathlib import Path
from app.system import answer_question
from app.di.factory import load_settings   # ✅ 新增


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)   # HotpotQA 是大 JSON 数组


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="full", choices=["graph_only", "full"])
    ap.add_argument("--output", type=str, default="results.json", help="批量模式结果输出文件")
    args = ap.parse_args()

    # ✅ 从 settings.yaml 读取 dataset 配置
    settings = load_settings("config/settings.yaml")
    ds_cfg = settings.get("dataset", {})
    dataset_path = ds_cfg.get("path", "data/hotpotqa/hotpot_dev_distractor_v1.json")
    index = ds_cfg.get("index", 0)
    count = ds_cfg.get("count", 1)

    data = load_dataset(dataset_path)

    # 截取子集
    if count == -1:
        subset = data[index:]
    else:
        subset = data[index:index+count]

    results = []
    for i, sample in enumerate(subset, start=index):
        q = sample["question"]
        gold = sample.get("answer", None)

        res = answer_question(q, mode=args.mode)

        results.append({
            "id": sample.get("_id", f"sample-{i}"),
            "question": q,
            "gold": gold,
            "result": res,
        })

        reasoning = res.get("reasoning") or {}
        answer = reasoning.get("answer") if isinstance(reasoning, dict) else None
        print(f"[{i}] Q: {q[:50]}... → Pred: {(answer or 'N/A')[:50]}")

        if count == 1:
            print("=== FULL RESULT ===")
            import pprint; pprint.pprint(res, width=120)
            if gold:
                print(f"\nGold Answer: {gold}")

    if count != 1:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
