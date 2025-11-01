import argparse, json
from pathlib import Path
from agent_graph import build_graph, GraphState

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--questions-file", required=True, help="JSONL with {\"question\": \"...\"}")
    ap.add_argument("--out", default="runs/langgraph_batch")
    args = ap.parse_args()

    out_dir = Path(args.out); (out_dir).mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"

    graph = build_graph(args.index_dir)

    with open(args.questions_file, "r", encoding="utf-8") as fin, \
         open(answers_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            q = json.loads(line).get("question")
            if not q: continue
            result = graph.invoke(GraphState(question=q))
            rec = {"question": q, "answer": result.answer}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\\n")

    print("Wrote", answers_path)

if __name__ == "__main__":
    main()
