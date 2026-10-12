"""Train/evaluate compact candidates without consuming any phone attempts."""

from pathlib import Path
import json
import torch
from starfall.experiments import train_policy, evaluate, compare


def main():
    torch.set_num_threads(2)
    ppo = sorted(Path("models").glob("ppo-synthetic-*/candidate.zip"))[-1]
    print("Training DQN candidate in synthetic sandbox", flush=True)
    dqn = train_policy("dqn", 8192, 42) / "candidate.zip"
    scenarios = [
        ("stay", None, "ppo"),
        ("nearest", None, "ppo"),
        ("random", None, "ppo"),
        ("mpc", None, "ppo"),
        ("learned", str(ppo), "ppo"),
        ("learned", str(dqn), "dqn"),
    ]
    reports = {}
    for policy, model, algorithm in scenarios:
        name = algorithm if model else policy
        print("Evaluating", name, "on 30 paired held-out synthetic seeds", flush=True)
        result = evaluate(
            policy,
            episodes=30,
            seed=10000,
            model_path=model,
            algorithm=algorithm,
            duration=20,
        )
        reports[name] = result
        Path(f"reports/sim-{name}.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "mean_score",
                        "mean_survival_s",
                        "mean_collections",
                        "collision_rate",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    gates = {name: compare(reports[name], reports["mpc"]) for name in ("ppo", "dqn")}
    summary = {
        "provenance": "synthetic_unvalidated_mechanics",
        "training_steps": 8192,
        "evaluation_seed": 10000,
        "episodes": 30,
        "duration_s": 20,
        "results": {
            name: {k: v for k, v in r.items() if k != "runs"}
            for name, r in reports.items()
        },
        "candidate_gates": gates,
        "real_phone_policy_promoted": False,
    }
    Path("reports/offline-experiments.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
