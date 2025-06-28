#!/usr/bin/env python3
# tools/export_optuna_trials.py

import json
from pathlib import Path

import optuna
import pandas as pd

def main():
    # 1. 结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 2. 加载回归模型的 Optuna Study
    study = optuna.load_study(
        study_name="opt_reg",
        storage="sqlite:///regression.db"
    )

    # 3. 导出搜索空间（使用 best_trial.distributions）
    space = {
        name: dist.__class__.__name__ + str(dist.__dict__)
        for name, dist in study.best_trial.distributions.items()
    }
    with open(results_dir / "regression_optuna_search_space.json", "w", encoding="utf-8") as f:
        json.dump(space, f, indent=2, ensure_ascii=False)
    print(f"✅ 搜索空间已保存到 {results_dir/'regression_optuna_search_space.json'}")

    # 4. 导出所有 trials 为 CSV
    df = study.trials_dataframe()
    df.to_csv(results_dir / "regression_optuna_trials.csv", index=False, encoding="utf-8")
    print(f"✅ 完整 trial 日志已保存到 {results_dir/'regression_optuna_trials.csv'}")

    # 5. 导出最优参数到 JSON
    best_params = study.best_params
    with open(results_dir / "best_params_regression.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"✅ 最优参数已保存到 {results_dir/'best_params_regression.json'}")

if __name__ == "__main__":
    main()


















# ======================================   BK   ==================================================


# #!/usr/bin/env python
# # tools/export_optuna_trials.py
#
# from pathlib import Path
# import json
# import optuna
# import pandas as pd
#
# def main():
#     # ────────────────────────────────────────────────────────────────
#     # 1. 加载已存在的 study
#     #    假设您已经通过 storage（如 SQLite）保存了 study
#     # ────────────────────────────────────────────────────────────────
#     study = optuna.load_study(
#         study_name="opt_reg",
#         storage="sqlite:///regression.db"  # 根据实际文件名修改
#     )
#
#     # ────────────────────────────────────────────────────────────────
#     # 2. 导出搜索空间（以 best_trial.distributions 为准）
#     #    distributions: dict[param_name → Distribution 对象]
#     # ────────────────────────────────────────────────────────────────
#     space = {
#         param_name: dist.__class__.__name__ + str(dist.__dict__)
#         for param_name, dist in study.best_trial.distributions.items()
#     }
#     Path("results/optuna_search_space.json") \
#         .write_text(json.dumps(space, indent=2, ensure_ascii=False), encoding="utf-8")
#
#     # ────────────────────────────────────────────────────────────────
#     # 3. 导出所有 trials 为 CSV
#     #    DataFrame 包含 trial_number、value、所有 params、state、datetime_start/end 等列
#     # ────────────────────────────────────────────────────────────────
#     df = optuna.trial._dataframe.build_trials_dataframe(study.trials)
#     # 或者使用官方 API：study.trials_dataframe()
#     # df = study.trials_dataframe()
#     df.to_csv("results/optuna_trials.csv", index=False, encoding="utf-8")
#
#     # ────────────────────────────────────────────────────────────────
#     # 4. 导出最优参数到 JSON
#     # ────────────────────────────────────────────────────────────────
#     best = study.best_params
#     Path("results/best_params_regression.json") \
#         .write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
#
#     print("✅ 已导出：")
#     print("   • results/optuna_search_space.json")
#     print("   • results/optuna_trials.csv")
#     print("   • results/best_params_regression.json")
#
# if __name__ == "__main__":
#     # 确保 results/ 目录存在
#     Path("results").mkdir(exist_ok=True)
#     main()
