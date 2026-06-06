# 自训 `Qwen3.5-27B` checkpoint 接入说明

已准备好的示例配置文件：

- `configs/yzh_qwen35_27b_4b_mme_demo.json`
- `configs/yzh_qwen35_27b_4b.json`

其中模型配置为：

- `class`: `Qwen3VLChat`
- `model_path`: `/root/autodl-fs/models/20260602_yzh_qwen35_27b_4b/global_step_60`
- `use_vllm`: `true`

后续开始评测时，可以在远端仓库目录执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl
source scripts/vlmeval_env.sh
mkdir -p "$LMUData" /root/autodl-fs/VLMEvalKit/outputs
python scripts/align_hf_data.py --force

python run.py --config configs/yzh_qwen35_27b_4b_mme_demo.json --work-dir /root/autodl-fs/VLMEvalKit/outputs
```

如果使用 `/root/autodl-fs/hf_data` 里已经下载好的数据集，可以先执行：

```bash
python scripts/align_hf_data.py --force
```

该脚本会将官方 TSV 软链到 `$LMUData`，并将本地 JSONL / parquet 转成 VLMEvalKit 可读 TSV。转换得到的非官方格式数据集使用 `_LOCAL` 后缀，避免触发官方数据集 MD5 校验后重新下载。
`ZoomBench_LOCAL` 只使用原图；其 TSV 会拆分 MCQ 的 A-D 选项，并使用专用 `ZoomBenchDataset` 对 MCQ 和数字短答案分别评分。

对齐后的多数据集配置文件为：

```bash
configs/yzh_qwen35_27b_4b.json
```

启动示例：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/yzh_qwen35_27b_4b.json --work-dir /root/autodl-fs/VLMEvalKit/outputs --verbose
```

如果使用 Seed / Ark 作为 judge，先把 `.seed.env` 放到 `/root/autodl-fs/VLMEvalKit/.seed.env`，或通过 `SEED_ENV_FILE` 指定路径，然后执行：

```bash
source scripts/seed_judge_env.sh

CUDA_VISIBLE_DEVICES=0 python run.py \
  --config configs/yzh_qwen35_27b_4b.json \
  --work-dir /root/autodl-fs/VLMEvalKit/outputs \
  --verbose
```

`source scripts/seed_judge_env.sh` 后，`run.py` 会默认使用 `SEED_JUDGE_MODEL` 和 `SEED_JUDGE_ARGS`；只有显式传入 `--judge` 时才会覆盖 Seed 默认 judge。`Seed` judge 主要影响 MCQ 无法用规则抽出选项时的兜底判定，以及那些本身需要 LLM judge 的开放题数据集；`ZoomBench_LOCAL` 的 MCQ 已复用 VLMEvalKit 原生 MCQ 规则和 judge 兜底，open-ended / blank 题会用 Seed judge 判断 Yes/No，Seed 不可用或输出无法解析时才回退数字规则。

如果需要改 benchmark，只需要调整配置文件里的 `data` 字段，例如：

```json
{
  "data": {
    "MME": {
      "class": "ImageYORNDataset",
      "dataset": "MME"
    },
    "MMBench_DEV_EN": {
      "class": "ImageMCQDataset",
      "dataset": "MMBench_DEV_EN"
    },
    "HallusionBench": {
      "class": "ImageYORNDataset",
      "dataset": "HallusionBench"
    }
  }
}
```

如果训练仍在占卡，建议开始评测前显式设置空闲卡：

```bash
CUDA_VISIBLE_DEVICES=4 python run.py --config configs/yzh_qwen35_27b_4b_mme_demo.json --work-dir /root/autodl-fs/VLMEvalKit/outputs
```
