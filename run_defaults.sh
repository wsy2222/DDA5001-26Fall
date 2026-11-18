#!/usr/bin/env bash
# DDA5001 Part I — 第一阶段：顺序训练三个默认学习率。
# 使用方法：把本文件放在机房 ~/work/project1，然后运行 bash run_defaults.sh
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"
source /opt/dda5001/venv/bin/activate

RUNS_DIR="runs/p1_comparison"
mkdir -p "$RUNS_DIR"

printf '\n[1/3] 代码与显卡检查...\n'
python check_solution.py
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU 不可用，请先检查机房课程环境'
assert hasattr(torch.optim, 'Muon'), 'PyTorch 缺少原生 torch.optim.Muon'
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)
PY

run_one() {
  local name="$1" optimizer="$2" lr_key="$3" lr="$4"
  local out="$RUNS_DIR/$name"
  if [ -e "$out" ]; then
    # 已完成且参数一致：安全跳过；不完整或参数不一致：停止，绝不覆盖。
    python - "$out" "$optimizer" "$lr_key" "$lr" <<'PY_CHECK'
import json, math, pathlib, sys
folder, optimizer, lr_key, lr = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
try:
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert config['optimizer_name'] == optimizer and math.isclose(config[lr_key], lr, rel_tol=1e-12)
    assert summary.get('status') == 'completed' and summary.get('tokens') == 92160000
    assert (folder / 'last.pt').is_file()
except Exception as exc:
    raise SystemExit(f'已存在但不符合预期：{folder}。脚本不会覆盖，请先调查或将失败目录移到 runs/p1_failed/。详情: {exc}')
print(f'已存在完整且配置一致的实验，安全跳过: {folder}')
PY_CHECK
    return 0
  fi
  printf '\n========== 开始 %s (%s=%s) ==========\n' "$name" "$lr_key" "$lr"
  python -u train.py config/train_shakespeare_char.py \
    "--optimizer_name=$optimizer" \
    "--${lr_key}=$lr" \
    "--out_dir=$out"
  python - "$out" <<'PY'
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
assert summary.get('status') == 'completed', f'训练没有完成: {folder}'
assert summary.get('tokens') == 92160000, f'Token 预算不完整: {folder}'
assert (folder / 'last.pt').is_file(), f'缺少 last.pt: {folder}'
print('已确认完成:', folder, '| endpoint val_loss =', summary['endpoint_scores']['loss'])
PY
  python plot.py "--run-dir=$out"
  printf '========== 结束 %s ==========\n' "$name"
}

# 顺序执行：前一组完整结束、检查成功后，才会启动下一组。
run_one adamw-default adamw learning_rate 0.0003
run_one sgdm-default sgdm sgdm_learning_rate 0.1
run_one muon-default muon muon_learning_rate 0.01

python plot.py "--runs-dir=$RUNS_DIR"
printf '\n三个默认实验均已完成。结果位于：%s\n' "$RUNS_DIR"
printf '现在停止自动训练；请查看各组 loss_vs_tokens.png 与 loss_vs_train_seconds.png。\n'
printf '选好三个新学习率后，修改 run_alternatives.sh 顶部的三个值，再运行第二阶段。\n'
printf '如果要离开机房，请在关机或注销前把这三组结果（必要时含 .pt）备份到外部存储！\n'
