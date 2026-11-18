#!/usr/bin/env bash
# DDA5001 Part I — 第二阶段：看完默认曲线、选定学习率后运行。
# 使用方法：把本文件放在机房 ~/work/project1，先填写下面三个 CHANGE_ME。
# 填写后运行 bash run_alternatives.sh
set -Eeuo pipefail

# ====== 在看过第一阶段的曲线后，只修改这三个学习率 ======
ADAMW_OTHER_LR="CHANGE_ME"   # 例如 0.00015（仅示例，不是老师指定）
SGDM_OTHER_LR="CHANGE_ME"    # 例如 0.05（仅示例）
MUON_OTHER_LR="CHANGE_ME"    # 例如 0.005（仅示例）
# ========================================================

cd "$(dirname "$(realpath "$0")")"
source /opt/dda5001/venv/bin/activate
RUNS_DIR="runs/p1_comparison"

# 不填写、填写非数值、负数、或者与默认学习率相同，都会在训练开始前报错。
python - "$ADAMW_OTHER_LR" "$SGDM_OTHER_LR" "$MUON_OTHER_LR" <<'PY'
import math, sys
names = ['AdamW', 'SGDM', 'Muon']
defaults = [0.0003, 0.1, 0.01]
for name, value, default in zip(names, sys.argv[1:], defaults):
    try:
        x = float(value)
    except ValueError:
        raise SystemExit(f'请先把 {name} 的 CHANGE_ME 改成自选学习率（正数）: {value}')
    if not math.isfinite(x) or x <= 0 or x == default:
        raise SystemExit(f'{name} 的自选学习率必须为正数，且不同于默认值 {default}: {value}')
    print(f'{name} 自选学习率: {x:g}（默认 {default:g}）')
PY

# 第二阶段必须在第一阶段三组已完成并已恢复到同一目录后运行。
python - "$RUNS_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name in ('adamw-default','sgdm-default','muon-default'):
    folder = root / name
    summary = folder / 'summary.json'
    if not summary.is_file() or json.loads(summary.read_text(encoding='utf-8')).get('status') != 'completed':
        raise SystemExit(f'缺少已完成的第一阶段实验：{folder}。若重启过，请先恢复备份。')
print('第一阶段三个默认实验存在，第二阶段可以开始。')
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

run_one adamw-other adamw learning_rate "$ADAMW_OTHER_LR"
run_one sgdm-other sgdm sgdm_learning_rate "$SGDM_OTHER_LR"
run_one muon-other muon muon_learning_rate "$MUON_OTHER_LR"

python plot.py "--runs-dir=$RUNS_DIR"
python sample.py "--checkpoint=$RUNS_DIR/adamw-default/last.pt" --tokens=400

# 每次生成一个带时间戳的新提交包，不覆盖旧文件；报告 PDF 需手动加进 ZIP。
ARCHIVE="part1-results-$(date +%Y%m%d-%H%M%S).zip"
python export_results.py "--runs-dir=$RUNS_DIR" "--output=$ARCHIVE"

printf '\n六组实验全部完成，曲线已生成，文本样本已保存到 adamw-default/samples.txt。\n'
printf '结果目录：%s\n提交素材 ZIP：%s（仍需加入报告 PDF）\n' "$RUNS_DIR" "$ARCHIVE"
printf '务必单独备份六组的 last.pt（以及需要保留的 best.pt）；提交 ZIP 不包含 .pt。\n'
