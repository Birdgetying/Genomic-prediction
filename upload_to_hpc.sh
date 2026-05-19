#!/bin/bash
# ============================================================================
# 自动上传代码到景行超算平台
# 用法:
#   bash upload_to_hpc.sh              # 上传所有 git 追踪文件
#   bash upload_to_hpc.sh --dry-run    # 仅列出将要上传的文件, 不实际传输
#
# 新增文件时: 先添加到 .gitignore 白名单, 然后 git add + git commit
# ============================================================================
set -euo pipefail

# ======================== 配置 (按需修改) ========================
SERVER="login.hpc.example.com"          # 景行平台登录节点 (改成实际地址)
USER="2024110093"                        # 超算用户名
REMOTE_DIR="/storage/public/home/${USER}/genomic_prediction"  # 服务器项目目录
# ==================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# 获取所有 git 追踪文件, 排除仅本地使用的文件
FILES=$(git ls-files | grep -v -E '^(\.gitignore|CLAUDE\.md|upload_to_hpc\.sh)$')

echo "======================================================================"
echo "  基因组预测代码上传 — 景行超算平台"
echo "======================================================================"
echo "  服务器: ${USER}@${SERVER}"
echo "  目标路径: ${REMOTE_DIR}"
echo "  文件数: $(echo "$FILES" | wc -l)"
echo "======================================================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] 以下文件将被上传:"
    for f in $FILES; do echo "  $f"; done
    echo ""
    echo "[DRY RUN] 未实际传输。去掉 --dry-run 执行上传。"
    exit 0
fi

# 确保远程目录存在
ssh "${USER}@${SERVER}" "mkdir -p ${REMOTE_DIR}" || {
    echo "ERROR: 无法连接服务器或创建目录"
    exit 1
}

# 批量上传 — 一次 scp 连接传输所有文件
# shellcheck disable=SC2086
scp $FILES "${USER}@${SERVER}:${REMOTE_DIR}/"

echo ""
echo "======================================================================"
echo "  上传完成"
echo "======================================================================"
