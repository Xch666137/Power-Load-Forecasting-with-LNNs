#!/bin/bash

# 电力负荷预测系统 - 实验执行脚本
# 提供简单的命令行接口来运行各种实验

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认配置
DEFAULT_CONFIG="configs/model_config.yaml"
DEFAULT_EXPERIMENT="STSF"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "电力负荷预测系统 - 实验执行脚本"
    echo ""
    echo "用法:"
    echo "  ./run_experiment.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -e, --experiment EXPERIMENT  实验类型: STSF, ModelComparison, all (默认: $DEFAULT_EXPERIMENT)"
    echo "  -c, --config CONFIG_FILE     配置文件路径 (默认: $DEFAULT_CONFIG)"
    echo "  -h, --help                   显示帮助信息"
    echo ""
    echo "示例:"
    echo "  ./run_experiment.sh"
    echo "  ./run_experiment.sh -e STSF"
    echo "  ./run_experiment.sh -e ModelComparison -c configs/model_config.yaml"
    echo "  ./run_experiment.sh -e all"
}

# 运行短时预测实验
run_stsf() {
    local config_file=$1
    print_info "运行短时预测实验..."
    python "$PROJECT_ROOT/run.py" --experiment STSF --config "$config_file"
}

# 运行模型对比实验
run_model_comparison() {
    local config_file=$1
    print_info "运行模型对比实验..."
    python "$PROJECT_ROOT/run.py" --experiment ModelComparison --config "$config_file"
}

# 运行所有实验
run_all() {
    local config_file=$1
    print_info "执行所有实验..."
    run_stsf "$config_file"
    run_model_comparison "$config_file"
}

# 解析命令行参数
EXPERIMENT="$DEFAULT_EXPERIMENT"
CONFIG_FILE="$DEFAULT_CONFIG"

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果配置文件路径是相对路径，则相对于项目根目录
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 显示执行信息
print_info "开始执行实验: $EXPERIMENT"
print_info "使用配置文件: $CONFIG_FILE"

# 根据实验类型执行相应操作
case $EXPERIMENT in
    STSF)
        run_stsf "$CONFIG_FILE"
        ;;
    ModelComparison)
        run_model_comparison "$CONFIG_FILE"
        ;;
    all)
        run_all "$CONFIG_FILE"
        ;;
    *)
        print_error "未知的实验类型: $EXPERIMENT"
        exit 1
        ;;
esac