#!/bin/bash

# 电力负荷预测系统 - 远程GPU训练脚本
# 通过SSH连接到远程服务器并在GPU上运行实验

# 默认配置
DEFAULT_CONFIG="configs/model_config.yaml"
DEFAULT_EXPERIMENT="STSF"
DEFAULT_REMOTE_USER="user"
DEFAULT_REMOTE_HOST="your-server-ip"
DEFAULT_REMOTE_PATH="/home/user/power-load-forecasting"
DEFAULT_SSH_KEY=""

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
    echo "电力负荷预测系统 - 远程GPU训练脚本"
    echo ""
    echo "用法:"
    echo "  ./remote_train.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -e, --experiment EXPERIMENT  实验类型: STSF, ModelComparison (默认: $DEFAULT_EXPERIMENT)"
    echo "  -c, --config CONFIG_FILE     配置文件路径 (默认: $DEFAULT_CONFIG)"
    echo "  -u, --user REMOTE_USER       远程服务器用户名 (默认: $DEFAULT_REMOTE_USER)"
    echo "  -H, --host REMOTE_HOST       远程服务器地址 (默认: $DEFAULT_REMOTE_HOST)"
    echo "  -p, --path REMOTE_PATH       远程服务器项目路径 (默认: $DEFAULT_REMOTE_PATH)"
    echo "  -i, --identity SSH_KEY       SSH私钥文件路径 (可选)"
    echo "  -h, --help                   显示帮助信息"
    echo ""
    echo "示例:"
    echo "  ./remote_train.sh"
    echo "  ./remote_train.sh -e STSF -H 192.168.1.100 -u ubuntu"
    echo "  ./remote_train.sh -e ModelComparison -H 192.168.1.100 -u ubuntu -p /home/ubuntu/projects/power-load-forecasting"
    echo "  ./remote_train.sh -e STSF -H 192.168.1.100 -u ubuntu -i ~/.ssh/id_rsa_custom"
}

# 运行远程训练
run_remote_training() {
    local experiment=$1
    local config_file=$2
    local remote_user=$3
    local remote_host=$4
    local remote_path=$5
    local ssh_key=$6
    
    print_info "准备在远程服务器上运行实验..."
    print_info "服务器地址: $remote_host"
    print_info "用户名: $remote_user"
    print_info "项目路径: $remote_path"
    print_info "实验类型: $experiment"
    print_info "配置文件: $config_file"
    
    # 构建SSH命令
    local ssh_cmd="ssh"
    if [ -n "$ssh_key" ]; then
        ssh_cmd="ssh -i $ssh_key"
    fi
    
    # 检查远程服务器连接
    print_info "检查远程服务器连接..."
    if ! $ssh_cmd -o ConnectTimeout=10 "$remote_user@$remote_host" "echo '连接成功'" >/dev/null 2>&1; then
        print_error "无法连接到远程服务器 $remote_user@$remote_host"
        exit 1
    fi
    
    # 检查远程项目目录是否存在
    print_info "检查远程项目目录..."
    if ! $ssh_cmd "$remote_user@$remote_host" "test -d $remote_path" >/dev/null 2>&1; then
        print_error "远程服务器上不存在项目目录: $remote_path"
        exit 1
    fi
    
    # 检查配置文件是否存在于远程服务器
    local remote_config_path="$remote_path/${config_file#*/}"
    print_info "检查远程配置文件..."
    if ! $ssh_cmd "$remote_user@$remote_host" "test -f $remote_config_path" >/dev/null 2>&1; then
        print_warning "远程服务器上不存在配置文件: $remote_config_path"
        print_info "将本地配置文件同步到远程服务器..."
        scp_cmd="scp"
        if [ -n "$ssh_key" ]; then
            scp_cmd="scp -i $ssh_key"
        fi
        
        if ! $scp_cmd "$config_file" "$remote_user@$remote_host:$remote_config_path"; then
            print_error "同步配置文件失败"
            exit 1
        fi
        print_info "配置文件同步完成"
    fi
    
    # 运行远程实验
    print_info "正在远程服务器上运行实验..."
    local remote_cmd="cd $remote_path && python run.py --experiment $experiment --config $remote_config_path"
    $ssh_cmd "$remote_user@$remote_host" "$remote_cmd"
    
    if [ $? -eq 0 ]; then
        print_info "远程实验执行完成"
    else
        print_error "远程实验执行失败"
        exit 1
    fi
}

# 解析命令行参数
EXPERIMENT="$DEFAULT_EXPERIMENT"
CONFIG_FILE="$DEFAULT_CONFIG"
REMOTE_USER="$DEFAULT_REMOTE_USER"
REMOTE_HOST="$DEFAULT_REMOTE_HOST"
REMOTE_PATH="$DEFAULT_REMOTE_PATH"
SSH_KEY="$DEFAULT_SSH_KEY"

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
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -H|--host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -p|--path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        -i|--identity)
            SSH_KEY="$2"
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

# 检查本地配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "本地配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查必要参数
if [ "$REMOTE_HOST" = "$DEFAULT_REMOTE_HOST" ]; then
    print_error "请指定远程服务器地址 (-H, --host)"
    show_help
    exit 1
fi

# 运行远程训练
run_remote_training "$EXPERIMENT" "$CONFIG_FILE" "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_PATH" "$SSH_KEY"