import os
import argparse
from pytdx.reader import TdxLCMinBarReader, TdxDailyBarReader
import pandas as pd
from pathlib import Path
import logging

# 配置日志记录
logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def validate_symbol(symbol: str) -> bool:
    """验证股票代码是否符合处理规则"""
    valid_prefix = symbol.startswith(('6', '3'))  # 沪市或深市
    reject_prefix = symbol.startswith('68')        # 排除科创板
    return valid_prefix and not reject_prefix

def get_extension_mapping(suffix: str) -> str:
    """获取文件扩展名映射"""
    return {
        'day': 'day',
        'lc1': 'm1',
        'lc5': 'm5'
    }.get(suffix.lstrip('.'), 'unknown')

def process_data_file(file_path: Path, data_root: Path) -> None:
    """处理单个数据文件"""
    try:
        # 解析文件信息
        file_suffix = file_path.suffix[1:]  # 去除点号的扩展名
        file_stem = file_path.stem          # 无扩展名的文件名
        
        # 过滤北京交易所数据
        if file_stem[:2].upper() == 'BJ':
            logger.debug(f"跳过北京交易所文件: {file_path}")
            return

        # 提取股票代码并验证
        symbol = file_stem[2:]
        if not validate_symbol(symbol):
            logger.debug(f"跳过不符合规则的股票代码: {symbol}")
            return

        # 初始化对应的读取器
        reader = TdxDailyBarReader() if file_suffix == 'day' else TdxLCMinBarReader()
        
        # 读取并处理数据
        df = reader.get_df(str(file_path))
        process_financial_data(df)

        # 生成输出路径
        ext_type = get_extension_mapping(file_suffix)
        save_paths = {
            'csv': data_root / 'csv' / ext_type / f"{symbol}.csv",
            'pkl': data_root / 'pkl' / ext_type / f"{symbol}.pkl"
        }

        # 保存处理结果
        save_paths['csv'].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_paths['csv'], index=True, encoding='utf_8_sig')
        df.to_pickle(save_paths['pkl'])
        
        logger.info(f"成功处理: {symbol} ({ext_type})")

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
        raise

def process_financial_data(df: pd.DataFrame) -> None:
    """处理金融数据精度和类型"""
    numeric_cols = ['open', 'high', 'low', 'close', 'amount']
    df[numeric_cols] = df[numeric_cols].round(2)
    df['volume'] = df['volume'].astype('int64')
    # df['date'] = pd.to_datetime(df['date'])

def setup_directories(base_path: Path) -> None:
    """创建所有需要的输出目录"""
    dir_structure = ['csv/day', 'csv/m1', 'csv/m5',
                    'pkl/day', 'pkl/m1', 'pkl/m5']
    for subdir in dir_structure:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)

def validate_source_dir(source_dir):
    """验证源目录及其子目录是否存在"""
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"指定的源目录不存在: {source_dir}")
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"指定的路径不是目录: {source_dir}")

    # 检查必需的子目录
    required_subdirs = ['bj', 'sh', 'sz']
    missing_subdirs = []
    
    for subdir in required_subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        if not os.path.exists(subdir_path):
            missing_subdirs.append(subdir)
    
    if missing_subdirs:
        raise FileNotFoundError(
            f"源目录中缺少必需的子目录: {missing_subdirs}\n"
            f"请确保 {source_dir} 包含以下子目录: {required_subdirs}"
        )
    
    print(f"✓ 源目录验证通过: {source_dir}")
    print("✓ 包含所有必需的子目录:", required_subdirs)
if __name__ == '__main__':
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent     
    # 参数解析
    parser = argparse.ArgumentParser(
        description='通达信数据处理工具.处理通达信数据为',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值
    )
    parser.add_argument('--source_dir', 
                        type=str, 
                        default='/opt/wangf/tdx/data/vipdoc',
                        help='通达信数据目录（必须包含bj/sh/sz子目录）')
    parser.add_argument('--output_dir', 
                        type=str, 
                        default=f'{project_root}/raw_data',
                        help='处理结果输出目录,支持相对目录')
    
    args = parser.parse_args()
    
    try:
        validate_source_dir(args.source_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"错误: {e}")
        parser.print_help()  # 打印用法帮助
        exit(1) 
    
    # 配置路径参数
    source_dir = Path(args.source_dir)        
    output_root = Path(args.output_dir).resolve()
    
    # 自动创建输出目录（包括所有父目录）
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录已准备: {output_root}")    

    # 初始化目录结构
    setup_directories(output_root)
    
    # # 获取待处理文件列表
    # file_patterns = ['*.day', '*.lc1', '*.lc5']
    # data_files = []
    # for pattern in file_patterns:
    #     data_files.extend(source_dir.rglob(pattern))
    
    # logger.info(f"发现 {len(data_files)} 个待处理文件")
    
    # # 处理每个数据文件
    # for data_file in data_files:
    #     process_data_file(data_file, output_root)
    
    # logger.info("数据处理任务完成")