import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "configs/train_config.yml") -> Dict[str, Any]:
    """
    加载YAML配置文件并转换为字典
    :param config_path: 配置文件的相对路径
    :return: 包含所有配置参数的字典
    """
    try:
        # 定位配置文件（兼容不同运行路径）
        project_root = Path(__file__).parent.parent
        full_path = (project_root / config_path).resolve()
        print(full_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必要配置项
        required_sections = ['data', 'features', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件中缺失必要段落: {section}")
        
        return config
    
    except FileNotFoundError:
        raise RuntimeError(f"未找到配置文件，请确保路径存在: {full_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"配置文件解析错误: {e}")

if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    print("配置加载成功，关键参数:")
    print(f"股票列表: {config['data']['stock_list']}")
    print(f"训练设备: {config['training']['model_params']['device']}")
    print(config)