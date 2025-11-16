# Copyright (c) Opendatalab. All rights reserved.
import json
import os
from loguru import logger

try:
    import torch
    import torch_npu
except ImportError:
    pass


# 定义配置文件名常量
CONFIG_FILE_NAME = os.getenv('MINERU_TOOLS_CONFIG_JSON', 'mineru.json')


def _find_project_root():
    """
    查找项目根目录（包含 pyproject.toml 的目录）
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # 向上查找，直到找到包含 pyproject.toml 的目录
    search_dir = current_dir
    for _ in range(10):  # 最多向上查找10层
        if os.path.exists(os.path.join(search_dir, 'pyproject.toml')):
            return search_dir
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:  # 已经到达根目录
            break
        search_dir = parent_dir
    
    # 如果找不到，返回当前目录的父目录（假设项目结构是 mineru/utils/config_reader.py）
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))


def read_config():
    """
    从项目目录下的 mineru/mineru.json 读取配置文件
    """
    # 如果环境变量指定了绝对路径，优先使用
    if os.getenv('MINERU_TOOLS_CONFIG_JSON'):
        env_config = os.getenv('MINERU_TOOLS_CONFIG_JSON')
        if os.path.isabs(env_config) and os.path.exists(env_config):
            with open(env_config, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    # 查找项目根目录
    project_root = _find_project_root()
    
    # 配置文件路径：项目根目录/mineru/mineru.json
    config_file = os.path.join(project_root, 'mineru', CONFIG_FILE_NAME)
    
    if not os.path.exists(config_file):
        # logger.warning(f'{config_file} not found, using default configuration')
        return None
    else:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config


def get_s3_config(bucket_name: str):
    """~/magic-pdf.json 读出来."""
    config = read_config()

    bucket_info = config.get('bucket_info')
    if bucket_name not in bucket_info:
        access_key, secret_key, storage_endpoint = bucket_info['[default]']
    else:
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception(f'ak, sk or endpoint not found in {CONFIG_FILE_NAME}')

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {'ak': access_key, 'sk': secret_key, 'endpoint': storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def parse_bucket_key(s3_full_path: str):
    """
    输入 s3://bucket/path/to/my/file.txt
    输出 bucket, path/to/my/file.txt
    """
    s3_full_path = s3_full_path.strip()
    if s3_full_path.startswith("s3://"):
        s3_full_path = s3_full_path[5:]
    if s3_full_path.startswith("/"):
        s3_full_path = s3_full_path[1:]
    bucket, key = s3_full_path.split("/", 1)
    return bucket, key


def get_device():
    device_mode = os.getenv('MINERU_DEVICE_MODE', None)
    if device_mode is not None:
        return device_mode
    else:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            try:
                if torch_npu.npu.is_available():
                    return "npu"
            except Exception as e:
                pass
        return "cpu"


def get_formula_enable(formula_enable):
    formula_enable_env = os.getenv('MINERU_FORMULA_ENABLE')
    formula_enable = formula_enable if formula_enable_env is None else formula_enable_env.lower() == 'true'
    return formula_enable


def get_table_enable(table_enable):
    table_enable_env = os.getenv('MINERU_TABLE_ENABLE')
    table_enable = table_enable if table_enable_env is None else table_enable_env.lower() == 'true'
    return table_enable


def get_latex_delimiter_config():
    config = read_config()
    if config is None:
        return None
    latex_delimiter_config = config.get('latex-delimiter-config', None)
    if latex_delimiter_config is None:
        # logger.warning(f"'latex-delimiter-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return latex_delimiter_config


def get_llm_aided_config():
    config = read_config()
    if config is None:
        return None
    llm_aided_config = config.get('llm-aided-config', None)
    if llm_aided_config is None:
        # logger.warning(f"'llm-aided-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return llm_aided_config


def get_local_models_dir():
    config = read_config()
    if config is None:
        return None
    models_dir = config.get('models-dir')
    if models_dir is None:
        logger.warning(f"'models-dir' not found in {CONFIG_FILE_NAME}, use None as default")
    return models_dir


def get_ai_config():
    """
    从配置文件读取 AI 相关配置
    
    Returns:
        dict: 包含以下键的字典：
            - model: 模型名称（如 "Qwen/Qwen2.5-1.5B-Instruct"）
            - quantization: 量化方法（"fp8", "int8", "int4" 或 None）
        如果配置文件中没有相关配置，返回 None
    """
    config = read_config()
    if config is None:
        return None
    
    ai_config = config.get('ai-config', None)
    if ai_config is None:
        return None
    
    result = {}
    if 'model' in ai_config:
        result['model'] = ai_config['model']
    if 'quantization' in ai_config:
        result['quantization'] = ai_config['quantization']
    
    return result if result else None