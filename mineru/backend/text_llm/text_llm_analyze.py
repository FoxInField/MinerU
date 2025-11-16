"""
文本LLM模型管理，支持多种后端（类似VLM架构）
"""
import os
import time
from typing import Optional
from loguru import logger
from packaging import version

from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.utils.config_reader import get_device
from mineru.backend.vlm.utils import set_default_gpu_memory_utilization, check_fp8_support


class TextLLMSingleton:
    """文本LLM模型单例，支持多种后端"""
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        backend: str,
        model_path: str | None = None,
        server_url: str | None = None,
        **kwargs,
    ):
        """
        获取文本LLM模型
        
        Args:
            backend: 后端类型，支持 'transformers', 'vllm-engine', 'vllm-async-engine', 'http-client'
            model_path: 模型路径，如果为None则自动下载
            server_url: HTTP客户端模式的服务器地址
            **kwargs: 其他参数
        
        Returns:
            模型实例或客户端
        """
        # 将量化参数包含在 key 中，确保不同的量化配置使用不同的模型实例
        quantization = kwargs.get("quantization", None)
        key = (backend, model_path, server_url, quantization)
        if key not in self._models:
            start_time = time.time()
            model = None
            tokenizer = None
            vllm_llm = None
            vllm_async_llm = None
            
            if backend in ['transformers', 'vllm-engine', 'vllm-async-engine'] and not model_path:
                # 使用 kwargs 中的 model 参数，如果也没有则使用默认模型
                model_name = kwargs.get("model", None)
                if model_name:
                    # 使用用户指定的模型名称
                    model_path = auto_download_and_get_model_root_path(model_name, "text_llm")
                else:
                    # 使用默认模型
                    model_path = auto_download_and_get_model_root_path("/", "text_llm")
            
            if backend == "transformers":
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                    from transformers import __version__ as transformers_version
                except ImportError:
                    raise ImportError("Please install transformers to use the transformers backend.")
                
                if version.parse(transformers_version) >= version.parse("4.56.0"):
                    dtype_key = "dtype"
                else:
                    dtype_key = "torch_dtype"
                
                device = get_device()
                import torch
                import gc
                
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # 检查量化参数
                quantization = kwargs.pop("quantization", None)
                use_quantization = quantization in ["int8", "int4"]
                
                # 对于 GPU，检查显存并决定加载方式
                if device.startswith("cuda") and torch.cuda.is_available():
                    # 获取设备索引
                    device_str = str(device)
                    device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                    
                    # 清理显存缓存
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()
                    
                    # 按照 CLI 的逻辑，在加载 AI 模型之前检查显存，如果不足则清理所有模型
                    try:
                        from mineru.utils.model_utils import (
                            unload_pipeline_models,
                            clean_memory,
                            get_vram,
                            get_allocated_vram,
                            estimate_llm_vram_requirement,
                            _move_model_to_cpu,
                        )
                        
                        # 获取模型名称（用于判断显存需求）
                        ai_model = kwargs.get("model", "Qwen/Qwen2.5-3B-Instruct")
                        
                        # 检查显存是否足够
                        total_vram = get_vram(device)
                        allocated_vram = get_allocated_vram(device)
                        
                        if total_vram and allocated_vram is not None:
                            available_vram = total_vram - allocated_vram
                            ai_vram_required = estimate_llm_vram_requirement(ai_model, "transformers", quantization)
                            
                            # 需要至少 1.2 倍的模型大小（考虑加载时的临时显存）
                            required_vram = ai_vram_required * 1.2
                            
                            if available_vram < required_vram:
                                logger.info(
                                    f"Low VRAM detected: Available={available_vram:.2f}GB, "
                                    f"Required={required_vram:.2f}GB (model={ai_vram_required:.2f}GB). "
                                    f"Unloading all models before loading AI model..."
                                )
                                
                                # 清理 pipeline 模型
                                unload_pipeline_models(device)
                                
                                # 清理 VLM 模型
                                try:
                                    from mineru.backend.vlm.vlm_analyze import ModelSingleton as VLMModelSingleton
                                    
                                    vlm_singleton = VLMModelSingleton()
                                    if hasattr(vlm_singleton, '_models') and vlm_singleton._models:
                                        model_count = len(vlm_singleton._models)
                                        for key, model in list(vlm_singleton._models.items()):
                                            try:
                                                _move_model_to_cpu(model)
                                                del model
                                            except Exception as e:
                                                logger.debug(f"Error moving VLM model {key} to CPU: {e}")
                                        vlm_singleton._models.clear()
                                        logger.info(f"Cleared {model_count} VLM models from ModelSingleton")
                                        
                                        # 强制垃圾回收
                                        for _ in range(3):
                                            gc.collect()
                                        clean_memory(device)
                                except Exception as e:
                                    logger.debug(f"Failed to unload VLM models: {e}")
                                
                                # 清理后再次检查显存
                                allocated_after = get_allocated_vram(device)
                                if total_vram and allocated_after is not None:
                                    available_after = total_vram - allocated_after
                                    logger.info(f"After cleanup, available VRAM: {available_after:.2f}GB")
                                    if available_after < ai_vram_required * 0.8:
                                        logger.warning(
                                            f"Available VRAM ({available_after:.2f}GB) may still be insufficient "
                                            f"for AI model ({ai_vram_required:.2f}GB). "
                                            f"Consider using quantization or a smaller model."
                                        )
                            else:
                                logger.info("Clearing GPU memory before loading AI model...")
                                clean_memory(device)
                                logger.info("GPU memory cleared.")
                        else:
                            # 无法获取显存信息，保守起见清理所有模型
                            logger.warning("Cannot get GPU memory info, unloading all models to be safe...")
                            unload_pipeline_models(device)
                            
                            # 清理 VLM 模型
                            try:
                                from mineru.backend.vlm.vlm_analyze import ModelSingleton as VLMModelSingleton
                                
                                vlm_singleton = VLMModelSingleton()
                                if hasattr(vlm_singleton, '_models') and vlm_singleton._models:
                                    for key, model in list(vlm_singleton._models.items()):
                                        try:
                                            _move_model_to_cpu(model)
                                            del model
                                        except Exception as e:
                                            logger.debug(f"Error moving VLM model {key} to CPU: {e}")
                                    vlm_singleton._models.clear()
                                    
                                    for _ in range(3):
                                        gc.collect()
                                    clean_memory(device)
                            except Exception as e:
                                logger.debug(f"Failed to unload VLM models: {e}")
                                
                    except Exception as e:
                        logger.warning(f"Failed to check/unload models: {e}, continuing with model loading...")
                    
                    # 检查可用显存（在清理后重新检查）
                    total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
                    reserved_vram = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
                    available_vram = total_vram - reserved_vram
                    
                    # 如果启用了量化，使用 bitsandbytes
                    if use_quantization:
                        try:
                            import bitsandbytes as bnb
                        except ImportError:
                            raise ImportError(
                                f"bitsandbytes is required for {quantization} quantization. "
                                "Please install it with: pip install bitsandbytes"
                            )
                        
                        logger.info(f"Loading model with {quantization.upper()} quantization using bitsandbytes...")
                        
                        # 配置量化参数
                        if quantization == "int8":
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0,
                            )
                        elif quantization == "int4":
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                            )
                        else:
                            quantization_config = None
                        
                        # 使用量化加载模型，添加 low_cpu_mem_usage 以减少峰值显存
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto",  # 使用 auto 让 transformers 自动管理，更安全
                            low_cpu_mem_usage=True,  # 关键：减少加载时的峰值显存
                            trust_remote_code=True,
                        )
                        logger.info(f"Model loaded with {quantization.upper()} quantization on {device}")
                    else:
                        # 如果可用显存小于 4GB，使用保守加载方式（先加载到 CPU）
                        if available_vram < 4.0:
                            logger.info(f"Low VRAM detected ({available_vram:.2f}GB), loading model to CPU first...")
                            
                            # 根据 transformers 版本使用不同的参数名
                            if version.parse(transformers_version) >= version.parse("4.56.0"):
                                dtype_param = {"dtype": torch.float16}
                            else:
                                dtype_param = {"torch_dtype": torch.float16}
                            
                            # 加载到 CPU
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                **dtype_param,
                                low_cpu_mem_usage=True,
                                device_map="cpu",
                                trust_remote_code=True,
                            )
                            
                            # 如果模型是 FP32，转换为 FP16
                            if next(model.parameters()).dtype == torch.float32:
                                logger.info("Converting model from FP32 to FP16...")
                                model = model.half()
                            
                            # 计算模型大小（只计算参数，不包括 buffers）
                            dtype_size_map = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}
                            model_size = sum(
                                p.numel() * dtype_size_map.get(p.dtype, 2) 
                                for p in model.parameters()
                            ) / (1024 ** 3)
                            
                            # 检查显存是否足够
                            required_vram = model_size * 1.2
                            if available_vram < required_vram:
                                raise RuntimeError(
                                    f"Insufficient GPU memory: Available {available_vram:.2f}GB, "
                                    f"Model requires {required_vram:.2f}GB (model size: {model_size:.2f}GB). "
                                    f"Consider using --ai-quantization int8 or int4 to reduce memory usage."
                                )
                            
                            # 移动到 GPU
                            logger.info(f"Moving model to GPU (size: {model_size:.2f}GB)...")
                            model = model.to(device)
                            torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            # 显存充足，使用 device_map
                            model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map={"": device},
                                **{dtype_key: "auto"},
                                trust_remote_code=True,
                            )
                else:
                    # CPU 或其他设备（量化不支持 CPU）
                    if use_quantization:
                        logger.warning(f"{quantization.upper()} quantization is not supported on {device}. Loading model without quantization.")
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **{dtype_key: "auto"},
                        trust_remote_code=True,
                    )
                    if device != "cpu":
                        model = model.to(device)
                
            elif backend == "vllm-engine":
                try:
                    import vllm
                    import torch
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-engine backend.")
                
                if "gpu_memory_utilization" not in kwargs:
                    kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                if "model" not in kwargs:
                    kwargs["model"] = model_path
                
                # 检查量化参数
                quantization = kwargs.pop("quantization", None)
                if quantization:
                    device = get_device()
                    if device.startswith("cuda"):
                        device_str = str(device)
                        device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                        
                        if quantization == "fp8":
                            if check_fp8_support(device_idx):
                                kwargs["quantization"] = "fp8"
                                logger.info("FP8 quantization enabled for vLLM engine")
                            else:
                                logger.warning("FP8 quantization requested but GPU does not support it (requires compute capability >= 8.9). Using default precision.")
                                kwargs.pop("quantization", None)
                        elif quantization in ["int8", "int4"]:
                            kwargs["quantization"] = quantization
                            logger.info(f"{quantization.upper()} quantization enabled for vLLM engine")
                        else:
                            logger.warning(f"Unknown quantization type: {quantization}. Using default precision.")
                            kwargs.pop("quantization", None)
                    else:
                        logger.warning(f"{quantization.upper()} quantization only supported on CUDA devices. Using default precision.")
                        kwargs.pop("quantization", None)
                
                vllm_llm = vllm.LLM(**kwargs)
                
            elif backend == "vllm-async-engine":
                try:
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.v1.engine.async_llm import AsyncLLM
                    import torch
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-async-engine backend.")
                
                if "gpu_memory_utilization" not in kwargs:
                    kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                if "model" not in kwargs:
                    kwargs["model"] = model_path
                
                # 检查量化参数
                quantization = kwargs.pop("quantization", None)
                if quantization:
                    device = get_device()
                    if device.startswith("cuda"):
                        device_str = str(device)
                        device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
                        
                        if quantization == "fp8":
                            if check_fp8_support(device_idx):
                                kwargs["quantization"] = "fp8"
                                logger.info("FP8 quantization enabled for vLLM async engine")
                            else:
                                logger.warning("FP8 quantization requested but GPU does not support it (requires compute capability >= 8.9). Using default precision.")
                                kwargs.pop("quantization", None)
                        elif quantization in ["int8", "int4"]:
                            kwargs["quantization"] = quantization
                            logger.info(f"{quantization.upper()} quantization enabled for vLLM async engine")
                        else:
                            logger.warning(f"Unknown quantization type: {quantization}. Using default precision.")
                            kwargs.pop("quantization", None)
                    else:
                        logger.warning(f"{quantization.upper()} quantization only supported on CUDA devices. Using default precision.")
                        kwargs.pop("quantization", None)
                
                vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
            
            elif backend == "http-client":
                # HTTP客户端模式，不需要加载模型
                pass
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            
            self._models[key] = {
                "backend": backend,
                "model": model,
                "tokenizer": tokenizer,
                "vllm_llm": vllm_llm,
                "vllm_async_llm": vllm_async_llm,
                "server_url": server_url,
            }
            
            elapsed = round(time.time() - start_time, 2)
            logger.info(f"get {backend} text LLM model cost: {elapsed}s")
        
        return self._models[key]


def build_prompt(
    text: str,
    prompt_template: str | None = None,
    **kwargs,
) -> str:
    """
    构建提示词
    
    Args:
        text: 输入文本（如果prompt_template中有{text}占位符，会被替换）
        prompt_template: 提示词模板，支持多个占位符：
            - {text}: 会被text参数替换
            - {resume_text}: 需要从kwargs中获取
            - {json_template}: 需要从kwargs中获取
            - 其他自定义占位符：从kwargs中获取同名参数
        **kwargs: 其他参数，包括：
            - resume_text: 简历文本（用于多占位符模板）
            - json_template: JSON模板（用于多占位符模板）
            - 其他自定义占位符的值
    
    Returns:
        构建好的完整提示词
    """
    if prompt_template:
        # 检查模板中实际使用了哪些占位符
        import re
        used_placeholders = set(re.findall(r'\{(\w+)\}', prompt_template))
        
        # 准备占位符字典，只设置模板中实际使用的占位符
        template_vars = {}
        
        # 如果模板中使用了 {text}，则设置 text
        if "text" in used_placeholders:
            template_vars["text"] = text
        
        # 从kwargs中提取可能的占位符值
        for key, value in kwargs.items():
            if key not in ["model", "timeout", "quantization"]:  # 排除这些特殊参数
                # 只有当模板中实际使用了这个占位符时才添加
                if key in used_placeholders:
                    template_vars[key] = value
        
        try:
            prompt = prompt_template.format(**template_vars)
        except KeyError as e:
            # 如果缺少必需的占位符，提供友好的错误信息
            missing = str(e).strip("'")
            raise ValueError(
                f"提示词模板中使用了占位符 {{{missing}}}，但未提供对应的值。"
                f"请通过kwargs参数提供，例如：build_prompt(..., {missing}='value')"
            )
    else:
        prompt = f"你是一名资深HR，收到了以下简历：\n\n{text}\n\n请总结求职者的优势和不足，给出是否进入下一轮面试的建议。"
    
    # 输出构建好的提示词（用于调试）
    logger.info("=" * 80)
    logger.info("构建好的提示词 (Prompt):")
    logger.info("=" * 80)
    logger.info(prompt)
    logger.info("=" * 80)
    logger.info(f"提示词长度: {len(prompt)} 字符")
    
    return prompt


async def generate_text_async(
    prompt: str,
    backend: str = "http-client",
    model_path: str | None = None,
    server_url: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    quantization: str | None = None,
    **kwargs,
) -> str:
    """
    异步生成文本
    
    Args:
        prompt: 完整的提示词
        backend: 后端类型
        model_path: 模型路径
        server_url: HTTP服务器地址（仅http-client模式）
        max_tokens: 最大token数
        temperature: 温度参数
        quantization: 量化方法
        **kwargs: 其他参数，包括：
            - model: 模型名称
            - timeout: 超时时间（仅http-client模式）
    
    Returns:
        生成的文本
    """
    
    model_kwargs = {}
    if quantization:
        model_kwargs["quantization"] = quantization
    if "model" in kwargs:
        model_kwargs["model"] = kwargs["model"]
    if "timeout" in kwargs:
        model_kwargs["timeout"] = kwargs["timeout"]
    
    model_client = TextLLMSingleton().get_model(backend, model_path, server_url, **model_kwargs)
    
    if backend == "http-client":
        # HTTP客户端模式
        import httpx
        if server_url is None:
            server_url = "http://localhost:30001"
        
        api_url = f"{server_url}/v1/chat/completions"
        payload = {
            "model": kwargs.get("model", "Qwen/Qwen2.5-3B-Instruct"),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        timeout = kwargs.get("timeout", 300.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format: {result}")
    
    elif backend == "transformers":
        # Transformers模式
        import torch
        model = model_client["model"]
        tokenizer = model_client["tokenizer"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_text
    
    elif backend == "vllm-async-engine":
        # vLLM异步引擎模式
        vllm_async_llm = model_client["vllm_async_llm"]
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        outputs = await vllm_async_llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    elif backend == "vllm-engine":
        # vLLM同步引擎模式（在异步函数中使用需要在线程池中运行）
        import asyncio
        vllm_llm = model_client["vllm_llm"]
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 在线程池中运行同步调用
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: vllm_llm.generate([prompt], sampling_params)
        )
        return outputs[0].outputs[0].text
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")

