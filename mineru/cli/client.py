# Copyright (c) Opendatalab. All rights reserved.
import os
import asyncio
import click
from pathlib import Path
from typing import Optional
from loguru import logger

from mineru.utils.check_sys_env import is_mac_os_version_supported
from mineru.utils.cli_parser import arg_parse
from mineru.utils.config_reader import get_device
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.utils.model_utils import get_vram
from ..version import __version__
from .common import do_parse, read_fn, pdf_suffixes, image_suffixes


backends = ['pipeline', 'vlm-transformers', 'vlm-vllm-engine', 'vlm-http-client']
if is_mac_os_version_supported():
    backends.append("vlm-mlx-engine")

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.version_option(__version__,
                      '--version',
                      '-v',
                      help='display the version and exit')
@click.option(
    '-p',
    '--path',
    'input_path',
    type=click.Path(exists=True),
    required=True,
    help='local filepath or directory. support pdf, png, jpg, jpeg files',
)
@click.option(
    '-o',
    '--output',
    'output_dir',
    type=click.Path(),
    required=True,
    help='output local directory',
)
@click.option(
    '-m',
    '--method',
    'method',
    type=click.Choice(['auto', 'txt', 'ocr']),
    help="""\b
    the method for parsing pdf:
      auto: Automatically determine the method based on the file type.
      txt: Use text extraction method.
      ocr: Use OCR method for image-based PDFs.
    Without method specified, 'auto' will be used by default.
    Adapted only for the case where the backend is set to 'pipeline'.""",
    default='auto',
)
@click.option(
    '-b',
    '--backend',
    'backend',
    type=click.Choice(backends),
    help="""\b
    the backend for parsing pdf:
      pipeline: More general.
      vlm-transformers: More general, but slower.
      vlm-mlx-engine: Faster than transformers.
      vlm-vllm-engine: Faster(engine).
      vlm-http-client: Faster(client).
    Without method specified, pipeline will be used by default.""",
    default='pipeline',
)
@click.option(
    '-l',
    '--lang',
    'lang',
    type=click.Choice(['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'th', 'el',
                       'latin', 'arabic', 'east_slavic', 'cyrillic', 'devanagari']),
    help="""
    Input the languages in the pdf (if known) to improve OCR accuracy.
    Without languages specified, 'ch' will be used by default.
    Adapted only for the case where the backend is set to "pipeline".
    """,
    default='ch',
)
@click.option(
    '-u',
    '--url',
    'server_url',
    type=str,
    help="""
    When the backend is `vlm-http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """,
    default=None,
)
@click.option(
    '-s',
    '--start',
    'start_page_id',
    type=int,
    help='The starting page for PDF parsing, beginning from 0.',
    default=0,
)
@click.option(
    '-e',
    '--end',
    'end_page_id',
    type=int,
    help='The ending page for PDF parsing, beginning from 0.',
    default=None,
)
@click.option(
    '-f',
    '--formula',
    'formula_enable',
    type=bool,
    help='Enable formula parsing. Default is True. Adapted only for the case where the backend is set to "pipeline".',
    default=True,
)
@click.option(
    '-t',
    '--table',
    'table_enable',
    type=bool,
    help='Enable table parsing. Default is True. Adapted only for the case where the backend is set to "pipeline".',
    default=True,
)
@click.option(
    '-d',
    '--device',
    'device_mode',
    type=str,
    help="""Device mode for model inference, e.g., "cpu", "cuda", "cuda:0", "npu", "npu:0", "mps".
         Adapted only for the case where the backend is set to "pipeline" and "vlm-transformers". """,
    default=None,
)
@click.option(
    '--vram',
    'virtual_vram',
    type=int,
    help='Upper limit of GPU memory occupied by a single process. Adapted only for the case where the backend is set to "pipeline". ',
    default=None,
)
@click.option(
    '--source',
    'model_source',
    type=click.Choice(['huggingface', 'modelscope', 'local']),
    help="""
    The source of the model repository. Default is 'huggingface'.
    """,
    default='huggingface',
)
# AI 处理相关参数
@click.option(
    '--ai-process',
    'ai_process',
    is_flag=True,
    default=False,
    help='Enable AI processing after PDF parsing. The parsed result will be processed by text LLM.',
)
@click.option(
    '--ai-backend',
    'ai_backend',
    type=click.Choice(['http-client', 'transformers', 'vllm-engine', 'vllm-async-engine']),
    default='http-client',
    help="""\b
    AI processing backend type:
      http-client: Connect to external text LLM server (requires mineru-text-llm-server).
      transformers: Load model inside the process (no separate server needed).
      vllm-engine: Use vllm engine inside the process (no separate server needed).
      vllm-async-engine: Use vllm async engine inside the process (no separate server needed).
    """,
)
@click.option(
    '--ai-server-url',
    'ai_server_url',
    type=str,
    default=None,
    help='Text LLM server URL (only for http-client mode). Default: http://localhost:30001',
)
@click.option(
    '--ai-model-path',
    'ai_model_path',
    type=str,
    default=None,
    help='AI model path (for transformers/vllm-engine mode). If not specified, will auto-download.',
)
@click.option(
    '--ai-prompt-template',
    'ai_prompt_template',
    type=str,
    default=None,
    help='Optional prompt template for AI processing. Supports multiple placeholders: {text} (replaced by parsed content), {resume_text}, {json_template}, etc. Custom placeholders can be passed via environment variables or API.',
)
@click.option(
    '--ai-max-tokens',
    'ai_max_tokens',
    type=int,
    default=2048,
    help='Maximum tokens for AI generation. Default: 2048',
)
@click.option(
    '--ai-temperature',
    'ai_temperature',
    type=float,
    default=0.7,
    help='Temperature parameter for AI generation. Default: 0.7',
)
@click.option(
    '--ai-device',
    'ai_device',
    type=str,
    default=None,
    help='Device for AI model inference (e.g., "cpu", "cuda", "cuda:0"). If not specified, will use the same device as PDF parsing. Use "cpu" if GPU memory is insufficient.',
)
@click.option(
    '--ai-json-template',
    'ai_json_template',
    type=str,
    default=None,
    help='JSON template for prompt template placeholder {json_template}. To read from file, use: --ai-json-template "$(cat template.json)"',
)
@click.option(
    '--ai-resume-text',
    'ai_resume_text',
    type=str,
    default=None,
    help='Resume text for prompt template placeholder {resume_text}. To read from file, use: --ai-resume-text "$(cat resume.txt)". If not provided and template needs it, will use parsed content as {text}.',
)


def main(
        ctx,
        input_path, output_dir, method, backend, lang, server_url,
        start_page_id, end_page_id, formula_enable, table_enable,
        device_mode, virtual_vram, model_source,
        ai_process, ai_backend, ai_server_url, ai_model_path,
        ai_prompt_template, ai_max_tokens, ai_temperature,
        ai_device, ai_json_template, ai_resume_text,
        **kwargs
):

    kwargs.update(arg_parse(ctx))

    if not backend.endswith('-client'):
        def get_device_mode() -> str:
            if device_mode is not None:
                return device_mode
            else:
                return get_device()
        if os.getenv('MINERU_DEVICE_MODE', None) is None:
            os.environ['MINERU_DEVICE_MODE'] = get_device_mode()

        def get_virtual_vram_size() -> int:
            if virtual_vram is not None:
                return virtual_vram
            if get_device_mode().startswith("cuda") or get_device_mode().startswith("npu"):
                return round(get_vram(get_device_mode()))
            return 1
        if os.getenv('MINERU_VIRTUAL_VRAM_SIZE', None) is None:
            os.environ['MINERU_VIRTUAL_VRAM_SIZE']= str(get_virtual_vram_size())

        if os.getenv('MINERU_MODEL_SOURCE', None) is None:
            os.environ['MINERU_MODEL_SOURCE'] = model_source

    os.makedirs(output_dir, exist_ok=True)

    def parse_doc(path_list: list[Path]):
        try:
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(lang)
            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                **kwargs,
            )
            
            # 如果启用了 AI 处理，在解析完成后进行 AI 处理
            if ai_process:
                asyncio.run(process_with_ai(
                    file_name_list=file_name_list,
                    output_dir=output_dir,
                    backend=backend,
                    parse_method=method,
                    ai_backend=ai_backend,
                    ai_server_url=ai_server_url,
                    ai_model_path=ai_model_path,
                    ai_prompt_template=ai_prompt_template,
                    ai_max_tokens=ai_max_tokens,
                    ai_temperature=ai_temperature,
                    ai_device=ai_device,
                    ai_json_template=ai_json_template,
                    ai_resume_text=ai_resume_text,
                ))
        except Exception as e:
            logger.exception(e)

    if os.path.isdir(input_path):
        doc_path_list = []
        for doc_path in Path(input_path).glob('*'):
            if guess_suffix_by_path(doc_path) in pdf_suffixes + image_suffixes:
                doc_path_list.append(doc_path)
        parse_doc(doc_path_list)
    else:
        parse_doc([Path(input_path)])

async def process_with_ai(
    file_name_list: list[str],
    output_dir: str,
    backend: str,
    parse_method: str,
    ai_backend: str,
    ai_server_url: Optional[str],
    ai_model_path: Optional[str],
    ai_prompt_template: Optional[str],
    ai_max_tokens: int,
    ai_temperature: float,
    ai_device: Optional[str],
    ai_json_template: Optional[str] = None,
    ai_resume_text: Optional[str] = None,
):
    """
    对解析结果进行 AI 处理
    """
    # 从配置文件读取模型名称和量化方法
    from mineru.utils.config_reader import get_ai_config
    ai_config = get_ai_config()
    
    # 从配置文件读取，如果没有则使用默认值
    ai_model = 'Qwen/Qwen2.5-1.5B-Instruct'  # 默认值
    ai_quantization = None  # 默认值
    if ai_config:
        ai_model = ai_config.get('model', ai_model)
        ai_quantization = ai_config.get('quantization', ai_quantization)
    try:
        from mineru.utils.model_utils import (
            clean_memory, 
            unload_pipeline_models, 
            should_unload_pipeline_models
        )
        from mineru.utils.config_reader import get_device as get_device_func
        
        # 在加载 AI 模型之前，根据 GPU 大小决定是否卸载 pipeline 模型
        device = get_device_func()
        if device.startswith("cuda") or device.startswith("npu"):
            # 智能判断是否需要卸载 pipeline 模型（考虑量化）
            need_unload = should_unload_pipeline_models(device, ai_model, ai_backend, backend, ai_quantization)
            
            if need_unload:
                logger.info("Unloading pipeline models before loading AI model...")
                unload_pipeline_models(device)
                
                # 卸载后再次检查显存是否足够
                from mineru.utils.model_utils import get_vram, get_allocated_vram, estimate_llm_vram_requirement
                total_vram = get_vram(device)
                reserved_after = get_allocated_vram(device)
                if total_vram and reserved_after is not None:
                    available_after = total_vram - reserved_after
                    ai_vram_required = estimate_llm_vram_requirement(ai_model, ai_backend, ai_quantization)
                    if available_after < ai_vram_required * 0.8:  # 需要至少 80% 的模型大小
                        logger.warning(
                            f"After unloading, available VRAM ({available_after:.2f}GB) may still be insufficient "
                            f"for AI model ({ai_vram_required:.2f}GB). "
                            f"Consider using --ai-device cpu or a smaller model."
                        )
            else:
                logger.info("Clearing GPU memory before loading AI model...")
                clean_memory(device)
                logger.info("GPU memory cleared.")
        
        for pdf_name in file_name_list:
            # 确定解析结果目录
            if backend.startswith("pipeline"):
                parse_dir = os.path.join(output_dir, pdf_name, parse_method)
            else:
                parse_dir = os.path.join(output_dir, pdf_name, "vlm")
            
            if not os.path.exists(parse_dir):
                logger.warning(f"Parse directory not found: {parse_dir}")
                continue
            
            # 读取解析结果（始终使用最终结果 .md 文件）
            parse_text = None
            md_path = os.path.join(parse_dir, f"{pdf_name}.md")
            if os.path.exists(md_path):
                with open(md_path, 'r', encoding='utf-8') as f:
                    parse_text = f.read()
            
            if not parse_text:
                logger.warning(f"No parse result found for {pdf_name}")
                continue
            
            # 调用 AI 处理
            logger.info(f"Processing {pdf_name} with AI (backend: {ai_backend})...")
            try:
                # 如果指定了 ai_device，设置环境变量
                if ai_device is not None:
                    original_device = os.environ.get('MINERU_DEVICE_MODE')
                    os.environ['MINERU_DEVICE_MODE'] = ai_device
                    logger.info(f"Using device '{ai_device}' for AI model")
                
                # 准备额外的占位符参数
                extra_kwargs = {}
                if ai_json_template is not None:
                    extra_kwargs["json_template"] = ai_json_template
                # 如果resume_text没有提供，但模板需要它，使用parse_text作为resume_text
                if ai_resume_text is not None:
                    extra_kwargs["resume_text"] = ai_resume_text
                elif ai_prompt_template and "{resume_text}" in ai_prompt_template:
                    # 如果模板中有{resume_text}但没有提供，使用parse_text
                    extra_kwargs["resume_text"] = parse_text
                
                # 构建提示词
                from mineru.backend.text_llm import build_prompt, generate_text_async
                prompt = build_prompt(
                    text=parse_text,
                    prompt_template=ai_prompt_template,
                    **extra_kwargs,
                )
                
                # 调用生成函数
                ai_result = await generate_text_async(
                    prompt=prompt,
                    backend=ai_backend,
                    model_path=ai_model_path,
                    server_url=ai_server_url,
                    max_tokens=ai_max_tokens,
                    temperature=ai_temperature,
                    model=ai_model,
                    quantization=ai_quantization,
                )
                
                # 恢复原始设备设置
                if ai_device is not None:
                    if original_device is not None:
                        os.environ['MINERU_DEVICE_MODE'] = original_device
                    else:
                        os.environ.pop('MINERU_DEVICE_MODE', None)
                
                # 保存 AI 处理结果
                ai_output_path = os.path.join(parse_dir, f"{pdf_name}_ai_result.txt")
                with open(ai_output_path, 'w', encoding='utf-8') as f:
                    f.write(ai_result)
                
                logger.info(f"AI processing completed for {pdf_name}. Result saved to: {ai_output_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_name} with AI: {e}")
                logger.exception(e)
                
    except ImportError as e:
        logger.error(f"Failed to import text LLM module: {e}")
        logger.error("Please ensure mineru[all] or mineru[vllm] is installed for AI processing.")
    except Exception as e:
        logger.error(f"Error in AI processing: {e}")
        logger.exception(e)


if __name__ == '__main__':
    main()
