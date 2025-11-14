import uuid
import os
import re
import tempfile
import asyncio
import uvicorn
import click
import zipfile
from pathlib import Path
import glob
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from typing import List, Optional
from loguru import logger
from base64 import b64encode

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.version import __version__

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


def sanitize_filename(filename: str) -> str:
    """
    格式化压缩文件的文件名
    移除路径遍历字符, 保留 Unicode 字母、数字、._- 
    禁止隐藏文件
    """
    sanitized = re.sub(r'[/\\\.]{2,}|[/\\]', '', filename)
    sanitized = re.sub(r'[^\w.-]', '_', sanitized, flags=re.UNICODE)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    return sanitized or 'unnamed'

def cleanup_file(file_path: str) -> None:
    """清理临时 zip 文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")

def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


@app.post(path="/file_parse",)
async def parse_pdf(
        files: List[UploadFile] = File(...),
        output_dir: str = Form("./output"),
        lang_list: List[str] = Form(["ch"]),
        backend: str = Form("pipeline"),
        parse_method: str = Form("auto"),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(True),
        server_url: Optional[str] = Form(None),
        return_md: bool = Form(True),
        return_middle_json: bool = Form(False),
        return_model_output: bool = Form(False),
        return_content_list: bool = Form(False),
        return_images: bool = Form(False),
        response_format_zip: bool = Form(False),
        start_page_id: int = Form(0),
        end_page_id: int = Form(99999),
):

    # 获取命令行配置参数
    config = getattr(app.state, "config", {})

    try:
        # 创建唯一的输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        # 处理上传的PDF文件
        pdf_file_names = []
        pdf_bytes_list = []

        for file in files:
            content = await file.read()
            file_path = Path(file.filename)

            # 创建临时文件
            temp_path = Path(unique_dir) / file_path.name
            with open(temp_path, "wb") as f:
                f.write(content)

            # 如果是图像文件或PDF，使用read_fn处理
            file_suffix = guess_suffix_by_path(temp_path)
            if file_suffix in pdf_suffixes + image_suffixes:
                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(file_path.stem)
                    os.remove(temp_path)  # 删除临时文件
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"}
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_suffix}"}
                )


        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

        # 调用异步处理函数
        await aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=return_md,
            f_dump_middle_json=return_middle_json,
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **config
        )

        # 根据 response_format_zip 决定返回类型
        if response_format_zip:
            zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_results_")
            os.close(zip_fd) 
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for pdf_name in pdf_file_names:
                    safe_pdf_name = sanitize_filename(pdf_name)
                    if backend.startswith("pipeline"):
                        parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                    else:
                        parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

                    if not os.path.exists(parse_dir):
                        continue

                    # 写入文本类结果
                    if return_md:
                        path = os.path.join(parse_dir, f"{pdf_name}.md")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}.md"))

                    if return_middle_json:
                        path = os.path.join(parse_dir, f"{pdf_name}_middle.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_middle.json"))

                    if return_model_output:
                        path = os.path.join(parse_dir, f"{pdf_name}_model.json")
                        if os.path.exists(path): 
                            zf.write(path, arcname=os.path.join(safe_pdf_name, os.path.basename(path)))

                    if return_content_list:
                        path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_content_list.json"))

                    # 写入图片
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        image_paths = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                        for image_path in image_paths:
                            zf.write(image_path, arcname=os.path.join(safe_pdf_name, "images", os.path.basename(image_path)))

            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="results.zip",
                background=BackgroundTask(cleanup_file, zip_path)
            )
        else:
            # 构建 JSON 结果
            result_dict = {}
            for pdf_name in pdf_file_names:
                result_dict[pdf_name] = {}
                data = result_dict[pdf_name]

                if backend.startswith("pipeline"):
                    parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                else:
                    parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

                if os.path.exists(parse_dir):
                    if return_md:
                        data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
                    if return_middle_json:
                        data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
                    if return_model_output:
                        data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
                    if return_content_list:
                        data["content_list"] = get_infer_result("_content_list.json", pdf_name, parse_dir)
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        safe_pattern = os.path.join(glob.escape(images_dir), "*.jpg")
                        image_paths = glob.glob(safe_pattern)
                        data["images"] = {
                            os.path.basename(
                                image_path
                            ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                            for image_path in image_paths
                        }

            return JSONResponse(
                status_code=200,
                content={
                    "backend": backend,
                    "version": __version__,
                    "results": result_dict
                }
            )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )


async def call_vllm_api(
    text: str,
    backend: str = "http-client",
    vllm_server_url: Optional[str] = None,
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    model_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    timeout: float = 300.0,
    quantization: Optional[str] = None,
    **kwargs,
) -> str:
    """
    调用文本LLM进行文本生成（支持多种后端，类似VLM架构）
    
    Args:
        text: 要处理的文本内容（如果prompt_template中有{text}占位符，会被替换）
        backend: 后端类型，支持 'http-client'（默认，需要外部服务器）、'transformers'（API内部）、'vllm-engine'（API内部）、'vllm-async-engine'（API内部）
        vllm_server_url: vllm服务器地址（仅http-client模式需要），如果为None则使用默认值
        model: 模型名称，默认为 Qwen/Qwen2.5-3B-Instruct
        model_path: 模型路径（transformers/vllm-engine模式），如果为None则自动下载
        prompt_template: 可选的提示词模板，支持多个占位符（{text}, {resume_text}, {json_template}等）
        max_tokens: 最大生成token数
        temperature: 温度参数
        timeout: 请求超时时间（秒，仅http-client模式）
        **kwargs: 其他参数，包括自定义占位符的值（如resume_text, json_template等）
    
    Returns:
        生成的文本结果
    """
    try:
        from mineru.backend.text_llm import generate_text_async
        
        # 如果是http-client模式且未指定server_url，使用默认值
        if backend == "http-client" and vllm_server_url is None:
            vllm_server_url = "http://localhost:30001"
        
        # 调用统一的生成函数
        result = await generate_text_async(
            text=text,
            backend=backend,
            model_path=model_path,
            server_url=vllm_server_url,
            prompt_template=prompt_template,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            timeout=timeout,
            quantization=quantization,
            **kwargs,
        )
        
        return result
                
    except Exception as e:
        logger.error(f"Error when calling text LLM API: {e}")
        raise Exception(f"Failed to process text with LLM: {str(e)}")


@app.post(path="/ai_process")
async def ai_process(
    parse_result: str = Form(..., description="解析结果文本（可以是md_content或middle_json的字符串形式）"),
    backend: str = Form("http-client", description="后端类型：http-client（需要外部服务器）、transformers（API内部）、vllm-engine（API内部）、vllm-async-engine（API内部）"),
    vllm_server_url: Optional[str] = Form(None, description="vllm服务器地址（仅http-client模式需要），如果为空则使用默认值"),
    model: str = Form("Qwen/Qwen2.5-3B-Instruct", description="模型名称"),
    model_path: Optional[str] = Form(None, description="模型路径（transformers/vllm-engine模式），如果为空则自动下载"),
    prompt_template: Optional[str] = Form(None, description="可选的提示词模板，支持多个占位符：{text}（会被parse_result替换）、{resume_text}、{json_template}等自定义占位符"),
    max_tokens: int = Form(2048, description="最大生成token数"),
    temperature: float = Form(0.7, description="温度参数"),
    quantization: Optional[str] = Form(None, description="量化方法（仅vllm-engine和vllm-async-engine支持）：fp8（需要GPU计算能力>=8.9）、int8、int4"),
    # 支持自定义占位符，通过额外的Form参数传递
    resume_text: Optional[str] = Form(None, description="简历文本（用于prompt_template中的{resume_text}占位符）"),
    json_template: Optional[str] = Form(None, description="JSON模板（用于prompt_template中的{json_template}占位符）"),
):
    """
    接收PDF解析结果，使用文本LLM进行处理（支持多种后端，类似VLM架构）
    
    这个接口接收已经解析好的PDF结果（可以是markdown文本或JSON格式），
    然后使用文本LLM进行AI处理。
    
    后端选项：
    - http-client: 连接到外部文本LLM服务器（需要单独启动mineru-text-llm-server）
    - transformers: 在API服务内部直接加载模型（无需单独服务器）
    - vllm-engine: 在API服务内部使用vllm引擎（无需单独服务器）
    - vllm-async-engine: 在API服务内部使用vllm异步引擎（无需单独服务器）
    """
    try:
        # 准备自定义占位符的值
        extra_kwargs = {}
        if resume_text is not None:
            extra_kwargs["resume_text"] = resume_text
        if json_template is not None:
            extra_kwargs["json_template"] = json_template
        
        # 调用文本LLM API处理文本
        ai_result = await call_vllm_api(
            text=parse_result,
            backend=backend,
            vllm_server_url=vllm_server_url,
            model=model,
            model_path=model_path,
            prompt_template=prompt_template,
            max_tokens=max_tokens,
            temperature=temperature,
            quantization=quantization,
            **extra_kwargs,
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "original_text": parse_result[:500] + "..." if len(parse_result) > 500 else parse_result,  # 只返回前500字符作为预览
                "ai_result": ai_result,
                "model": model,
            }
        )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process text with AI: {str(e)}"}
        )


@app.post(path="/file_parse_with_ai")
async def parse_pdf_with_ai(
    files: List[UploadFile] = File(...),
    output_dir: str = Form("./output"),
    lang_list: List[str] = Form(["ch"]),
    backend: str = Form("pipeline"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    server_url: Optional[str] = Form(None),
    return_md: bool = Form(True),
    return_middle_json: bool = Form(False),
    return_model_output: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(False),
    response_format_zip: bool = Form(False),
    start_page_id: int = Form(0),
    end_page_id: int = Form(99999),
    # AI处理相关参数
    ai_backend: str = Form("http-client", description="AI处理后端类型：http-client（需要外部服务器）、transformers（API内部）、vllm-engine（API内部）、vllm-async-engine（API内部）"),
    vllm_server_url: Optional[str] = Form(None, description="vllm服务器地址（仅http-client模式需要），如果为空则使用默认值"),
    model: str = Form("Qwen/Qwen2.5-3B-Instruct", description="模型名称"),
    model_path: Optional[str] = Form(None, description="模型路径（transformers/vllm-engine模式），如果为空则自动下载"),
    prompt_template: Optional[str] = Form(None, description="可选的提示词模板，支持多个占位符：{text}（会被解析结果替换）、{resume_text}、{json_template}等自定义占位符"),
    max_tokens: int = Form(2048, description="最大生成token数"),
    temperature: float = Form(0.7, description="温度参数"),
    quantization: Optional[str] = Form(None, description="量化方法（仅vllm-engine和vllm-async-engine支持）：fp8（需要GPU计算能力>=8.9）、int8、int4"),
    use_md_for_ai: bool = Form(True, description="如果为True，使用md_content进行AI处理；否则使用middle_json"),
    resume_text: Optional[str] = Form(None, description="简历文本（用于prompt_template中的{resume_text}占位符）"),
    json_template: Optional[str] = Form(None, description="JSON模板（用于prompt_template中的{json_template}占位符）"),
):
    """
    完整流程：PDF解析 + AI处理（支持多种后端，类似VLM架构）
    
    这个接口会先解析PDF文件，然后使用文本LLM进行AI处理，
    最后返回解析结果和AI处理结果。
    
    AI处理后端选项：
    - http-client: 连接到外部文本LLM服务器（需要单独启动mineru-text-llm-server）
    - transformers: 在API服务内部直接加载模型（无需单独服务器）
    - vllm-engine: 在API服务内部使用vllm引擎（无需单独服务器）
    - vllm-async-engine: 在API服务内部使用vllm异步引擎（无需单独服务器）
    """
    # 获取命令行配置参数
    config = getattr(app.state, "config", {})

    try:
        # 创建唯一的输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        # 处理上传的PDF文件
        pdf_file_names = []
        pdf_bytes_list = []

        for file in files:
            content = await file.read()
            file_path = Path(file.filename)

            # 创建临时文件
            temp_path = Path(unique_dir) / file_path.name
            with open(temp_path, "wb") as f:
                f.write(content)

            # 如果是图像文件或PDF，使用read_fn处理
            file_suffix = guess_suffix_by_path(temp_path)
            if file_suffix in pdf_suffixes + image_suffixes:
                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(file_path.stem)
                    os.remove(temp_path)  # 删除临时文件
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"}
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_suffix}"}
                )

        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

        # 调用异步处理函数进行PDF解析
        await aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=return_md or use_md_for_ai,  # 如果要用md进行AI处理，确保生成md
            f_dump_middle_json=return_middle_json or (not use_md_for_ai),  # 如果不用md，确保生成middle_json
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **config
        )

        # 构建结果并调用AI处理
        result_dict = {}
        for pdf_name in pdf_file_names:
            result_dict[pdf_name] = {}
            data = result_dict[pdf_name]

            if backend.startswith("pipeline"):
                parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
            else:
                parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

            if not os.path.exists(parse_dir):
                continue

            # 读取解析结果
            parse_text = None
            if use_md_for_ai:
                parse_text = get_infer_result(".md", pdf_name, parse_dir)
            else:
                parse_text = get_infer_result("_middle.json", pdf_name, parse_dir)
            
            # 如果找到了解析结果，调用AI处理
            if parse_text:
                try:
                    # 准备自定义占位符的值
                    extra_kwargs = {}
                    if resume_text is not None:
                        extra_kwargs["resume_text"] = resume_text
                    if json_template is not None:
                        extra_kwargs["json_template"] = json_template
                    
                    ai_result = await call_vllm_api(
                        text=parse_text,
                        backend=ai_backend,
                        vllm_server_url=vllm_server_url,
                        model=model,
                        model_path=model_path,
                        prompt_template=prompt_template,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        quantization=quantization,
                        **extra_kwargs,
                    )
                    data["ai_result"] = ai_result
                except Exception as e:
                    logger.error(f"Failed to process {pdf_name} with AI: {e}")
                    data["ai_error"] = str(e)
            else:
                data["ai_error"] = "No parse result found for AI processing"

            # 添加其他返回结果
            if return_md:
                data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
            if return_middle_json:
                data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
            if return_model_output:
                data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
            if return_content_list:
                data["content_list"] = get_infer_result("_content_list.json", pdf_name, parse_dir)
            if return_images:
                images_dir = os.path.join(parse_dir, "images")
                safe_pattern = os.path.join(glob.escape(images_dir), "*.jpg")
                image_paths = glob.glob(safe_pattern)
                data["images"] = {
                    os.path.basename(
                        image_path
                    ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                    for image_path in image_paths
                }

        return JSONResponse(
            status_code=200,
            content={
                "backend": backend,
                "version": __version__,
                "model": model,
                "results": result_dict
            }
        )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file with AI: {str(e)}"}
        )


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
@click.option('--port', default=8000, type=int, help='Server port (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):

    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs

    """启动MinerU FastAPI服务器的命令行入口"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "mineru.cli.fast_api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()