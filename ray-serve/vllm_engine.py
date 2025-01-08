import os
from typing import Dict, Optional
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from ray import serve
import ray

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

app = FastAPI()

# Ray remote class for handling the VLLM deployment
@ray.remote
class AsyncVLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        chat_template: Optional[str] = None,
    ):
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            served_model_names = self.engine_args.served_model_name or [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names=served_model_names,
                response_role=self.response_role,
                chat_template=self.chat_template,
            )
        
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

# FastAPI endpoint to handle chat completions requests
@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest, raw_request: Request
):
    cli_args = { "response_role": "system" }  # Example CLI arguments
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    
    # Instantiate the distributed deployment using Ray remote
    async_deployment = AsyncVLLMDeployment.options(name="async_vllm_deployment").remote(
        engine_args, parsed_args.response_role, parsed_args.chat_template
    )
    
    # Perform the async chat completion by calling the remote deployment
    result = await async_deployment.create_chat_completion.remote(request, raw_request)
    
    return result

def parse_vllm_args(cli_args: Dict[str, Optional[str]]):
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    
    arg_strings = []
    for key, value in cli_args.items():
        if value is True:  # Handle boolean flags set to True
            arg_strings.append(f"--{key}")
        elif value not in (None, "None"):  # Ignore None or 'None' string
            arg_strings.extend([f"--{key}", str(value)])
        else:
            arg_strings.append(f"--{key}")
    
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args

def build_app(cli_args: Dict[str, Optional[str]]) -> serve.Application:
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    
    engine_args.worker_use_ray = True
    
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.chat_template,
    )
