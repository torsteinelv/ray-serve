import os
from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
#from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
#        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
#        self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names=served_model_names,
                response_role=self.response_role,
#                lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                prompt_adapters=None,
                request_logger=None,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, Optional[str]]):
    # Logg inngangsargumentene
    logger.info(f"Initial CLI arguments: {cli_args}")
    
    # Opprett parser
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    
    arg_strings = []
    for key, value in cli_args.items():
        logger.info(f"Processing argument: --{key} with value: {value}")
        
        if value is True:  # Håndter boolske flagg satt til True
            arg_strings.append(f"--{key}")
            logger.info(f"Added argument: --{key} (boolean flag)")
        elif value not in (None, "None"):  # Ignorer None eller 'None' som streng
            arg_strings.extend([f"--{key}", str(value)])
            logger.info(f"Added argument: --{key} with value: {value}")
        else:
            arg_strings.append(f"--{key}")
            # Dette skjer hvis argumentet er None eller 'None'
            logger.info(f"Skipping argument: --{key} because its value is {value}")
    
    # Logg de ferdig prosesserte argumentene før parsing
    logger.info(f"Final argument list to be parsed: {arg_strings}")
    
    parsed_args = parser.parse_args(args=arg_strings)
    
    # Logg de parsed argumentene
    logger.info(f"Parsed arguments: {parsed_args}")
    
    return parsed_args

def build_app(cli_args: Dict[str, Optional[str]]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""
    
    # Logg start av build_app
    logger.info(f"Building the application with CLI arguments: {cli_args}")
    
    parsed_args = parse_vllm_args(cli_args)
    
    # Logg parsed_args
    logger.info(f"Parsed engine arguments: {parsed_args}")
    
    # Lag AsyncEngineArgs
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    
    # Logg engine_args
    logger.info(f"Engine arguments after parsing: {engine_args}")
    
    # Sett worker_use_ray til True for Ray integrasjon
    engine_args.worker_use_ray = True
    
    # Logg opprettelsen av VLLMDeployment
    logger.info(f"Binding VLLMDeployment with engine args: {engine_args} and response_role: {parsed_args.response_role}")
    
    # Returner bindet VLLMDeployment
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
#        parsed_args.lora_modules,
        parsed_args.chat_template,
    )
