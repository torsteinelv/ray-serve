from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from starlette.responses import Response
from starlette.requests import Request
from http import HTTPStatus

from fastapi import FastAPI, HTTPException
import logging
import uuid

import nest_asyncio
from ray import serve
from ray.serve import Application
from typing import Optional, Literal, List, Dict
from pydantic import BaseModel


logger = logging.getLogger("ray.serve")

app = FastAPI()

class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content

class GenerateRequest(BaseModel):
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    prompt: Optional[str]
    messages: Optional[List[Message]]

class GenerateResponse(BaseModel):
    output: Optional[str]
    finish_reason: Optional[str]
    prompt: Optional[str]

def _prepare_engine_args(num_gpus: int, tensor_parallel_size: int, model: str, dtype: str, trust_remote_code: bool):
    engine_args = AsyncEngineArgs(
        model=model,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,  # Support tensor parallelism
    )
    return engine_args

@serve.deployment(name='VLLMInference', 
                  num_replicas=1, 
                  max_concurrent_queries=256,
                  ray_actor_options={"num_gpus": 1.0}  # Set this as per GPU requirements
                 )
@serve.ingress(app)
class VLLMInference:
    def __init__(self, model: str, num_gpus: int, tensor_parallel_size: int, **kwargs):
        super().__init__(app)
        self.args = _prepare_engine_args(num_gpus, tensor_parallel_size, model, dtype="float16", trust_remote_code=True)
        self.engine = AsyncLLMEngine.from_engine_args(self.args)
        self.tokenizer = self._prepare_tokenizer()

    def _prepare_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model, trust_remote_code=True)
        return tokenizer

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_text(self, request: GenerateRequest, raw_request: Request) -> GenerateResponse:
        logging.info(f"Received request: {request}")
        try:
            generation_args = request.dict(exclude={'prompt', 'messages'})
            if generation_args is None:
                generation_args = {
                    "max_tokens": 500,
                    "temperature": 0.1,
                }
            
            if request.prompt:
                prompt = request.prompt
            elif request.messages:
                prompt = self.tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                raise ValueError("Prompt or Messages is required")

            sampling_params = SamplingParams(**generation_args)
            request_id = self._next_request_id()
            results_generator = self.engine.generate(prompt, sampling_params, request_id)

            final_result = None
            async for result in results_generator:
                if await raw_request.is_disconnected():
                    await self.engine.abort(request_id)
                    return GenerateResponse()
                final_result = result  # Store the last result
            if final_result:
                return GenerateResponse(output=final_result.outputs[0].text,
                                        finish_reason=final_result.outputs[0].finish_reason,
                                        prompt=final_result.prompt)
            else:
                raise ValueError("No results found")
        except ValueError as e:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            logger.error('Error in generate()', exc_info=1)
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, 'Server error')

    @staticmethod
    def _next_request_id():
        return str(uuid.uuid1().hex)

    async def _abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

def deployment_llm(args: Dict[str, str]) -> Application:
    return VLLMInference.bind(**args)
