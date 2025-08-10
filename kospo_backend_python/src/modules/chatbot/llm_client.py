"""
LLM 호출 클라이언트 모듈 (OpenRouter API 사용)
"""

import json
import os
import logging
from typing import Optional, Dict, Any, List, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenRouter API를 사용하는 LLM 클라이언트.
    
    이 클래스는 다양한 LLM 모델에 대한 통합 인터페이스를 제공
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        openrouter_api_key: Optional[str] = None
    ):
        """
        LLMClient 초기화.
        
        Args:
            model_name: 사용할 LLM 모델 이름 (OpenRouter 형식)
            temperature: 생성 다양성 조절 파라미터 (0.0 ~ 2.0)
            max_tokens: 최대 생성 토큰 수
            openrouter_api_key: OpenRouter API 키 (환경변수에서 자동 로드)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # API 키 설정
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API 키가 필요합니다. 환경변수 OPENROUTER_API_KEY를 설정하거나 파라미터로 전달하세요.")
        
        # LLM 인스턴스 생성
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.api_key,
            max_tokens=max_tokens
        )
        
        logger.info(f"LLM 클라이언트 초기화 완료: {model_name}")
        
    async def generate_response(
        self,
        messages: List[BaseMessage]
    ) -> str:
        """
        메시지 목록을 받아 LLM 응답을 생성합니다.
        
        Args:
            messages: 입력 메시지 목록
            
        Returns:
            str: 생성된 응답 텍스트
        """
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            raise
            
    async def generate_streaming_response(
        self,
        messages: List[BaseMessage],
        on_token_callback: Optional[Callable[[str, str], None]] = None
    ) -> str:
        """
        토큰별 스트리밍으로 LLM 응답을 생성합니다.
        
        Args:
            messages: 입력 메시지 목록
            on_token_callback: 토큰 생성 시 호출될 콜백 함수 (token, partial_text)
            
        Returns:
            str: 최종 생성된 응답 텍스트
        """
        try:
            final_answer = ""
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    token = chunk.content
                    final_answer += token
                    
                    # 콜백이 있으면 호출
                    if on_token_callback:
                        await on_token_callback(token, final_answer)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"스트리밍 응답 생성 실패: {e}")
            raise