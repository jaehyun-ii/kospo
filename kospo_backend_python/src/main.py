"""
FastAPI 메인 애플리케이션
- src/api 폴더의 모든 라우터를 포함
- LLM 호출 방식은 openrouter를 사용
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.chatbot import router as chatbot_router
from src.api.chat_history import router as chat_history_router
import uvicorn

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리.
    
    Args:
        app: FastAPI 애플리케이션 인스턴스
    """
    # 시작 시 실행
    logger.info("🚀 LangGraph 챗봇 서비스 시작 중...")
    
    # 환경변수 확인
    required_env_vars = ["OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"⚠️ 다음 환경변수가 설정되지 않았습니다: {missing_vars}")
        logger.warning("   일부 기능이 제한될 수 있습니다.")
    
    logger.info("✅ 서비스 시작 완료")
    
    yield
    
    # 종료 시 실행
    logger.info("🛑 서비스 종료 중...")

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="LangGraph 챗봇 서비스",
    description="LangGraph를 활용한 LLM 기반 챗봇 서비스 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 엔드포인트
@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    루트 엔드포인트.
    
    Returns:
        Dict[str, Any]: 서비스 정보
    """
    return {
        "service": "LangGraph 챗봇 서비스",
        "version": "1.0.0",
        "description": "LangGraph를 활용한 LLM 기반 챗봇 서비스",
        "endpoints": {
            "docs": "/docs",
            "chat": "/api/v1/chatbot/ws/chat",
            "chat_conversations": "/api/v1/chat-history/ws/conversations"
        }
    }

# API 라우터 등록
app.include_router(chatbot_router)
app.include_router(chat_history_router)


# 예외처리 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    전역 예외 처리기.
    
    Args:
        request: 요청 객체
        exc: 발생한 예외
        
    Returns:
        JSONResponse: 오류 응답
    """
    logger.error(f"전역 예외 발생: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "내부 서버 오류가 발생했습니다.",
            "error_code": "INTERNAL_SERVER_ERROR"
        }
    )

# 예외처리 핸들러
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    HTTP 예외 처리기.
    
    Args:
        request: 요청 객체
        exc: HTTP 예외
        
    Returns:
        JSONResponse: 오류 응답
    """
    logger.error(f"HTTP 예외 발생: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}"
        }
    )

if __name__ == "__main__":
    
    # 개발 서버 실행
    port = int(os.getenv("PORT", "3002"))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    ) 