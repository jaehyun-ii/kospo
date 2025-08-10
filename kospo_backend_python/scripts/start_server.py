#!/usr/bin/env python3
"""
LangGraph 챗봇 서비스 실행 스크립트.

이 스크립트는 FastAPI 서버를 실행하기 위한 편의 스크립트입니다.
"""

import os
import sys
import uvicorn
import argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def main():
    """메인 실행 함수."""
    parser = argparse.ArgumentParser(description="LangGraph 챗봇 서비스 실행")
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="서버 호스트 (기본값: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="서버 포트 (기본값: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="개발 모드 (자동 리로드)"
    )
    
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="로그 레벨 (기본값: info)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="워커 프로세스 수 (기본값: 1)"
    )
    
    args = parser.parse_args()
    
    # 프로젝트 루트 디렉토리를 Python 경로에 추가
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 환경변수 확인
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  경고: OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export OPENROUTER_API_KEY='your-api-key' 를 실행하세요.")
        print("   또는 .env 파일에 설정하세요.")
        print()
    
    print("🚀 LangGraph 챗봇 서비스 시작 중...")
    print(f"   호스트: {args.host}")
    print(f"   포트: {args.port}")
    print(f"   개발 모드: {args.reload}")
    print(f"   로그 레벨: {args.log_level}")
    print(f"   워커 수: {args.workers}")
    print()
    
    try:
        # uvicorn 서버 실행
        uvicorn.run(
            "src.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1
        )
    except KeyboardInterrupt:
        print("\n🛑 서버가 중단되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 