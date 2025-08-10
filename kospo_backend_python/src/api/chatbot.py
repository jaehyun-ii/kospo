"""
챗봇 API 엔드포인트 모듈.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional
import pytz
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from src.modules.chatbot.workflow import ChatbotWorkflow
from src.modules.chatbot.redis_client import ChatRedisClient, ChatConversation

logger = logging.getLogger(__name__)

# API 라우터 생성
router = APIRouter(prefix="/api/v1/chatbot", tags=["chatbot"])

# 워크플로우 인스턴스 저장소 (실제로는 Redis나 DB 사용 권장)
workflow_instances: Dict[str, ChatbotWorkflow] = {}

# Redis 클라이언트 인스턴스
redis_client = ChatRedisClient()

# 헬스체크 엔드포인트
@router.get("/health")
async def health_check():
    """
    서비스 헬스체크 엔드포인트.
    
    Returns:
        HealthCheckResponse: 서비스 상태 정보
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(pytz.timezone('Asia/Seoul')).timestamp()
    }

# 실제 대화 엔드포인트
@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    챗봇 대화 WebSocket 엔드포인트
    
    Args:
        websocket: WebSocket 연결 객체
    """
    # WebSocket 연결 수락
    await websocket.accept()
    logger.info("WebSocket 연결이 수락되었습니다.")
    
    # Redis 연결 시도 (채팅 대화 저장 및 불러오기 위해 필요)
    try:
        await redis_client.connect()
    except Exception as e:
        logger.error(f"Redis 연결 실패: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": "Redis 연결 실패"}
        }))
        return
    
    # 메시지 수신 루프 (초기 연결시 웹소켓 연결해두면, 지속적으로 메시지 수신)
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            
            try:
                # JSON 파싱
                request_data = json.loads(data)
                message = request_data.get("message", "")
                user_id = request_data.get("user_id")
                conversation_id = request_data.get("conversation_id")
                model_name = request_data.get("model_name", "openai/gpt-oss-120b")
                
                # 메시지가 비어있으면 에러 반환
                if not message:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "메시지가 비어있습니다."}
                    })) 
                    continue
                
                # user_id가 비어있으면 에러 반환
                if not user_id:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "user_id가 필요합니다."}
                    }))
                    continue
                
                # 대화 처리 (대화 ID 있으면 기존 대화 불러오기, 없으면 새로운 대화 생성)
                current_conversation_id = await _handle_chat_conversation(
                    websocket, user_id, conversation_id, model_name
                )
                
                # 대화 ID가 없으면 루프 종료
                if not current_conversation_id:
                    continue
                
                # 사용자 메시지를 대화에 저장 (Redis에 저장)
                await redis_client.add_message_to_conversation(
                    conversation_id=current_conversation_id,
                    role="user",
                    content=message
                )
                
                # 워크플로우 인스턴스 가져오기 (워크플로우 인스턴스 생성 함수)
                workflow = _get_workflow_instance(model_name)
                
                # WebSocket을 워크플로우에 연결 (실시간 응답 전송을 위해)
                workflow.websocket = websocket
                
                # 워크플로우 실행
                result = await workflow.process_message(
                    user_message=message,
                    conversation_id=current_conversation_id
                )
                
                # 어시스턴트 응답을 대화에 저장
                if result.get("final_answer"):
                    await redis_client.add_message_to_conversation(
                        conversation_id=current_conversation_id,
                        role="assistant",
                        content=result["final_answer"]
                    )
                
                # 최종 리턴 값 (연결 상태 확인 후)
                try:
                    final_result = {
                        "type": "final_result",
                        "data": {
                            "success": result["success"],
                            "conversation_id": current_conversation_id,
                            "final_answer": result.get("final_answer"),
                            "messages": result.get("messages", [])
                        },
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    await websocket.send_text(json.dumps(final_result, ensure_ascii=False))
                    logger.info("워크플로우 실행 완료")
                    
                except Exception as e:
                    logger.warning(f"최종 결과 전송 실패 (연결이 끊어졌을 수 있음): {e}")
                    # 연결이 끊어진 경우 루프를 종료하지 않고 다음 메시지를 기다림
                    continue
                
            except json.JSONDecodeError:
                logger.warning("잘못된 JSON 형식의 메시지 수신")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "잘못된 JSON 형식입니다."}
                    }))
                except Exception as e:
                    logger.warning(f"오류 메시지 전송 실패: {e}")
                    break
            except Exception as e:
                logger.error(f"WebSocket 처리 오류: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": f"서버 오류: {str(e)}"}
                    }))
                except Exception as send_error:
                    logger.warning(f"오류 메시지 전송 실패: {send_error}")
                    break
                
    except WebSocketDisconnect:
        logger.info("WebSocket 연결이 종료되었습니다.")
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": f"연결 오류: {str(e)}"}
            }))
        except:
            pass
    finally:
        # Redis 연결 종료
        await redis_client.disconnect() 
        
# 워크플로우 인스턴스 생성 함수
def _get_workflow_instance(model_name: str = "openai/gpt-oss-120b") -> ChatbotWorkflow:
    """
    워크플로우 인스턴스를 가져오거나 생성
    - 동일한 모델명일 경우 최초에 한번만 생성하고, 이후에는 기존 인스턴스 반환
    - 즉, 랭그래프 워크플로우를 초기에 한번만 생성하고, 이후에는 기존 인스턴스 반환하는 것
    
    Args:
        model_name: 사용할 LLM 모델 이름
        
    Returns:
        ChatbotWorkflow: 워크플로우 인스턴스
    """
    if model_name not in workflow_instances:
        try:
            workflow_instances[model_name] = ChatbotWorkflow(
                model_name=model_name,
                temperature=0.2,
                redis_client=redis_client
            )
            logger.info(f"워크플로우 인스턴스 생성: {model_name}")
        except Exception as e:
            logger.error(f"워크플로우 인스턴스 생성 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"워크플로우 초기화 실패: {str(e)}"
            )
    
    return workflow_instances[model_name]


# 채팅 대화 처리 함수
async def _handle_chat_conversation(
    websocket: WebSocket,
    user_id: str,
    conversation_id: Optional[str],
    model_name: str
) -> Optional[str]:
    """
    채팅 대화를 처리합니다.
    
    Args:
        websocket: WebSocket 연결 객체
        user_id: 사용자 ID
        conversation_id: 대화 ID (선택사항)
        model_name: 모델 이름
        
    Returns:
        Optional[str]: 현재 대화 ID 또는 None
    """
    try:
        current_conversation_id = conversation_id
        
        # 대화 ID가 제공된 경우 기존 대화 불러오기
        if conversation_id:
            conversation = await redis_client.load_chat_conversation(conversation_id)
            if not conversation:
                # 새 대화 생성
                current_conversation_id = str(uuid.uuid4())
                new_conversation = ChatConversation(
                    conversation_id=current_conversation_id,
                    user_id=user_id,
                    messages=[],
                    created_at=datetime.now().timestamp(),
                    updated_at=datetime.now().timestamp(),
                    model_name=model_name
                )
                
                # 새 대화 저장 (새로운 사용자 ID 저장, 메시지는 비어있음.)
                await redis_client.save_chat_conversation(new_conversation)
        
        # 새 대화 생성 (프론트측에서 null로 넘길경우, 새로운 대화 생성)
        else:
            current_conversation_id = str(uuid.uuid4())
            new_conversation = ChatConversation(
                conversation_id=current_conversation_id,
                user_id=user_id,
                messages=[],
                created_at=datetime.now().timestamp(),
                updated_at=datetime.now().timestamp(),
                model_name=model_name
            )
            
            # 새 대화 저장 (새로운 사용자 ID 저장, 메시지는 비어있음.)
            await redis_client.save_chat_conversation(new_conversation)
        
        return current_conversation_id
        
    except Exception as e:
        logger.error(f"대화 처리 오류: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": f"대화 처리 오류: {str(e)}"}
        }))
        return None


