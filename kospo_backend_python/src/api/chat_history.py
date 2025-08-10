"""
채팅 세션 관리 API 엔드포인트 모듈.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from src.modules.chatbot.redis_client import ChatRedisClient, ChatConversation, ChatMessage

logger = logging.getLogger(__name__)

# API 라우터 생성
router = APIRouter(prefix="/api/v1/chat-history", tags=["chat-history"])

# Redis 클라이언트 인스턴스
redis_client = ChatRedisClient()


@router.websocket("/ws/conversations")
async def websocket_conversations_endpoint(websocket: WebSocket):
    """
    채팅 대화 관리 WebSocket 엔드포인트.
    
    지원하는 작업:
    - 대화 생성
    - 대화 불러오기
    - 대화 목록 조회
    - 대화 삭제
    - 메시지 추가
    
    Args:
        websocket: WebSocket 연결 객체
    """
    await websocket.accept()
    logger.info("채팅 세션 WebSocket 연결이 수락되었습니다.")
    
    # Redis 연결
    try:
        await redis_client.connect()
    except Exception as e:
        logger.error(f"Redis 연결 실패: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": "Redis 연결 실패"}
        }))
        return
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            
            try:
                # JSON 파싱
                request_data = json.loads(data)
                action = request_data.get("action")
                
                if not action:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": "action이 지정되지 않았습니다."}
                    }))
                    continue
                
                # 액션별 처리
                if action == "create_conversation":
                    await _handle_create_conversation(websocket, request_data)
                elif action == "load_conversation":
                    await _handle_load_conversation(websocket, request_data)
                elif action == "list_conversations":
                    await _handle_list_conversations(websocket, request_data)
                elif action == "delete_conversation":
                    await _handle_delete_conversation(websocket, request_data)
                elif action == "add_message":
                    await _handle_add_message(websocket, request_data)
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"error": f"지원하지 않는 액션: {action}"}
                    }))
                    
            except json.JSONDecodeError:
                logger.warning("잘못된 JSON 형식의 메시지 수신")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"error": "잘못된 JSON 형식입니다."}
                }))
            except Exception as e:
                logger.error(f"세션 관리 처리 오류: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"error": f"서버 오류: {str(e)}"}
                }))
                
    except WebSocketDisconnect:
        logger.info("채팅 세션 WebSocket 연결이 종료되었습니다.")
    except Exception as e:
        logger.error(f"채팅 세션 WebSocket 오류: {e}")
    finally:
        # Redis 연결 종료
        await redis_client.disconnect()


async def _handle_create_conversation(websocket: WebSocket, request_data: Dict) -> None:
    """새 채팅 대화를 생성합니다."""
    try:
        user_id = request_data.get("user_id")
        conversation_id = request_data.get("conversation_id")
        model_name = request_data.get("model_name", "openai/gpt-oss-120b")
        
        if not user_id:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "user_id가 필요합니다."}
            }))
            return
        
        # 대화 ID 생성
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # 새 대화 생성
        conversation = ChatConversation(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[],
            created_at=datetime.now().timestamp(),
            updated_at=datetime.now().timestamp(),
            model_name=model_name
        )
        
        # Redis에 저장
        success = await redis_client.save_chat_conversation(conversation)
        
        if success:
            await websocket.send_text(json.dumps({
                "type": "conversation_created",
                "data": {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "model_name": model_name,
                    "created_at": conversation.created_at
                }
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "대화 생성에 실패했습니다."}
            }))
            
    except Exception as e:
        logger.error(f"대화 생성 오류: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": f"대화 생성 오류: {str(e)}"}
        }))


async def _handle_load_conversation(websocket: WebSocket, request_data: Dict) -> None:
    """채팅 대화를 불러옵니다."""
    try:
        conversation_id = request_data.get("conversation_id")
        
        if not conversation_id:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "conversation_id가 필요합니다."}
            }))
            return
        
        # 대화 불러오기
        conversation = await redis_client.load_chat_conversation(conversation_id)
        
        if conversation:
            # 메시지 데이터 변환
            messages_data = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "message_id": msg.message_id
                }
                for msg in conversation.messages
            ]
            
            await websocket.send_text(json.dumps({
                "type": "conversation_loaded",
                "data": {
                    "conversation_id": conversation.conversation_id,
                    "user_id": conversation.user_id,
                    "model_name": conversation.model_name,
                    "messages": messages_data,
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at
                }
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "대화를 찾을 수 없습니다."}
            }))
            
    except Exception as e:
        logger.error(f"대화 불러오기 오류: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": f"대화 불러오기 오류: {str(e)}"}
        }))


async def _handle_list_conversations(websocket: WebSocket, request_data: Dict) -> None:
    """사용자의 대화 목록을 조회합니다."""
    try:
        user_id = request_data.get("user_id")
        
        if not user_id:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "user_id가 필요합니다."}
            }))
            return
        
        # 대화 ID 목록 조회
        conversation_ids = await redis_client.get_user_conversations(user_id)
        
        # 각 대화의 상세 정보 조회
        conversations_data = []
        for conversation_id in conversation_ids:
            conversation = await redis_client.load_chat_conversation(conversation_id)
            if conversation:
                conversations_data.append({
                    "conversation_id": conversation.conversation_id,
                    "user_id": conversation.user_id,
                    "model_name": conversation.model_name,
                    "message_count": len(conversation.messages),
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at,
                    "last_message": conversation.messages[-1].content if conversation.messages else None
                })
        
        await websocket.send_text(json.dumps({
            "type": "conversations_listed",
            "data": {
                "user_id": user_id,
                "conversations": conversations_data,
                "total_count": len(conversations_data)
            }
        }))
        
    except Exception as e:
        logger.error(f"대화 목록 조회 오류: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": f"대화 목록 조회 오류: {str(e)}"}
        }))


async def _handle_delete_conversation(websocket: WebSocket, request_data: Dict) -> None:
    """채팅 대화를 삭제합니다."""
    try:
        conversation_id = request_data.get("conversation_id")
        
        if not conversation_id:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "conversation_id가 필요합니다."}
            }))
            return
        
        # 대화 삭제
        success = await redis_client.delete_conversation(conversation_id)
        
        if success:
            await websocket.send_text(json.dumps({
                "type": "conversation_deleted",
                "data": {
                    "conversation_id": conversation_id,
                    "message": "대화가 성공적으로 삭제되었습니다."
                }
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "대화 삭제에 실패했습니다."}
            }))
            
    except Exception as e:
        logger.error(f"대화 삭제 오류: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": f"대화 삭제 오류: {str(e)}"}
        }))


async def _handle_add_message(websocket: WebSocket, request_data: Dict) -> None:
    """대화에 새 메시지를 추가합니다."""
    try:
        conversation_id = request_data.get("conversation_id")
        role = request_data.get("role")
        content = request_data.get("content")
        message_id = request_data.get("message_id")
        
        if not all([conversation_id, role, content]):
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "conversation_id, role, content가 모두 필요합니다."}
            }))
            return
        
        if role not in ["user", "assistant"]:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "role은 'user' 또는 'assistant'여야 합니다."}
            }))
            return
        
        # 메시지 추가
        success = await redis_client.add_message_to_conversation(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_id=message_id
        )
        
        if success:
            await websocket.send_text(json.dumps({
                "type": "message_added",
                "data": {
                    "conversation_id": conversation_id,
                    "role": role,
                    "content": content,
                    "message_id": message_id,
                    "timestamp": datetime.now().timestamp()
                }
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"error": "메시지 추가에 실패했습니다."}
            }))
            
    except Exception as e:
        logger.error(f"메시지 추가 오류: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": {"error": f"메시지 추가 오류: {str(e)}"}
        })) 