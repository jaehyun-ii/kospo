"""
Redis 클라이언트 모듈.
- 채팅 세션과 사용자 정보를 Redis에 저장하고 불러오는 기능을 제공
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import redis.asyncio as redis
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """채팅 메시지 데이터 클래스."""
    role: str  # 'user' 또는 'assistant'
    content: str
    timestamp: float
    message_id: Optional[str] = None


@dataclass
class ChatConversation:
    """채팅 대화 데이터 클래스."""
    conversation_id: str
    user_id: str
    messages: List[ChatMessage]
    created_at: float
    updated_at: float
    model_name: str = "openai/gpt-oss-120b"


class ChatRedisClient:
    """Redis를 사용한 채팅 저장/불러오기 클라이언트."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Redis 클라이언트를 초기화합니다.
        
        Args:
            redis_url: Redis 연결 URL (선택사항, 기본값은 환경변수 사용)
        """
        if redis_url:
            self.redis_url = redis_url
        else:
            # 환경변수에서 Redis 설정 가져오기
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = os.getenv("REDIS_PORT", "6379")
            redis_db = os.getenv("REDIS_DB", "0")
            self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
            
        self.redis_client: Optional[redis.Redis] = None
        self.conversation_ttl = 86400 * 7  # 7일 (초 단위)
        logger.info(f"Redis URL 설정: {self.redis_url}")
        
    async def connect(self) -> None:
        """Redis에 연결합니다."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                encoding="utf-8"
            )
            # 연결 테스트
            await self.redis_client.ping()
            logger.info("Redis 연결 성공")
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Redis 연결을 종료합니다."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis 연결 종료")
    
    def _get_conversation_key(self, conversation_id: str) -> str:
        """대화 키를 생성합니다."""
        return f"chat:conversation:{conversation_id}"
    
    def _get_user_conversations_key(self, user_id: str) -> str:
        """사용자 대화 목록 키를 생성합니다."""
        return f"chat:user_conversations:{user_id}"
    
    async def save_chat_conversation(self, conversation: ChatConversation) -> bool:
        """
        채팅 대화를 Redis에 저장합니다.
        
        Args:
            conversation: 저장할 채팅 대화
            
        Returns:
            bool: 저장 성공 여부
        """
        if not self.redis_client:
            logger.error("Redis 클라이언트가 초기화되지 않았습니다.")
            return False
        
        try:
            # 대화 데이터를 JSON으로 직렬화
            conversation_data = {
                "conversation_id": conversation.conversation_id,
                "user_id": conversation.user_id,
                "messages": [asdict(msg) for msg in conversation.messages],
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "model_name": conversation.model_name
            }
            
            # 대화 저장
            conversation_key = self._get_conversation_key(conversation.conversation_id)
            await self.redis_client.setex(
                conversation_key,
                self.conversation_ttl,
                json.dumps(conversation_data, ensure_ascii=False)
            )
            
            # 사용자 대화 목록에 추가
            user_conversations_key = self._get_user_conversations_key(conversation.user_id)
            await self.redis_client.sadd(user_conversations_key, conversation.conversation_id)
            await self.redis_client.expire(user_conversations_key, self.conversation_ttl)
            
            logger.info(f"채팅 대화 저장 완료: {conversation.conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"채팅 대화 저장 실패: {e}")
            return False
    
    async def load_chat_conversation(self, conversation_id: str) -> Optional[ChatConversation]:
        """
        채팅 대화를 Redis에서 불러옵니다.
        
        Args:
            conversation_id: 불러올 대화 ID
            
        Returns:
            Optional[ChatConversation]: 불러온 대화 또는 None
        """
        if not self.redis_client:
            logger.error("Redis 클라이언트가 초기화되지 않았습니다.")
            return None
        
        try:
            conversation_key = self._get_conversation_key(conversation_id)
            conversation_data = await self.redis_client.get(conversation_key)
            
            if not conversation_data:
                logger.warning(f"대화를 찾을 수 없습니다: {conversation_id}")
                return None
            
            # JSON 역직렬화
            data = json.loads(conversation_data)
            
            # 메시지 리스트 복원
            messages = [
                ChatMessage(**msg_data) for msg_data in data["messages"]
            ]
            
            conversation = ChatConversation(
                conversation_id=data["conversation_id"],
                user_id=data["user_id"],
                messages=messages,
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                model_name=data.get("model_name", "openai/gpt-oss-120b")
            )
            
            logger.info(f"채팅 대화 불러오기 완료: {conversation_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"채팅 대화 불러오기 실패: {e}")
            return None
    
    async def get_user_conversations(self, user_id: str) -> List[str]:
        """
        사용자의 모든 대화 ID 목록을 반환합니다.
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            List[str]: 대화 ID 목록
        """
        if not self.redis_client:
            logger.error("Redis 클라이언트가 초기화되지 않았습니다.")
            return []
        
        try:
            user_conversations_key = self._get_user_conversations_key(user_id)
            conversation_ids = await self.redis_client.smembers(user_conversations_key)
            return list(conversation_ids)
            
        except Exception as e:
            logger.error(f"사용자 대화 목록 조회 실패: {e}")
            return []
    
    async def add_message_to_conversation(
        self,
        conversation_id: str,
        role: str,
        content: str,
        message_id: Optional[str] = None
    ) -> bool:
        """
        대화에 새 메시지를 추가합니다.
        
        Args:
            conversation_id: 대화 ID
            role: 메시지 역할 ('user' 또는 'assistant')
            content: 메시지 내용
            message_id: 메시지 ID (선택사항)
            
        Returns:
            bool: 추가 성공 여부
        """
        if not self.redis_client:
            logger.error("Redis 클라이언트가 초기화되지 않았습니다.")
            return False
        
        try:
            # 기존 대화 불러오기
            conversation = await self.load_chat_conversation(conversation_id)
            if not conversation:
                logger.error(f"대화를 찾을 수 없습니다: {conversation_id}")
                return False
            
            # 새 메시지 추가
            new_message = ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.now().timestamp(),
                message_id=message_id
            )
            
            conversation.messages.append(new_message)
            conversation.updated_at = datetime.now().timestamp()
            
            # 대화 다시 저장
            return await self.save_chat_conversation(conversation)
            
        except Exception as e:
            logger.error(f"메시지 추가 실패: {e}")
            return False
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        채팅 대화를 삭제합니다.
        
        Args:
            conversation_id: 삭제할 대화 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        if not self.redis_client:
            logger.error("Redis 클라이언트가 초기화되지 않았습니다.")
            return False
        
        try:
            # 대화 정보 먼저 불러오기
            conversation = await self.load_chat_conversation(conversation_id)
            if not conversation:
                logger.warning(f"삭제할 대화를 찾을 수 없습니다: {conversation_id}")
                return True  # 이미 없는 경우 성공으로 처리
            
            # 대화 삭제
            conversation_key = self._get_conversation_key(conversation_id)
            await self.redis_client.delete(conversation_key)
            
            # 사용자 대화 목록에서 제거
            user_conversations_key = self._get_user_conversations_key(conversation.user_id)
            await self.redis_client.srem(user_conversations_key, conversation_id)
            
            logger.info(f"채팅 대화 삭제 완료: {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"채팅 대화 삭제 실패: {e}")
            return False 