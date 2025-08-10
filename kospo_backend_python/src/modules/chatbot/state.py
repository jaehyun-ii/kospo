"""
챗봇 워크플로우에서 사용하는 상태 관리 모듈.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

@dataclass
class ChatbotState:
    """
    챗봇 워크플로우의 상태를 관리하는 데이터 클래스.
    
    Attributes:
        messages: 사용자와 시스템 간의 대화 메시지 목록
        user_input: 사용자가 입력한 원본 메시지
        final_answer: 최종 생성된 답변
        rag_context: RAG 검색 결과 컨텍스트
        is_dashboard_question: 남부발전 대시보드 관련 질문 여부
        metadata: 추가 메타데이터를 저장하는 딕셔너리
    """
    
    messages: List[BaseMessage] = field(default_factory=list)
    user_input: str = ""
    conversation_id: Optional[str] = None
    final_answer: str = ""
    rag_context: str = ""
    is_dashboard_question: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)