"""
랭그래프 기반 챗봇 워크플로우
"""

import json
import logging
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from src.modules.chatbot.state import ChatbotState
from src.modules.chatbot.llm_client import LLMClient
from src.modules.chatbot.redis_client import ChatRedisClient
from src.modules.chatbot.prompt import ChatbotPrompts
from src.modules.chatbot.rag_client import RAGClient

logger = logging.getLogger(__name__)


class ChatbotWorkflow:
    """
    LangGraph 기반 챗봇 워크플로우 클래스.
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        openrouter_api_key: Optional[str] = None,
        websocket=None,
        redis_client: Optional[ChatRedisClient] = None
    ):
        """
        ChatbotWorkflow 초기화.
        
        Args:
            model_name: 사용할 LLM 모델 이름
            temperature: 생성 다양성 조절 파라미터
            openrouter_api_key: OpenRouter API 키
            websocket: WebSocket 연결 객체 (실시간 응답 전송용)
            redis_client: Redis 클라이언트 (대화 기록 저장/불러오기용)
        """
        self.llm_client = LLMClient(
            model_name=model_name,
            temperature=temperature,
            openrouter_api_key=openrouter_api_key
        )
        self.websocket = websocket
        self.redis_client = redis_client
        self.rag_client = RAGClient()
        self.graph = self._build_graph()
        
        logger.info("챗봇 워크플로우 초기화 완료")

    # 최종 메시지 처리 함수
    async def process_message(
        self,
        user_message: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        사용자 메시지를 처리하여 워크플로우를 실행합니다.
        
        Args:
            user_message: 사용자가 입력한 메시지
            conversation_id: 대화 ID (체크포인트용)
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 초기 상태 설정
            initial_state = ChatbotState(
                user_input=user_message,
                conversation_id=conversation_id,
                messages=[HumanMessage(content=user_message)]
            )
            
            # 워크플로우 실행 (단순화)
            final_state = None
            async for event in self.graph.astream(initial_state):
                # 각 노드의 실행 결과를 로깅
                for node_name, node_output in event.items():
                    if node_name != "__end__":
                        logger.info(f"노드 {node_name} 실행 완료")
                        final_state = node_output
            
            # 안전한 상태 접근
            try:
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "final_answer": getattr(final_state, 'final_answer', ''),
                    "messages": [msg.content for msg in getattr(final_state, 'messages', [])]
                }
            except Exception as state_error:
                logger.error(f"상태 접근 오류: {state_error}")
                return {
                    "success": False,
                    "error": f"상태 접근 오류: {str(state_error)}",
                    "conversation_id": conversation_id
                }
            
        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "conversation_id": conversation_id
            } 
            
    def _build_graph(self) -> StateGraph:
        """
        LangGraph 워크플로우를 구성합니다.
        
        Returns:
            StateGraph: 구성된 워크플로우 그래프
        """
        # 상태 그래프 생성
        workflow = StateGraph(ChatbotState)
        
        # 노드 추가
        workflow.add_node("classify_node", self._classify_node)
        workflow.add_node("rag_node", self._rag_node)
        workflow.add_node("answer_node", self._answer_node)
        
        # 엣지 연결
        workflow.add_edge(START, "classify_node")
        workflow.add_conditional_edges(
            "classify_node",
            self._route_after_classification,
            {
                "dashboard": "rag_node",
                "general": "answer_node"
            }
        )
        workflow.add_edge("rag_node", "answer_node")
        workflow.add_edge("answer_node", END)
        
        # 체크포인트 설정 (단순화)
        return workflow.compile()
        
    async def _send_websocket_message(self, message_type: str, data: Dict[str, Any]) -> None:
        """
        WebSocket을 통해 메시지를 전송합니다.
        
        Args:
            message_type: 메시지 타입
            data: 전송할 데이터
        """
        if not self.websocket:
            logger.debug(f"WebSocket이 없습니다. 메시지: {message_type} - {data}")
            return
            
        try:
            event_data = {
                "type": message_type,
                "data": data,
                "timestamp": self._get_timestamp()
            }
            
            await self.websocket.send_text(json.dumps(event_data, ensure_ascii=False))
            
        except Exception as e:
            # WebSocket 연결 오류 시 더 자세한 로깅
            if "close message has been sent" in str(e):
                logger.debug(f"WebSocket 연결이 이미 닫혔습니다: {message_type}")
            else:
                logger.warning(f"WebSocket 메시지 전송 실패: {e}")
    
    def _get_timestamp(self) -> float:
        """현재 타임스탬프를 반환합니다."""
        import time
        return time.time()

    async def _classify_node(self, state: ChatbotState) -> ChatbotState:
        """
        질문 분류 노드.
        
        사용자 질문이 남부발전 대시보드 관련인지 일반 질의응답인지 판단합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            ChatbotState: 업데이트된 상태
        """
        try:
            logger.info("질문 분류 노드 시작")
            
            # 노드 시작 알림
            await self._send_websocket_message("stream_start", {
                "node": "질문 분류",
                "message": "질문을 분석하고 있습니다..."
            })
            
            # 질문 분류를 위한 프롬프트 생성
            classification_prompt = ChatbotPrompts.get_classification_prompt(state.user_input)

            # 분류 수행
            classification_messages = [SystemMessage(content=classification_prompt)]
            classification_result = await self.llm_client.generate_response(classification_messages)
            
            # 결과 파싱
            is_dashboard = "dashboard" in classification_result.lower().strip()
            state.is_dashboard_question = is_dashboard
            
            # 분류 완료 알림
            question_type = "남부발전 대시보드" if is_dashboard else "일반 질의응답"
            await self._send_websocket_message("stream_end", {
                "node": "질문 분류",
                "message": f"질문 유형: {question_type}"
            })
            
            logger.info(f"질문 분류 완료: {question_type}")
            return state
            
        except Exception as e:
            logger.error(f"질문 분류 노드 오류: {e}")
            
            # 오류 시 기본값으로 대시보드 질문으로 설정
            state.is_dashboard_question = True
            
            # 오류 메시지 전송
            await self._send_websocket_message("error", {
                "node": "질문 분류",
                "error": str(e),
                "message": "질문 분류 중 오류가 발생했습니다."
            })
            
            return state

    def _route_after_classification(self, state: ChatbotState) -> str:
        """
        분류 후 라우팅 함수.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            str: 다음 노드 이름
        """
        if state.is_dashboard_question:
            return "dashboard"
        else:
            return "general"
        
    async def _rag_node(self, state: ChatbotState) -> ChatbotState:
        """
        RAG 검색 노드.
        
        사용자 질문에 대한 관련 정보를 검색하여 컨텍스트를 생성합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            ChatbotState: 업데이트된 상태
        """
        try:
            logger.info("RAG 검색 노드 시작")
            
            # 노드 시작 알림
            await self._send_websocket_message("stream_start", {
                "node": "RAG 검색",
                "message": "관련 정보를 검색하고 있습니다..."
            })
            
            # RAG 클라이언트 초기화 (필요한 경우)
            if not self.rag_client.vector_store:
                await self.rag_client.initialize()
            
            # 사용자 질문으로 검색 수행
            rag_context = await self.rag_client.search(state.user_input, k=5)
            # logger.info(f"RAG 검색 결과: {rag_context}")
            
            # 상태에 RAG 컨텍스트 저장
            state.rag_context = rag_context
            
            # 검색 완료 알림
            if rag_context:
                # 검색 결과 개수 계산
                result_count = len([line for line in rag_context.split('\n') if line.strip().startswith('센서 ID:')])
                await self._send_websocket_message("stream_end", {
                    "node": "RAG 검색",
                    "message": f"관련 정보 {result_count}개를 찾았습니다."
                })
            else:
                await self._send_websocket_message("stream_end", {
                    "node": "RAG 검색",
                    "message": "관련 정보를 찾지 못했습니다."
                })
            
            logger.info("RAG 검색 완료")
            return state
            
        except Exception as e:
            logger.error(f"RAG 검색 노드 오류: {e}")
            
            # 오류 메시지 전송
            await self._send_websocket_message("error", {
                "node": "RAG 검색",
                "error": str(e),
                "message": "정보 검색 중 오류가 발생했습니다."
            })
            
            # 오류가 발생해도 빈 컨텍스트로 계속 진행
            state.rag_context = ""
            return state
            
    async def _answer_node(self, state: ChatbotState) -> ChatbotState:
        """
        최종 답변 생성 노드.
        
        위 단계에서 수집한 정보를 기반으로 최종 답변을 생성합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            ChatbotState: 업데이트된 상태
        """
        try:
            logger.info("답변 생성 노드 시작")
            
            # 노드 시작 알림
            await self._send_websocket_message("stream_start", {
                "node": "답변",
                "message": "답변을 생성하고 있습니다..."
            })
            
            # 이전 대화 기록 불러오기
            conversation_history = await self._load_conversation_history(state.conversation_id)
            
            # 답변 생성을 위한 메시지 생성
            messages = await self._create_answer_messages(
                user_message=state.user_input,
                conversation_history=conversation_history,
                rag_context=state.rag_context,
                is_dashboard_question=state.is_dashboard_question
            )
            
            # 토큰 콜백 함수 정의
            async def token_callback(token: str, partial_text: str) -> None:
                await self._send_websocket_message("token", {
                    "node": "답변",
                    "token": token,
                    "partial_text": partial_text
                })
            
            # 최종 답변 생성 (토큰별 스트리밍)
            final_answer = await self.llm_client.generate_streaming_response(
                messages=messages,
                on_token_callback=token_callback
            )
            
            # 상태 업데이트
            state.final_answer = final_answer
            state.messages.append(AIMessage(content=final_answer))
            
            # 최종 답변을 사용자에게 전송
            await self._send_websocket_message("stream_end", {
                "node": "답변",
                "final_text": final_answer,
                "message": "답변이 완료되었습니다."
            })
            
            logger.info("답변 생성 완료")
            return state
            
        except Exception as e:
            logger.error(f"답변 생성 노드 오류: {e}")
            
            # 오류 메시지 전송
            await self._send_websocket_message("error", {
                "node": "답변",
                "error": str(e),
                "message": "답변 생성 중 오류가 발생했습니다."
            })
            
            error_message = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
            state.final_answer = error_message
            state.messages.append(AIMessage(content=error_message))
            return state

    async def _load_conversation_history(self, conversation_id: Optional[str]) -> Optional[List[BaseMessage]]:
        """
        Redis에서 이전 대화 기록을 불러와 LangChain 메시지 형태로 변환합니다.
        
        Args:
            conversation_id: 대화 ID
            
        Returns:
            Optional[List[BaseMessage]]: 변환된 대화 기록 메시지 목록, 없으면 None
        """
        if not self.redis_client or not conversation_id:
            return None
            
        try:
            conversation = await self.redis_client.load_chat_conversation(conversation_id)
            if not conversation or not conversation.messages:
                return None
                
            # Redis 메시지를 LangChain 메시지로 변환
            conversation_history = []
            for msg in conversation.messages:
                if msg.role == "user":
                    conversation_history.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    conversation_history.append(AIMessage(content=msg.content))
                    
            return conversation_history
            
        except Exception as e:
            logger.warning(f"대화 기록 불러오기 실패: {e}")
            return None
                    
    async def _create_answer_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[BaseMessage]] = None,
        rag_context: Optional[str] = None,
        is_dashboard_question: bool = True
    ) -> List[BaseMessage]:
        """
        답변 생성을 위한 메시지 목록을 생성합니다.
        
        Args:
            user_message: 사용자 원본 메시지
            conversation_history: 이전 대화 기록 (멀티턴 대화용)
            rag_context: RAG 검색 결과 컨텍스트
            is_dashboard_question: 남부발전 대시보드 관련 질문 여부
            
        Returns:
            List[BaseMessage]: 답변 생성을 위한 메시지 목록
        """
        # 질문 타입에 따른 시스템 프롬프트 가져오기
        system_prompt = ChatbotPrompts.get_system_prompt(
            rag_context=rag_context, 
            is_dashboard_question=is_dashboard_question
        )

        # 메시지 목록 구성
        messages = [SystemMessage(content=system_prompt)]
        
        # 이전 대화 기록이 있으면 추가 (최근 10개 메시지로 제한)
        if conversation_history:
            # 최근 10개 메시지만 사용하여 토큰 제한 방지
            recent_messages = conversation_history[-10:]
            messages.extend(recent_messages)
            logger.info(f"이전 대화 기록 {len(recent_messages)}개 메시지 추가됨")
        
        # 현재 사용자 메시지 추가
        messages.append(HumanMessage(content=user_message))
        
        return messages
        
