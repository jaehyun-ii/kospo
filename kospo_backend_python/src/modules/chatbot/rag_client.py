"""
RAG (Retrieval-Augmented Generation) 클라이언트 모듈
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGClient:
    """
    FAISS 벡터 스토어를 사용한 RAG 검색 클라이언트.
    센서 데이터, phase 정보 등 다양한 참고 정보를 검색할 수 있습니다.
    """
    
    def __init__(self, 
                 combustion_tag_path: str = "src/data/combustion_tag.json", 
                 vibration_tag_path: str = "src/data/vibration_tag.json",
                 phase_info_path: str = "src/data/phase_infomation.json",
                 index_path: str = "src/data/faiss_index"):
        """
        RAGClient 초기화.
        
        Args:
            combustion_tag_path: 연소 태그 데이터 JSON 파일 경로
            vibration_tag_path: 진동 태그 데이터 JSON 파일 경로
            phase_info_path: phase 정보 JSON 파일 경로
            index_path: FAISS 인덱스 저장 경로
        """
        self.combustion_tag_path = combustion_tag_path
        self.vibration_tag_path = vibration_tag_path
        self.phase_info_path = phase_info_path
        self.index_path = index_path
        self.vector_store = None
        self.reference_data = None
        self.embeddings = None
        
        logger.info("RAG 클라이언트 초기화 시작")
        
    async def initialize(self) -> None:
        """
        RAG 클라이언트를 초기화합니다.
        기존 인덱스가 있으면 로드하고, 없으면 새로 생성합니다.
        """
        try:
            # 참고 정보 로드
            self.reference_data = self._load_reference_data()
            logger.info(f"참고 정보 로드 완료: {len(self.reference_data)}개 항목")
            
            # # reference_data 저장 확인
            # with open('src/data/reference_data.json', 'w', encoding='utf-8') as file:
            #     json.dump(self.reference_data, file, ensure_ascii=False, indent=4)
            
            # 임베딩 모델 로드
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            logger.info("임베딩 모델 로드 완료")
            
            # 기존 인덱스 확인 및 로드/생성
            self.vector_store = await self._load_or_create_vector_store()
            
        except Exception as e:
            logger.error(f"RAG 클라이언트 초기화 실패: {e}")
            raise
    
    def _load_reference_data(self) -> Dict[str, Any]:
        """
        여러 JSON 파일에서 참고 정보를 로드하고 병합합니다.
        
        Returns:
            Dict[str, Any]: 병합된 참고 정보 딕셔너리
        """
        combined_data = {}
        
        # 연소 태그 데이터 로드
        try:
            with open(self.combustion_tag_path, 'r', encoding='utf-8') as file:
                combustion_data = json.load(file)
                # 연소 태그 데이터에 카테고리 정보 추가
                for sensor_id, sensor_info in combustion_data.items():
                    sensor_info['category'] = 'combustion'
                    sensor_info['type'] = 'sensor'
                    combined_data[sensor_id] = sensor_info
                logger.info(f"연소 태그 데이터 로드 완료: {len(combustion_data)}개 센서")
        except FileNotFoundError:
            logger.warning(f"연소 태그 데이터 파일을 찾을 수 없습니다: {self.combustion_tag_path}")
        except Exception as e:
            logger.error(f"연소 태그 데이터 로드 실패: {e}")
        
        # 진동 태그 데이터 로드
        try:
            with open(self.vibration_tag_path, 'r', encoding='utf-8') as file:
                vibration_data = json.load(file)
                # 진동 태그 데이터에 카테고리 정보 추가
                for sensor_id, sensor_info in vibration_data.items():
                    sensor_info['category'] = 'vibration'
                    sensor_info['type'] = 'sensor'
                    combined_data[sensor_id] = sensor_info
                logger.info(f"진동 태그 데이터 로드 완료: {len(vibration_data)}개 센서")
        except FileNotFoundError:
            logger.warning(f"진동 태그 데이터 파일을 찾을 수 없습니다: {self.vibration_tag_path}")
        except Exception as e:
            logger.error(f"진동 태그 데이터 로드 실패: {e}")
        
        # Phase 정보 로드
        try:
            with open(self.phase_info_path, 'r', encoding='utf-8') as file:
                phase_data = json.load(file)
                # Phase 정보에 카테고리 정보 추가
                for phase in phase_data.get('phases', []):
                    phase_number = phase['phase_number']
                    phase_id = f"phase_{phase_number}"
                    phase['category'] = 'phase'
                    phase['type'] = 'process'
                    combined_data[phase_id] = phase
                logger.info(f"Phase 정보 로드 완료: {len(phase_data.get('phases', []))}개 phase")
        except FileNotFoundError:
            logger.warning(f"Phase 정보 파일을 찾을 수 없습니다: {self.phase_info_path}")
        except Exception as e:
            logger.error(f"Phase 정보 로드 실패: {e}")
        
        return combined_data
    
    async def _load_or_create_vector_store(self) -> FAISS:
        """
        기존 FAISS 인덱스를 로드하거나 새로 생성합니다.
        
        Returns:
            FAISS: 로드되거나 생성된 벡터 스토어
        """
        try:
            # 기존 인덱스 확인
            if Path(self.index_path).exists():
                logger.info(f"기존 인덱스 로드 중: {self.index_path}")
                vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("기존 인덱스 로드 완료")
                return vector_store
            else:
                logger.info("기존 인덱스가 없습니다. 새로 생성합니다.")
                return await self._create_new_vector_store()
                
        except Exception as e:
            logger.warning(f"기존 인덱스 로드 실패: {e}")
            logger.info("새로운 인덱스를 생성합니다.")
            return await self._create_new_vector_store()
    
    async def _create_new_vector_store(self) -> FAISS:
        """
        새로운 벡터 스토어를 생성하고 로컬에 저장합니다.
        
        Returns:
            FAISS: 생성된 벡터 스토어
        """
        # Document 생성
        documents = self._create_documents()
        
        # 벡터 스토어 생성
        vector_store = FAISS.from_documents(
            documents=documents, 
            embedding=self.embeddings
        )
        
        # 로컬에 저장
        vector_store.save_local(self.index_path)
        logger.info(f"새로운 벡터 스토어 생성 및 저장 완료: {self.index_path}")
        
        return vector_store
    
    def _create_documents(self) -> List[Document]:
        """
        참고 정보로부터 Document 객체를 생성합니다.
        
        Returns:
            List[Document]: 생성된 Document 객체 목록
        """
        documents = []
        
        for item_id, item_info in self.reference_data.items():
            item_type = item_info.get('type', 'unknown')
            category = item_info.get('category', 'unknown')
            
            if item_type == 'sensor':
                # 센서 정보 처리
                category_korean = '연소' if category == 'combustion' else '진동' if category == 'vibration' else '기타'
                
                content = f"""
                센서 ID: {item_id}
                센서명: {item_id}
                카테고리: {category_korean}
                설명: {item_info['description']}
                단위: {item_info['unit']}
                
                이 센서는 {item_id}로 불리며, {category_korean} 관련 {item_info['description']}을 측정합니다.
                측정 단위는 {item_info['unit']}입니다.
                """
                
                document = Document(
                    page_content=content.strip(),
                    metadata={
                        "item_id": item_id,
                        "type": "sensor",
                        "category": category,
                        "category_korean": category_korean,
                        "description": item_info["description"],
                        "unit": item_info["unit"]
                    }
                )
                
            elif item_type == 'process':
                # Process 정보 처리 (Phase 등)
                if category == 'phase':
                    content = f"""
                    Phase 번호: {item_info['phase_number']}
                    Phase 이름: {item_info['name']}
                    설명: {item_info['description']}
                    조건: {item_info['condition']}
                    
                    이는 발전기 기동 과정의 {item_info['phase_number']}단계로, {item_info['name']}입니다.
                    {item_info['description']}
                    해당 phase의 조건은 {item_info['condition']}입니다.
                    """
                    
                    document = Document(
                        page_content=content.strip(),
                        metadata={
                            "item_id": item_id,
                            "type": "process",
                            "category": "phase",
                            "phase_number": item_info["phase_number"],
                            "name": item_info["name"],
                            "description": item_info["description"],
                            "condition": item_info["condition"]
                        }
                    )
                else:
                    # 기타 process 정보
                    content = f"""
                    항목 ID: {item_id}
                    카테고리: {category}
                    정보: {item_info}
                    """
                    
                    document = Document(
                        page_content=content.strip(),
                        metadata={
                            "item_id": item_id,
                            "type": "process",
                            "category": category,
                            "info": item_info
                        }
                    )
            else:
                # 기타 정보 (미정)
                content = f"""
                항목 ID: {item_id}
                타입: {item_type}
                카테고리: {category}
                정보: {item_info}
                """
                
                document = Document(
                    page_content=content.strip(),
                    metadata={
                        "item_id": item_id,
                        "type": item_type,
                        "category": category,
                        "info": item_info
                    }
                )
            
            documents.append(document)
        
        return documents
    
    async def search(self, query: str, k: int = 5) -> str:
        """
        사용자 질문에 대한 관련 정보를 검색합니다.
        
        Args:
            query: 사용자 질문
            k: 검색할 문서 수
            
        Returns:
            str: 검색 결과를 포맷팅한 문자열
        """
        if not self.vector_store:
            logger.warning("벡터 스토어가 초기화되지 않았습니다.")
            return ""
        
        try:
            # 벡터 검색 수행
            results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                logger.info("검색 결과가 없습니다.")
                return ""
            
            # 검색 결과 포맷팅 (리턴 텍스트)
            formatted_results = []
            for i, doc in enumerate(results, 1):
                item_id = doc.metadata.get('item_id', 'Unknown')
                item_type = doc.metadata.get('type', 'unknown')
                
                if item_type == 'sensor':
                    category_korean = doc.metadata.get('category_korean', '기타')
                    description = doc.metadata.get('description', 'No description')
                    unit = doc.metadata.get('unit', 'No unit')
                    
                    formatted_results.append(
                        f"{i}. 센서 ID: {item_id}\n"
                        f"   카테고리: {category_korean}\n"
                        f"   설명: {description}\n"
                        f"   단위: {unit}"
                    )
                elif item_type == 'process':
                    if doc.metadata.get('category') == 'phase':
                        phase_number = doc.metadata.get('phase_number', 'Unknown')
                        name = doc.metadata.get('name', 'No name')
                        description = doc.metadata.get('description', 'No description')
                        condition = doc.metadata.get('condition', 'No condition')
                        
                        formatted_results.append(
                            f"{i}. Phase {phase_number}: {name}\n"
                            f"   설명: {description}\n"
                            f"   조건: {condition}"
                        )
                    else:
                        category = doc.metadata.get('category', '기타')
                        info = doc.metadata.get('info', {})
                        
                        formatted_results.append(
                            f"{i}. Process ID: {item_id}\n"
                            f"   카테고리: {category}\n"
                            f"   정보: {info}"
                        )
                else:
                    category = doc.metadata.get('category', '기타')
                    info = doc.metadata.get('info', {})
                    
                    formatted_results.append(
                        f"{i}. 항목 ID: {item_id}\n"
                        f"   타입: {item_type}\n"
                        f"   카테고리: {category}\n"
                        f"   정보: {info}"
                    )
            
            # 센서 ID 직접 매칭 시도
            import re
            sensor_id_match = re.search(r'11G_[A-Z0-9_]+', query)
            if sensor_id_match:
                sensor_id = sensor_id_match.group()
                if sensor_id in self.reference_data:
                    exact_sensor = self.reference_data[sensor_id]
                    if exact_sensor.get('type') == 'sensor':
                        category_korean = '연소' if exact_sensor.get('category') == 'combustion' else '진동' if exact_sensor.get('category') == 'vibration' else '기타'
                        exact_info = (
                            f"정확한 센서 정보:\n"
                            f"센서 ID: {sensor_id}\n"
                            f"카테고리: {category_korean}\n"
                            f"설명: {exact_sensor['description']}\n"
                            f"단위: {exact_sensor['unit']}"
                        )
                        formatted_results.insert(0, exact_info)
            
            # Phase 번호 직접 매칭 시도
            phase_match = re.search(r'phase\s*(\d+)', query, re.IGNORECASE)
            if phase_match:
                phase_number = int(phase_match.group(1))
                phase_id = f"phase_{phase_number}"
                if phase_id in self.reference_data:
                    exact_phase = self.reference_data[phase_id]
                    exact_info = (
                        f"정확한 Phase 정보:\n"
                        f"Phase {phase_number}: {exact_phase['name']}\n"
                        f"설명: {exact_phase['description']}\n"
                        f"조건: {exact_phase['condition']}"
                    )
                    formatted_results.insert(0, exact_info)
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"RAG 검색 실패: {e}")
            return "" 