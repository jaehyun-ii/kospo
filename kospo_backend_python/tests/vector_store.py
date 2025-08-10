import asyncio
import json
from typing import Dict, List
from langchain_openai import OpenAIEmbeddings   
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

def load_sensor_data(json_file_path: str) -> Dict:
    """JSON 파일에서 센서 데이터 로드"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def create_documents(sensor_data: Dict) -> List[Document]:
    """센서 데이터로부터 Document 객체 생성"""
    documents = []
    
    for sensor_id, sensor_info in sensor_data.items():
        # 센서 ID를 더 강조하고 다양한 표현으로 포함
        content = f"""
        센서 ID: {sensor_id}
        센서명: {sensor_id}
        설명: {sensor_info['description']}
        단위: {sensor_info['unit']}
        
        이 센서는 {sensor_id}로 불리며, {sensor_info['description']}을 측정합니다.
        측정 단위는 {sensor_info['unit']}입니다.
        """
        
        document = Document(
            page_content=content.strip(),
            metadata={
                "sensor_id": sensor_id,
                "description": sensor_info["description"],
                "unit": sensor_info["unit"]
            }
        )
        documents.append(document)
    
    return documents


async def test_faiss_rag():
    """FAISS RAG 간단 테스트 - 데이터 로드, 벡터 스토어 생성, 유사도 검색"""
    print("🚀 FAISS RAG 테스트 시작")
    print("=" * 50)
    
    # 1. 센서 데이터 로드
    print("1. 센서 데이터 로드 중...")
    sensor_data = load_sensor_data("src/data/sensor_data.json")
    print(f"   로드된 센서 수: {len(sensor_data)}")
    
    # 2. Document 생성
    print("2. Document 생성 중...")
    documents = create_documents(sensor_data)
    print(f"   생성된 Document 수: {len(documents)}")
    
    # 3. 임베딩 모델 로드
    print("3. 임베딩 모델 로드 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # 4. FAISS 벡터 스토어 생성
    print("4. FAISS 벡터 스토어 생성 중...")
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    print("   벡터 스토어 생성 완료!")
    
    # 5. 테스트 질문들
    test_queries = [
        "압축비 (Compressor Pressure Ratio)",
        "11G_CPR 이거뭐임?",
        "11G_A_96KP05_F1 이거뭐임?"
    ]
    
    print("\n5. 유사도 검색 테스트")
    print("-" * 30)
    
    for query in test_queries:
        print(f"\n질문: '{query}'")
        
        # 센서 ID 추출 (11G_로 시작하는 부분)
        import re
        sensor_id_match = re.search(r'11G_[A-Z0-9_]+', query)
        if sensor_id_match:
            sensor_id = sensor_id_match.group()
            print(f"  추출된 센서 ID: {sensor_id}")
            
            # 정확한 센서 ID로 직접 검색
            if sensor_id in sensor_data:
                exact_sensor = sensor_data[sensor_id]
                print(f"  ✅ 정확한 센서 정보: {sensor_id} - {exact_sensor['description']} ({exact_sensor['unit']})")
            else:
                print(f"  ❌ 센서 ID를 찾을 수 없음: {sensor_id}")
        
        # 벡터 검색 결과
        results = vector_store.similarity_search(query, k=5)
        print("  벡터 검색 결과:")
        for i, doc in enumerate(results, 1):
            print(f"    {i}. {doc.metadata['sensor_id']} - {doc.metadata['description']}")
    
    print("\n✅ FAISS RAG 테스트 완료!")


async def main():
    """메인 함수"""
    await test_faiss_rag()


if __name__ == "__main__":
    asyncio.run(main())
