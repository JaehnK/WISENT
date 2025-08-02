import argparse
import pandas as pd
from colorama import init, Fore, Style

# 리팩토링된 서비스들을 임포트합니다.
from core.services.main_service import MainService
from core.services.graph_service import GraphService

def main():
    """메인 실행 함수"""
    init(autoreset=True)

    parser = argparse.ArgumentParser(description="WISENT: Word Graph-based Topic-Sentence Extraction")
    parser.add_argument("-f", "--file", required=True, help="Input CSV file path containing text data.")
    parser.add_argument("-c", "--column", required=True, help="The name of the column with the text data.")
    parser.add_argument("-n", "--top_nodes", type=int, default=100, help="Number of top words to use as graph nodes.")
    parser.add_argument("-o", "--output", default="word_graph.png", help="Path to save the output graph visualization.")

    args = parser.parse_args()

    print(Fore.CYAN + "--- 1. Loading Data ---")
    try:
        df = pd.read_csv(args.file)
        if args.column not in df.columns:
            raise ValueError(f"Column '{args.column}' not found in the CSV file.")
        # 결측치가 있는 행은 무시하고 텍스트 데이터만 리스트로 추출
        raw_docs = df[args.column].dropna().astype(str).tolist()
        print(f"Successfully loaded {len(raw_docs)} documents from '{args.file}'.")
    except Exception as e:
        print(Fore.RED + f"Error loading data: {e}")
        return

    print(Fore.CYAN + "--- 2. Processing Documents ---")
    # MainService를 사용하여 문서 처리 파이프라인 실행
    main_service = MainService(spacy_model_name='en_core_web_sm')
    processed_documents = main_service.process_documents(raw_docs)
    print("Document processing complete.")
    print(processed_documents.get_stats())

    print(Fore.CYAN + "--- 3. Building Word Graph ---")
    # GraphService를 사용하여 단어 그래프 생성
    graph_service = GraphService()
    try:
        word_graph = graph_service.build_graph_from_documents(
            documents=processed_documents,
            top_n=args.top_nodes,
            exclude_stopwords=True
        )
        print("Word graph built successfully.")
        print(graph_service.get_graph_statistics(word_graph))
    except ValueError as e:
        print(Fore.RED + f"Error building graph: {e}")
        return

    print(Fore.CYAN + "--- 4. Visualizing Graph ---")
    # 생성된 그래프 시각화 및 저장
    graph_service.visualize_graph(
        word_graph,
        top_k=args.top_nodes, # 시각화할 노드 수도 top_nodes와 맞춤
        save_path=args.output
    )
    print(f"Graph visualization saved to '{args.output}'.")

    print(Fore.GREEN + Style.BRIGHT + "\nPipeline finished successfully!")

if __name__ == "__main__":
    main()