from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 엔티티 및 사이썬 모듈 임포트
from ..entities import Documents, WordGraph, Word
from ..entities.co_occurence import build_cooccurrence_edges

class GraphService:
    """
    Documents 엔티티로부터 WordGraph를 구축, 분석, 시각화하는 서비스.
    이 서비스는 상태를 가지지 않습니다 (stateless).
    """

    def build_graph_from_documents(
        self, 
        documents: Documents, 
        top_n: int = 500, 
        exclude_stopwords: bool = True
    ) -> WordGraph:
        """
        처리된 Documents 객체로부터 완전한 WordGraph를 생성합니다.

        Args:
            documents: 문장과 단어 정보가 채워진 Documents 객체.
            top_n: 그래프 노드로 사용할 상위 빈도 단어의 수.
            exclude_stopwords: 불용어 제외 여부.

        Returns:
            노드와 공출현 엣지로 구성된 WordGraph 객체.
        """
        if not documents.sentence_list:
            raise ValueError("Documents object has not been processed. No sentences found.")

        # 1. 상위 N개 단어 추출
        all_words = documents.words_list
        if exclude_stopwords:
            all_words = [word for word in all_words if not word.get_stopword_status()]
        
        # 빈도순으로 정렬
        all_words.sort(key=lambda w: w.freq, reverse=True)
        top_words = all_words[:top_n]

        if not top_words:
            raise ValueError("No words found to build the graph.")

        # 2. WordGraph 객체 생성 (노드 포함)
        word_graph = WordGraph(top_words)
        print(f"Created WordGraph with {word_graph.num_nodes} nodes.")

        # 3. 공출현 엣지 계산 및 설정
        try:
            edges, weights = build_cooccurrence_edges(
                word_graph.word_to_node_id, 
                documents.sentence_list
            )
            
            if edges:
                word_graph.set_edges_from_co_occurrence(edges, weights)
                print(f"Set {len(edges)} co-occurrence edges.")
            else:
                print("Warning: No co-occurrence edges found.")
                word_graph.set_edges_from_co_occurrence([], [])

        except Exception as e:
            print(f"Error building co-occurrence edges: {e}")
            word_graph.set_edges_from_co_occurrence([], [])

        return word_graph

    def visualize_graph(
        self, 
        word_graph: WordGraph, 
        top_k: int = 50, 
        figsize: Tuple[int, int] = (12, 8), 
        min_weight: float = None, 
        show_labels: bool = True, 
        save_path: str = None, 
        dpi: int = 300
    ) -> None:
        """
        WordGraph 객체를 시각화합니다.
        """
        if word_graph.num_edges == 0 or word_graph.num_nodes == 0:
            print("Warning: Cannot visualize an empty or edgeless graph.")
            return

        G = word_graph.export_to_networkx(include_weights=True)
        nodes_to_display = list(range(min(top_k, word_graph.num_nodes)))
        G_sub = G.subgraph(nodes_to_display).copy()
        labels = {i: word_graph.words[i].content for i in G_sub.nodes()}

        if min_weight is not None:
            edges_to_remove = [
                (u, v) for u, v, d in G_sub.edges(data=True) 
                if d.get('weight', 0) < min_weight
            ]
            G_sub.remove_edges_from(edges_to_remove)
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G_sub, k=0.8, iterations=50, seed=42)
        
        node_sizes = [max(20, word_graph.words[i].freq * 5) for i in G_sub.nodes()]
        nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
        
        if G_sub.number_of_edges() > 0:
            edge_weights = np.array([d['weight'] for u, v, d in G_sub.edges(data=True)])
            edge_widths = (edge_weights / edge_weights.max() * 4.0) if edge_weights.max() > 0 else 1.0
            nx.draw_networkx_edges(G_sub, pos, width=edge_widths, alpha=0.6, edge_color='gray')

        if show_labels:
            nx.draw_networkx_labels(G_sub, pos, labels, font_size=9, font_family='sans-serif')
        
        plt.title(f"Word Co-occurrence Graph (Top {len(G_sub.nodes())} Words)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Graph saved to {save_path}")
            
        plt.show()

    def get_graph_statistics(self, word_graph: WordGraph) -> Dict[str, any]:
        """WordGraph의 통계 정보를 반환합니다."""
        return word_graph.get_graph_stats()