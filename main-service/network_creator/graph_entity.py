from typing import Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from Preprocess import Docs, Word

class graph_entity:
    def __init__(self, docs: Docs):
        self.docs_ref: Docs = docs
        
        self.nodes: Optional[torch.Tensor] = None
        self.edges: Optional[torch.Tensor] = None
        self.edges_weight: Optional[torch.Tensor] = None
        
        self.nodes_words: Optional[List[Word]] = None
        self.word_to_node: Optional[Dict[str, int]] = None
        
        self._create_graph()
        
    def _create_graph(self, nodes: int = 500) -> None:
        top_words = self.docs_ref._word_trie.get_top_words_by_pos(top_n = nodes, exclude_stopwords=True)
        
        self.node_words = top_words
        self.word_to_node = {word.content: idx for idx, word in enumerate(top_words)}
        
    def create_co_occurrence_edges(self):
        if self.word_to_node is None:
            raise ValueError("Word mapping not initialized.")
        
        edges, weights = self.docs_ref.get_co_occurrence_edges(self.word_to_node)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
        
        self.edges, self.edges_weight = to_undirected(edge_index, edge_weight)
        
    def visualize(self, top_k: int = 50, figsize: tuple = (12, 8), 
                    min_weight: float = None, show_labels: bool = True, 
                    save_path: str = None, dpi: int = 300):
        """
        그래프를 시각화합니다.
        
        Args:
            top_k: 상위 k개 노드만 표시
            figsize: 그림 크기
            min_weight: 최소 엣지 가중치 (필터링용)
            show_labels: 노드 라벨 표시 여부
        """
        
        if self.edges is None or self.node_words is None:
            raise ValueError("그래프가 생성되지 않았습니다. _create_graph()와 create_co_occurrence_edges()를 먼저 실행하세요.")
        
        # NetworkX 그래프 생성
        G = nx.Graph()
        
        # 상위 k개 노드만 선택
        selected_nodes = list(range(min(top_k, len(self.node_words))))
        node_labels = {i: self.node_words[i].content for i in selected_nodes}
        
        # 노드 추가
        G.add_nodes_from(selected_nodes)
        
        # 엣지 추가 (선택된 노드들 사이의 엣지만)
        edges_np = self.edges.numpy()
        weights_np = self.edges_weight.numpy()
        
        for i, (src, dst) in enumerate(edges_np.T):
            if src < top_k and dst < top_k:  # 선택된 노드들 사이의 엣지만
                weight = weights_np[i]
                if min_weight is None or weight >= min_weight:
                    G.add_edge(src, dst, weight=weight)
        
        # 시각화
        plt.figure(figsize=figsize)
        
        # 레이아웃 계산
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 노드 그리기 (빈도수에 따라 크기 조정)
        node_sizes = [self.node_words[i].freq * 50 for i in selected_nodes if i < len(self.node_words)]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                            node_color='lightblue', alpha=0.7)
        
        # 엣지 그리기 (가중치에 따라 두께 조정)
        if G.edges():
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_widths = np.array(edge_weights) / max(edge_weights) * 3
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        
        # 라벨 그리기
        if show_labels:
            nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
        
        plt.title(f"co_occurence graph (top {top_k} words)")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"그래프가 저장되었습니다: {save_path}")
        
        plt.show()
        plt.show()
        
        print(f"노드 수: {G.number_of_nodes()}")
        print(f"엣지 수: {G.number_of_edges()}")