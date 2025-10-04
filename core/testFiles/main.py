import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'main-service'))

import time
from typing import List, Optional

import pandas as pd
import numpy as np

# 새로운 아키텍처 import
from services import *
from entities import *

class TextProcessingBenchmark:
    """텍스트 처리 및 그래프 생성 벤치마크 클래스"""
    
    def __init__(self):
        self.doc_service: Optional[DocumentService] = None
        self.graph_service: Optional[GraphService] = None
        self.word_graph: Optional[WordGraph] = None
        
        # 성능 측정 결과
        self.timings = {}
        self.stats = {}
    
    def load_data(self, csv_path: str, text_column: str = 'body', limit: int = 5000) -> List[str]:
        """CSV 데이터 로드"""
        print(f"📄 Loading data from {csv_path} (limit: {limit})")
        
        start_time = time.time()
        df = pd.read_csv(csv_path)
        text_data = df[text_column][:limit].tolist()
        del df  # 메모리 정리
        
        load_time = time.time() - start_time
        self.timings['data_loading'] = load_time
        
        print(f"✅ Loaded {len(text_data)} texts in {load_time:.3f} seconds")
        return text_data
    
    def preprocess_documents(self, texts: List[str], 
                           model_name: str = 'en_core_web_sm',
                           use_parallel: bool = True,
                           batch_size: int = 100,
                           n_process: int = None) -> DocumentService:
        """문서 전처리 실행"""
        print(f"🔄 Starting document preprocessing...")
        print(f"   - Model: {model_name}")
        print(f"   - Parallel: {use_parallel}")
        print(f"   - Batch size: {batch_size}")
        
        start_time = time.time()
        
        # DocumentService 생성
        self.doc_service = DocumentService(model_name=model_name)
        
        # 텍스트 설정 및 처리
        if use_parallel:
            # rawdata 설정은 자동으로 create_sentence_list() 호출
            # 하지만 병렬처리를 원하면 수동으로 호출
            self.doc_service._documents.rawdata = texts  # 자동 처리 우회
            self.doc_service.create_sentence_list_parallel(batch_size=batch_size, n_process=n_process)
        else:
            self.doc_service.rawdata = texts  # 자동으로 순차 처리
        
        preprocess_time = time.time() - start_time
        self.timings['preprocessing'] = preprocess_time
        
        # === 디버깅: 중간 결과 확인 ===
        print(f"\n🔍 DEBUGGING PREPROCESSING RESULTS:")
        print(f"   📊 Document count: {len(self.doc_service.rawdata) if self.doc_service.rawdata else 0}")
        print(f"   📊 Sentence count: {len(self.doc_service.sentence_list) if self.doc_service.sentence_list else 0}")
        
        # 첫 번째 문장 확인
        if self.doc_service.sentence_list and len(self.doc_service.sentence_list) > 0:
            first_sentence = self.doc_service.sentence_list[0]
            print(f"   📊 First sentence processed: {first_sentence.is_processed}")
            print(f"   📊 First sentence words: {len(first_sentence.lemmatised)}")
            print(f"   📊 First sentence preview: '{first_sentence.get_text_preview(50)}'")
            if first_sentence.lemmatised:
                print(f"   📊 First few lemmas: {first_sentence.lemmatised[:5]}")
            if first_sentence.processing_errors:
                print(f"   ⚠️  First sentence errors: {first_sentence.processing_errors}")
        
        # 단어 통계 확인
        words_list = self.doc_service.words_list
        print(f"   📊 Total unique words: {len(words_list) if words_list else 0}")
        
        if words_list and len(words_list) > 0:
            print(f"   📊 First few words: {[w.content for w in words_list[:5]]}")
            top_words = self.doc_service.get_top_words(top_n=10)
            print(f"   📊 Top 10 words: {[(w.content, w.freq) for w in top_words]}")
        else:
            print(f"   ❌ No words found in words_list!")
            
            # WordTrie 직접 확인
            word_trie = self.doc_service._documents.word_trie
            print(f"   📊 WordTrie word_count: {word_trie.word_count}")
            
            all_words = word_trie.get_all_words()
            print(f"   📊 WordTrie all_words: {len(all_words) if all_words else 0}")
            
            if all_words:
                print(f"   📊 WordTrie first words: {[w.content for w in all_words[:5]]}")
        
        # 통계 수집
        self.stats['document_stats'] = self.doc_service.get_stats()
        
        print(f"✅ Document preprocessing completed in {preprocess_time:.3f} seconds")
        print(f"   📊 {self.doc_service}")
        
        return self.doc_service
    
    def create_graph(self, top_n: int = 500, 
                    exclude_stopwords: bool = True,
                    use_word_graph: bool = True) -> GraphService:
        """그래프 생성 실행"""
        if self.doc_service is None:
            raise ValueError("Documents not preprocessed. Call preprocess_documents() first.")
        
        print(f"📊 Starting graph creation...")
        print(f"   - Top words: {top_n}")
        print(f"   - Exclude stopwords: {exclude_stopwords}")
        print(f"   - Use WordGraph: {use_word_graph}")
        
        start_time = time.time()
        
        # GraphService 생성
        self.graph_service = GraphService(self.doc_service)
        
        if use_word_graph:
            # 새로운 방식: WordGraph 객체 생성
            self.word_graph = self.graph_service.build_complete_graph(
                top_n=top_n, 
                exclude_stopwords=exclude_stopwords
            )
        else:
            # 기존 방식 (호환성)
            self.graph_service.create_graph(top_n=top_n, exclude_stopwords=exclude_stopwords)
            self.graph_service.create_co_occurrence_edges()
        
        graph_time = time.time() - start_time
        self.timings['graph_creation'] = graph_time
        
        # 그래프 통계 수집
        if use_word_graph and self.word_graph:
            self.stats['graph_stats'] = self.word_graph.get_graph_stats()
        else:
            self.stats['graph_stats'] = self.graph_service.get_graph_statistics()
        
        print(f"✅ Graph creation completed in {graph_time:.3f} seconds")
        print(f"   📊 {self.graph_service}")
        
        return self.graph_service
    
    def visualize_graph(self, top_k: int = 50, 
                       save_path: str = './graph_visualization.png',
                       min_weight: float = None,
                       show_labels: bool = True,
                       figsize: tuple = (12, 8)) -> None:
        """그래프 시각화"""
        if self.graph_service is None:
            raise ValueError("Graph not created. Call create_graph() first.")
        
        print(f"🎨 Creating graph visualization...")
        print(f"   - Top nodes: {top_k}")
        print(f"   - Save path: {save_path}")
        
        start_time = time.time()
        
        if self.word_graph:
            # 새로운 방식: WordGraph 시각화
            self.graph_service.visualize_word_graph(
                self.word_graph,
                top_k=top_k,
                save_path=save_path,
                min_weight=min_weight,
                show_labels=show_labels,
                figsize=figsize
            )
        else:
            # 기존 방식 (호환성)
            self.graph_service.visualize(
                top_k=top_k,
                save_path=save_path,
                min_weight=min_weight,
                show_labels=show_labels,
                figsize=figsize
            )
        
        viz_time = time.time() - start_time
        self.timings['visualization'] = viz_time
        
        print(f"✅ Visualization completed in {viz_time:.3f} seconds")
    
    def print_performance_report(self) -> None:
        """성능 리포트 출력"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE REPORT")
        print("="*60)
        
        # 시간 측정 결과
        print("⏱️  TIMING RESULTS:")
        total_time = 0
        for stage, duration in self.timings.items():
            print(f"   🕐 {stage.replace('_', ' ').title()}: {duration:.3f} sec")
            total_time += duration
        print(f"   🕐 Total Processing Time: {total_time:.3f} sec")
        
        # 문서 통계
        if 'document_stats' in self.stats:
            print(f"\n📄 DOCUMENT STATISTICS:")
            doc_stats = self.stats['document_stats']
            print(f"   📊 Documents: {doc_stats.get('document_count', 0)}")
            print(f"   📊 Sentences: {doc_stats.get('sentence_count', 0)}")
            print(f"   📊 Unique Words: {doc_stats.get('unique_word_count', 0)}")
            print(f"   📊 Content Words: {doc_stats.get('content_words', 0)}")
        
        # 그래프 통계
        if 'graph_stats' in self.stats:
            print(f"\n🕸️  GRAPH STATISTICS:")
            graph_stats = self.stats['graph_stats']
            print(f"   📊 Nodes: {graph_stats.get('num_nodes', 0)}")
            print(f"   📊 Edges: {graph_stats.get('num_edges', 0)}")
            print(f"   📊 Density: {graph_stats.get('density', 0):.4f}")
            print(f"   📊 Avg Degree: {graph_stats.get('avg_degree', 0):.2f}")
        
        print("="*60 + "\n")
    
    def run_complete_benchmark(self, csv_path: str, 
                                limit: int = 10000,
                                top_n: int = 500,
                                top_k_viz: int = 50,
                                save_path: str = './graph_output.png') -> None:
        """전체 벤치마크 실행"""
        try:
            # 1. 데이터 로드
            texts = self.load_data(csv_path, limit=limit)
            
            # 2. 문서 전처리
            self.preprocess_documents(texts, use_parallel=True)
            
            # 3. 그래프 생성
            self.create_graph(top_n=top_n, use_word_graph=True)
            
            # # 4. 그래프 시각화
            # self.visualize_graph(top_k=top_k_viz, save_path=save_path)
            
            # # 5. 성능 리포트
            # self.print_performance_report()
            
            try:
                # Word2Vec 서비스 생성
                print("Creating Word2Vec service...")
                word2vec_service = Word2VecService.create_default(self.doc_service)
                
                # 어휘 정보 출력
                vocab_info = word2vec_service.get_vocabulary_info()
                print(f"Vocabulary size: {vocab_info['vocab_size']}")
                print(f"Total sentences: {vocab_info['total_sentences']}")
                print(f"Total tokens: {vocab_info['total_tokens']}")
                print(f"Most frequent words: {[word for word, _ in vocab_info['most_frequent_words'][:10]]}")
                
                # 모델 훈련
                print("\nStarting Word2Vec training...")
                word2vec_service.train("word2vec_embeddings.txt")
                
                # 7. 단어 벡터 테스트
                print("\n" + "-"*40)
                print("Word Vector Tests")
                print("-"*40)
                
                # 빈도 높은 단어들로 테스트
                test_words = [word for word, _ in vocab_info['most_frequent_words'][:5]]
                
                for word in test_words:
                    vector = word2vec_service.get_word_vector(word)
                    if vector is not None:
                        print(f"'{word}' vector shape: {vector.shape}, norm: {np.linalg.norm(vector):.4f}")
                        print(f"  First 5 dimensions: {vector[:5]}")
                
                # 8. 유사도 테스트
                print("\n" + "-"*40)
                print("Word Similarity Tests")
                print("-"*40)
                
                # 상위 빈도 단어들의 유사 단어 찾기
                for word in test_words[:3]:
                    similar_words = word2vec_service.find_similar_words(word, top_k=5)
                    if similar_words:
                        print(f"\nWords similar to '{word}':")
                        for similar_word, similarity in similar_words:
                            print(f"  {similar_word}: {similarity:.4f}")
                
                # 9. 단어 유추 테스트 (가능한 경우)
                print("\n" + "-"*40)
                print("Word Analogy Tests")
                print("-"*40)
                
                # 어휘에서 적절한 단어 쌍을 찾아서 테스트
                vocab_words = list(word2vec_service.data_loader.word2id.keys())
                
                # 일반적인 유추 테스트 단어들 (있는 경우에만)
                analogy_tests = [
                    ("man", "woman", "king"),  # man:woman = king:?
                    ("good", "better", "bad"), # good:better = bad:?
                    ("big", "bigger", "small") # big:bigger = small:?
                ]
                
                for word_a, word_b, word_c in analogy_tests:
                    if all(word in vocab_words for word in [word_a, word_b, word_c]):
                        result = word2vec_service.word_analogy(word_a, word_b, word_c, top_k=3)
                        if result:
                            print(f"\n{word_a} is to {word_b} as {word_c} is to:")
                            for word, score in result:
                                print(f"  {word}: {score:.4f}")
                    else:
                        missing = [w for w in [word_a, word_b, word_c] if w not in vocab_words]
                        print(f"Skipping analogy test - missing words: {missing}")
                
                # 10. 벡터 연산 테스트
                print("\n" + "-"*40)
                print("Vector Operations Tests")
                print("-"*40)
                
                if len(test_words) >= 3:
                    word1, word2, word3 = test_words[:3]
                    
                    vec1 = word2vec_service.get_word_vector(word1)
                    vec2 = word2vec_service.get_word_vector(word2)
                    vec3 = word2vec_service.get_word_vector(word3)
                    
                    if all(v is not None for v in [vec1, vec2, vec3]):
                        # 벡터 합
                        vec_sum = vec1 + vec2
                        print(f"Vector addition: {word1} + {word2}")
                        print(f"  Result norm: {np.linalg.norm(vec_sum):.4f}")
                        
                        # 코사인 유사도 계산
                        def cosine_similarity(v1, v2):
                            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        
                        sim_12 = cosine_similarity(vec1, vec2)
                        sim_13 = cosine_similarity(vec1, vec3)
                        sim_23 = cosine_similarity(vec2, vec3)
                        
                        print(f"\nCosine similarities:")
                        print(f"  {word1} <-> {word2}: {sim_12:.4f}")
                        print(f"  {word1} <-> {word3}: {sim_13:.4f}")
                        print(f"  {word2} <-> {word3}: {sim_23:.4f}")
                
                # 11. 모델 저장
                print("\n" + "-"*40)
                print("Model Saving")
                print("-"*40)
                
                model_path = "word2vec_model.pth"
                word2vec_service.save_model(model_path)
                print(f"Model saved to {model_path}")
                
                # 12. 전체 평가
                print("\n" + "-"*40)
                print("Overall Evaluation")
                print("-"*40)
                
                word2vec_service.evaluate_similarity(test_words[:5], top_k=3)
                
                # 13. 서비스 정보 요약
                print("\n" + "-"*40)
                print("Word2Vec Service Summary")
                print("-"*40)
                print(word2vec_service)
            
            except Exception as e:
                print(f"Word2Vec training/testing failed: {e}")
                import traceback
                traceback.print_exc()            
            
            
        except Exception as e:
            print(f"❌ Error during benchmark: {e}")
            raise


def main():
    """메인 함수 - 리팩토링된 버전"""
    print("Starting Text Processing & Graph Creation Benchmark")
    print("Using New Architecture: DocumentService + GraphService + WordGraph")
    print("-" * 60)
    
    # 벤치마크 실행
    benchmark = TextProcessingBenchmark()
    
    try:
        benchmark.run_complete_benchmark(
            csv_path="../kaggle_RC_2019-05.csv",
            limit=10000,
            top_n=500,
            top_k_viz=50,
            save_path='./graphOutput.png'
        )
        
        print("🎉 Benchmark completed successfully!")
        
    except FileNotFoundError:
        print("❌ CSV file not found. Please check the path.")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def comparison_test():
    """기존 방식 vs 새로운 방식 비교 테스트"""
    print("🔄 Running Comparison Test: Old vs New Architecture")
    print("-" * 60)
    
    # 작은 데이터셋으로 테스트
    try:
        df = pd.read_csv("../kaggle_RC_2019-05.csv")
        texts = df['body'][:1000].tolist()  # 1000개만 테스트
        del df
        
        # 새로운 방식 테스트
        print("🆕 Testing New Architecture...")
        new_benchmark = TextProcessingBenchmark()
        
        start_time = time.time()
        new_benchmark.preprocess_documents(texts, use_parallel=True)
        new_benchmark.create_graph(top_n=100, use_word_graph=True)
        new_total_time = time.time() - start_time
        
        print(f"✅ New Architecture Total Time: {new_total_time:.3f} seconds")
        
        # 호환성 테스트 (기존 인터페이스)
        print("🔄 Testing Compatibility Mode...")
        compat_benchmark = TextProcessingBenchmark()
        
        start_time = time.time()
        compat_benchmark.preprocess_documents(texts, use_parallel=False)
        compat_benchmark.create_graph(top_n=100, use_word_graph=False)  # 기존 방식
        compat_total_time = time.time() - start_time
        
        print(f"✅ Compatibility Mode Total Time: {compat_total_time:.3f} seconds")
        
        # 결과 비교
        print(f"\n📊 Performance Comparison:")
        print(f"   🆕 New Architecture: {new_total_time:.3f}s")
        print(f"   🔄 Compatibility Mode: {compat_total_time:.3f}s")
        print(f"   📈 Speed Difference: {((compat_total_time - new_total_time) / compat_total_time * 100):+.1f}%")
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")


if __name__ == "__main__":
    # 기본 벤치마크 실행
    main()
    
    # 선택적으로 비교 테스트 실행
    # comparison_test()