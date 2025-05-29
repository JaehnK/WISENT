from Preprocessing import *

if __name__ == "__main__":
    # 사용 예시
    docs = Docs()
    
    # 테스트 데이터
    test_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox is jumping over a lazy dog.",
        "The lazy dog sleeps under the tree.",
        "Quick foxes are running in the forest.",
        "Dogs and foxes are playing together.",
        "The forest is full of trees and animals.",
        "Animals like dogs and foxes live in forests.",
        "Trees provide shelter for many animals.",
        "The quick animals are running fast.",
        "Lazy animals prefer to sleep all day."
    ]
    
    docs.rawdata = test_docs
    
    print(docs)
    print(f"\nSample word frequencies:")
    for word in docs.words_list[:10]:  # 첫 10개 단어만 출력
        print(f"'{word.content}': freq={word.freq}, idx={word.idx}")
    
    print(f"\nSample sentence word indices:")
    for i, sentence in enumerate(docs.sentence_list[:3]):  # 첫 3개 문장만 출력
        print(f"Sentence {i+1}: {sentence.word_indices}")
        print(f"Words: {[word.content for word in sentence.word_objects]}")