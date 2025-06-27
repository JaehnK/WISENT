import re
import time

from preprocess import *

def comparsion(docs):
    print("\n" + "="*60)
    print("CONTRACTION PROCESSING ANALYSIS")
    print("="*60)
    
    # 축약형이 어떻게 처리되었는지 확인
    print("Raw vs Processed comparison:")
    for i, sentence in enumerate(docs.sentence_list[:5]):
        if sentence and sentence.word_indices is not None:
            print(f"\nSentence {i+1}:")
            print(f"  📝 Raw: {sentence.raw}")
            
            # 전처리 과정 단계별 확인
            expanded = sentence._expand_contractions(sentence.raw)
            cleaned_step1 = re.sub(r"'s\b", " s", expanded)
            cleaned_final = re.sub(r'[^\w\s]', '', cleaned_step1)
            
            print(f"  🔄 Expanded: '{expanded}'")
            print(f"  🧹 Cleaned: '{cleaned_final}'")
            print(f"  🔤 Lemmatised: {sentence._lemmatised}")
            print(f"  📊 Word indices: {sentence.word_indices}")
            
            # 문제가 될 수 있는 단어들 체크
            problematic = []
            for word in sentence._lemmatised:
                if len(word) <= 1 or word in ['t', 's', 'll', 've', 're', 'm', 'd']:
                    problematic.append(word)
            
            if problematic:
                print(f"  ⚠️  Potentially problematic tokens: {problematic}")
            else:
                print(f"  ✅ No problematic tokens detected")
        else:
            print(f"\nSentence {i+1}: ❌ Processing failed")
    
def part_of_speech_analysis(docs):
    print("\n" + "="*60)
    print("PART-OF-SPEECH ANALYSIS")
    print("="*60)
    
    # 품사별 단어 분류
    pos_categories = {}
    for word in docs.words_list:
        category = word.get_pos_category()
        if category not in pos_categories:
            pos_categories[category] = []
        pos_categories[category].append(word)
    
    print("Words by POS category:")
    for category in sorted(pos_categories.keys()):
        words_in_category = pos_categories[category]
        print(f"\n📂 {category}: {len(words_in_category)} words")
        
        # 빈도순으로 정렬해서 상위 5개만 표시
        top_words = sorted(words_in_category, key=lambda w: w.freq, reverse=True)[:5]
        for word in top_words:
            pos_info = f"{word.dominant_pos}"
            if len(word.pos_counts) > 1:
                pos_info += f" (다중: {word.pos_counts})"
            print(f"    '{word.content}' - freq: {word.freq}, pos: {pos_info}")
        
        if len(words_in_category) > 5:
            print(f"    ... and {len(words_in_category) - 5} more {category.lower()}s")
    
    print(f"\n🔍 Multi-POS words (동일 단어의 다른 품사 용법):")
    multi_pos_words = [w for w in docs.words_list if len(w.pos_counts) > 1]
    if multi_pos_words:
        for word in sorted(multi_pos_words, key=lambda w: len(w.pos_counts), reverse=True)[:10]:
            print(f"  '{word.content}': {word.pos_counts} (주품사: {word.dominant_pos})")
    else:
        print("  No words with multiple POS tags found")
    
    print(f"\n📊 POS Category Distribution:")
    total_words = len(docs.words_list)
    for category in sorted(pos_categories.keys()):
        count = len(pos_categories[category])
        percentage = (count / total_words) * 100
        print(f"  {category}: {count} words ({percentage:.1f}%)")
    
def detailed_word_analysis(docs):
    print("\n" + "="*60)
    print("DETAILED WORD ANALYSIS")
    print("="*60)
    
    print("Sample detailed word information:")
    sample_words = sorted(docs.words_list, key=lambda w: w.freq, reverse=True)[:30]
    for word in sample_words:
        print(f"\n🔤 Word: '{word.content}'")
        print(f"   📊 Frequency: {word.freq}")
        print(f"   🔢 Index: {word.idx}")
        print(f"   📝 Category: {word.get_pos_category()}")
        print(f"   🏷️  Dominant POS: {word.dominant_pos}")
        print(f"   🏷️  StopWord: {word.is_stopword}")
        print(f"   📋 All POS tags: {word.pos_counts}")
        
        # 편의 메서드 테스트
        pos_checks = []
        if word.is_noun(): pos_checks.append("NOUN")
        if word.is_verb(): pos_checks.append("VERB") 
        if word.is_adjective(): pos_checks.append("ADJECTIVE")
        if word.is_pronoun(): pos_checks.append("PRONOUN")
        if word.is_adverb(): pos_checks.append("ADVERB")
        
        if pos_checks:
            print(f"   ✅ Type checks: {', '.join(pos_checks)}")
        else:
            print(f"   ℹ️  Type: OTHER ({word.get_pos_category()})")

def contraction_test(docs):
    print("\n" + "="*60)
    print("CONTRACTION DETECTION")
    print("="*60)
    
    # 원본 문장에서 축약형 패턴 찾기
    contraction_patterns = [
        "don't", "can't", "won't", "it's", "we're", "wasn't", 
        "I'd", "shouldn't", "I'll", "aren't", "you're", "he'll", "I've"
    ]
    
    found_contractions = []
    for sentence in docs.sentence_list:
        if sentence and sentence.raw:
            for pattern in contraction_patterns:
                if pattern.lower() in sentence.raw.lower():
                    found_contractions.append((pattern, sentence.raw))
    
    print(f"Found contractions in original text:")
    for contraction, sentence in found_contractions[:5]:  # 처음 5개만
        print(f"  '{contraction}' in: {sentence}")
    
    # 전체 토큰 수 vs 예상 토큰 수 비교
    total_original_tokens = sum(len(doc.split()) for doc in test_docs)
    total_processed_tokens = sum(len(s.word_indices) if s.word_indices else 0 for s in docs.sentence_list)
    
    print(f"\n📊 Token count comparison:")
    print(f"  Original word count (space-split): {total_original_tokens}")
    print(f"  Processed token count: {total_processed_tokens}")
    print(f"  Difference: {total_processed_tokens - total_original_tokens}")
    
    if total_processed_tokens > total_original_tokens:
        print("  ℹ️  More tokens after processing (contractions likely split)")
    elif total_processed_tokens < total_original_tokens:
        print("  ⚠️  Fewer tokens after processing (some tokens lost)")
    else:
        print("  ✅ Same token count")
        
    suspicious_words = [w for w in docs.words_list if w.content in ['t', 's', 'll', 've', 're', 'm', 'd', 'nt']]
    if suspicious_words:
        print("🚨 Issues detected with contraction handling:")
        print("   - Contractions are being split into fragments")
        print("   - Consider using a contraction expansion library")
        print("   - Or modify preprocessing to handle apostrophes better")
    else:
        print("✅ No obvious contraction handling issues detected")

if __name__ == "__main__":
    # 사용 예시
    docs = Docs()
    
    # 축약형과 특수 케이스가 포함된 테스트 데이터
    test_docs = [
        "I don't think he'll come today.",
        "She can't find her keys anywhere.",
        "They won't be here until tomorrow.",
        "It's not what I've been looking for.",
        "We're going to John's house tonight.",
        "That wasn't the answer I'd expected.",
        "You shouldn't have done that.",
        "I'll be there in five minutes.",
        "The dog's tail is wagging happily.",
        "These aren't the droids you're looking for."
    ]
    
    print("🧪 TESTING CONTRACTIONS AND APOSTROPHES")
    print("="*60)
    start = time.time()
    docs.rawdata = test_docs
    end = time.time()
    print(f"🕐 {end - start:.5f} sec") 
    
    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSING SUMMARY")
    print("="*60)
    print(docs)
    
    # comparsion(docs)
    # part_of_speech_analysis(docs)
    detailed_word_analysis(docs)
    contraction_test(docs)