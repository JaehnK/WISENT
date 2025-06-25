cd ./main-service/Preprocess
python setup_trie.py build_ext --inplace
cp build/lib.linux-x86_64-cpython-312/Preprocess/trie.cpython-312-x86_64-linux-gnu.so .
python setup_cooccur.py build_ext --inplace
cp build/lib.linux-x86_64-cpython-312/Preprocess/co_occurence.cpython-312-x86_64-linux-gnu.so .
cd ../../