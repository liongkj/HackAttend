# mkdir glue
# DREAM
cd glue
mkdir dream
cd dream
wget https://raw.githubusercontent.com/nlpdata/dream/master/data/train.json
wget https://raw.githubusercontent.com/nlpdata/dream/master/data/test.json
wget https://raw.githubusercontent.com/nlpdata/dream/master/data/dev.json

cd ..

mkdir hellaswag
cd hellaswag
wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl
wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl
wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl
mv hellaswag_test.jsonl hellaswag_test_temp.jsonl
mv hellaswag_val hellaswag_test.jsonl

cd ..

mkdir alphanli
cd alphanli
wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip
unzip anli.zip
rm anli.zip
mv anli/* .
rm -rf anli

cd ..

mkdir sst-2
cd sst-2
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
unzip SST-2.zip
rm SST-2.zip
mv SST-2/* .
rm -rf SST-2
rm -rf original
mv test.tsv test_temp.tsv
mv dev.tsv test.tsv

cd ..

mkdir mnli
cd mnli
wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
unzip MNLI.zip
rm MNLI.zip
mv MNLI/* .
rm -rf MNLI
rm -rf original
mv test_matched.tsv test_matched_temp.tsv
mv test_mismatched.tsv test_mismatched.tsv
mv dev_matched.tsv test_matched.tsv
mv dev_mismatched.tsv test_mismatched.tsv

cd .. 

# mkdir qnli
# cd qnli
# wget https://dl.fbaipublicfiles.com/glue/data/QNLI.zip
# unzip QNLI.zip
# rm QNLI.zip
# mv QNLI/* .
# rm -rf QNLI
# rm -rf original
# mv test.tsv test_temp.tsv
# mv dev.tsv test.tsv
# cd ..

# mkdir qqp
# cd qqp
# wget https://dl.fbaipublicfiles.com/glue/data/QQP.zip
# unzip QQP.zip
# rm QQP.zip
# mv QQP/* .
# rm -rf QQP
# mv test.tsv test_temp.tsv
# mv dev.tsv test.tsv
# cd ..

mkdir reclor
cd reclor
wget https://github.com/yuweihao/reclor/releases/download/v1/reclor_data.zip
# unzip with password for_non-commercial_research_purpose_only
unzip -P for_non-commercial_research_purpose_only reclor_data.zip
rm reclor_data.zip
rm *.txt 
rm question_type_names.json
mv test.json test_temp.json
mv val.json test.json
cd ..

mkdir paws-qqp
cd paws-qqp
wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz
tar -xvzf paws_wiki_labeled_final.tar.gz
rm paws_wiki_labeled_final.tar.gz
mv final/* .
rm -rf final
cd ..

# mkdir hans
# cd hans
# wget https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_train_set.jsonl
# wget https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.jsonl





# https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py