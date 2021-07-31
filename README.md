# knowledge-acquisition-paper
Experiments for paper on Knowledge Acquisition

python example/train_supervised_bert.py --batch_size 32 --lr 2e-5 --max_epoch 2 --max_length 250 --seed 42 --train_file data/distant_supervision_matches_open_nre_train.txt --val_file data/distant_supervision_matches_open_nre_dev.txt --test_file data\distant_supervision_matches_open_nre_final_test.txt --rel2id_file data/distant_supervision_matches_open_nre_rel_2_id.jsonl --ckpt supervised_bert_v1  