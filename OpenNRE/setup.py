'''
python example/train_supervised_cnn.py --metric micro_f1 --batch_size 160 --lr 0.1 --weight_decay 1e-5 --max_epoch 100 --max_length 250 --seed 42 --encoder pcnn --train_file C:\Users\danil\Documents\Northwestern\QRG\Rep\ea\v8\question-answering\ke\preprocessing\distant_supervision_matches_open_nre_train.txt --val_file C:\Users\danil\Documents\Northwestern\QRG\Rep\ea\v8\question-answering\ke\preprocessing\distant_supervision_matches_open_nre_dev.txt --test_file C:\Users\danil\Documents\Northwestern\QRG\Rep\ea\v8\question-answering\ke\preprocessing\distant_supervision_matches_open_nre_test.txt --rel2id_file C:\Users\danil\Documents\Northwestern\QRG\Rep\ea\v8\question-answering\ke\preprocessing\distant_supervision_matches_open_nre_rel_2_id.jsonl --ckpt supervised_cnn_v1
'''

import setuptools
with open("README.md", "r") as fh:
    setuptools.setup(
        name='opennre',  
        version='0.1',
        author="Tianyu Gao",
        author_email="gaotianyu1350@126.com",
        description="An open source toolkit for relation extraction",
        url="https://github.com/thunlp/opennre",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Linux",
        ],
        setup_requires=['wheel']
     )
