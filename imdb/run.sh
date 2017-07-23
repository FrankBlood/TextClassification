# bidirectional_lstm
# nohup python train.py --max_features 20000 --maxlen 100 --embedding_dims 128 --dropout 0.5 --model_name bidirectional_lstm --batch_size 128  --lstm_output_size 64 --nb_epoch 100 > ./logs/bidirectional_lstm &
nohup python train.py --max_features 20000 --maxlen 100 --embedding_dims 300 --dropout 0.5 --model_name bidirectional_lstm --batch_size 128  --lstm_output_size 64 --nb_epoch 100 --embedding_file /home/irlab0/Research/kaggle/Quora_Question_Pairs/data/glove.840B.300d.txt --train_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/train/ --test_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/test/ > ./logs/bidirectional_lstm1 &

# cnn_lstm
# nohup python train.py --max_features 20000 --maxlen 100 --embedding_dims 128 --dropout 0.25 --model_name cnn_lstm --batch_size 128 --nb_filter 64 --filter_length 5 --nb_epoch 100 --pool_length 4 --lstm_output_size 70 > ./logs/cnn_lstm &
# nohup python train.py --max_features 20000 --maxlen 100 --embedding_dims 300 --dropout 0.25 --model_name cnn_lstm --batch_size 128 --nb_filter 64 --filter_length 5 --nb_epoch 100 --pool_length 4 --lstm_output_size 70 --embedding_file /home/irlab0/Research/kaggle/Quora_Question_Pairs/data/glove.840B.300d.txt --train_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/train/ --test_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/test/ > ./logs/cnn_lstm1 &
202.114.
# cnn
# nohup python train.py --max_features 5000 --maxlen 400 --embedding_dims 50 --dropout 0.2 --model_name cnn --batch_size 128 --nb_filter 250 --filter_length 3 --nb_epoch 100 --hidden_dims 250 > ./logs/cnn &
# nohup python train.py --max_features 5000 --maxlen 400 --embedding_dims 300 --dropout 0.2 --model_name cnn --batch_size 128 --nb_filter 250 --filter_length 3 --nb_epoch 100 --hidden_dims 250 --embedding_file /home/irlab0/Research/kaggle/Quora_Question_Pairs/data/glove.840B.300d.txt --train_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/train/ --test_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/test/ > ./logs/cnn1 &

# lstm
# nohup python train.py --max_features 20000 --maxlen 80 --embedding_dims 128 --dropout 0.2 --model_name lstm --batch_size 128 --nb_epoch 100 --lstm_output_size 128 > ./logs/lstm &
# nohup python train.py --max_features 20000 --maxlen 80 --embedding_dims 300 --dropout 0.2 --model_name lstm --batch_size 128 --nb_epoch 100 --lstm_output_size 128 --embedding_file /home/irlab0/Research/kaggle/Quora_Question_Pairs/data/glove.840B.300d.txt --train_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/train/ --test_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/test/ > ./logs/lstm1 &

# cnn_based_rnn
# nohup python train.py --max_features 20000 --maxlen 1000 --embedding_dims 300 --dropout 0.5 --model_name cnn_based_rnn --batch_size 50 --nb_filter 128 --filter_length 3 --nb_epoch 100 --hidden_dims 300 --lstm_output_size 300 > ./logs/cnn_based_rnn &
# nohup python train.py --max_features 200000 --maxlen 400 --embedding_dims 300 --dropout 0.5 --model_name cnn_based_rnn --batch_size 50 --nb_filter 250 --filter_length 3 --nb_epoch 100 --hidden_dims 300 --lstm_output_size 300 --embedding_file /home/irlab0/Research/kaggle/Quora_Question_Pairs/data/glove.840B.300d.txt --train_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/train/ --test_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/test/ > ./logs/cnn_based_rnn1 &
# nohup python train.py --max_features 200000 --maxlen 400 --embedding_dims 300 --dropout 0.5 --model_name cnn_based_rnn --batch_size 50 --nb_filter 250 --filter_length 3 --nb_epoch 100 --hidden_dims 300 --lstm_output_size 600 --embedding_file /home/irlab0/Research/kaggle/Quora_Question_Pairs/data/glove.840B.300d.txt --train_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/train/ --test_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/test/ > ./logs/cnn_based_rnn2 &

# python train.py --max_features 200000 --maxlen 1000 --embedding_dims 300 --dropout 0.5 --model_name cnn_based_rnn --batch_size 50 --nb_filter 128 --filter_length 3 --nb_epoch 100 --hidden_dims 300 --lstm_output_size 300 --embedding_file /home/irlab0/Research/kaggle/Quora_Question_Pairs/data/glove.840B.300d.txt --train_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/train/ --test_path /home/irlab0/Research/TextClassification/imdb/data/aclImdb/test/

# nohup python train.py --max_features 20000 --maxlen 80 --embedding_dims 128 --dropout 0.2 --model_name model --batch_size 128 --nb_filter 64 --filter_length 5 --nb_epoch 10 --pool_length 4 --lstm_output_size 70 > ./logs/lstm &
