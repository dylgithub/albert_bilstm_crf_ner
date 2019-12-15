#encoding=utf-8
class BertConfig(object):
    def __init__(self):
        #数据地址
        self.data_dir='data'
        self.bert_config_file='albert_base_zh/albert_config_base.json'
        self.task_name='ner'
        self.vocab_file='albert_base_zh/vocab.txt'
        self.output_dir='bert_output'
        self.init_checkpoint='albert_base_zh/albert_model.ckpt'
        self.max_seq_length=128
        self.num_labels=8
        self.do_train=True
        self.do_eval=True
        self.do_predict=False
        self.train_batch_size=64
        self.eval_batch_size=500
        self.predict_batch_size=8
        self.learning_rate=1e-5
        self.num_train_epochs=30
        #保存检测点的步数
        self.save_checkpoints_steps=100
        # 训练集打印信息的步数
        self.train_print_steps=200
        # 验证集测试的步数
        self.val_print_steps=500
        self.use_gpu='1'

        self.rnn_hidden_size=300
        #是否只使用CRF不加bi_lstm
        self.only_use_crf=True
        #若使用双向的lstm，则此处为lstm_cell的丢失率
        self.lstm_dropout_rate=0.6