#encoding=utf-8
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
import tensorflow as tf
class BI_LSTM_CRF(object):
    def __init__(self,embedding_input,config,labels,sequence_lengths):
        '''
        :param embedding_input: Fine-tuning之后的特征向量
        :param config: 一些参数配置
        :param only_use_crf: 是否只使用crf层
        :param labels: 真实的标签
        :param sequence_lengths: [batch_size] 每个batch下序列的真实长度
        '''
        self.embedding_input=embedding_input
        self.config=config
        self.labels=labels
        self.sequence_lengths=sequence_lengths
    def _witch_cell(self,cell_type):
        '''
        :param cell_type: 产生RNN的类型['lstm','gru']
        :return:
        '''
        cell_tmp=None
        if cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.config.rnn_hidden_size)
        elif cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.config.rnn_hidden_size)
        return cell_tmp
    def bilstm_layer(self):
        cell_fw = self._witch_cell('lstm')
        cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_rate)
        cell_bw = self._witch_cell('lstm')
        cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_rate)
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedding_input,
                                                                        self.sequence_lengths,dtype=tf.float32)
        self.lstm_out_put = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        self.lstm_out_put=tf.layers.dense(self.lstm_out_put,self.config.rnn_hidden_size,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),activation=None)
        return self.lstm_out_put
    def crf_layer(self,input):
        from tensorflow.contrib.layers.python.layers import initializers
        self.output_layer = tf.reshape(input, [-1,input.shape[-1].value])
        self.pred = tf.layers.dense(self.output_layer, self.config.num_labels,activation=tf.tanh,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.logits = tf.reshape(self.pred, [-1, self.config.max_seq_length, self.config.num_labels])
        trans = tf.get_variable(
            "transitions",
            shape=[self.config.num_labels, self.config.num_labels],
            initializer=initializers.xavier_initializer())
        self.log_likelihood, self.transition_params =crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
                                                               sequence_lengths=self.sequence_lengths,transition_params=trans)
        return self.log_likelihood,self.transition_params,self.logits
    def add_bilstm_crf_layer(self):
        self.log_likelihood, self.transition_params,self.logits=None,None,None
        if self.config.only_use_crf:
            self.log_likelihood, self.transition_params,self.logits=self.crf_layer(self.embedding_input)
        else:
            self.lstm_out_put=self.bilstm_layer()
            self.log_likelihood, self.transition_params,self.logits = self.crf_layer(self.lstm_out_put)
        return self.log_likelihood, self.transition_params,self.logits


