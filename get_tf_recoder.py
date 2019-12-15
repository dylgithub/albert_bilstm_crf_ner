#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import albert_modeling as modeling
import tokenization
import tensorflow as tf
from bert_config import BertConfig

config=BertConfig()
flags = tf.flags

FLAGS = flags.FLAGS
## Required parameters
flags.DEFINE_string(
    "data_dir", config.data_dir,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", config.bert_config_file,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name",config.task_name, "The name of the task to train.")

flags.DEFINE_string("vocab_file",config.vocab_file,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir",config.output_dir,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", config.init_checkpoint,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length",config.max_seq_length,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train",config.do_train, "Whether to run training.")

flags.DEFINE_bool("do_eval",config.do_eval, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict",config.do_predict,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size",config.train_batch_size, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size",config.eval_batch_size, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size",config.predict_batch_size, "Total batch size for predict.")

flags.DEFINE_float("learning_rate",config.learning_rate, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", config.num_train_epochs,
                   "Total number of training epochs to perform.")
#warmup_proportion：warm up 步数的比例，
# 比如说总共学习100步，warmup_proportion=0.1表示前10步用来warm up，
# warm up时以较低的学习率进行学习(lr = global_step/num_warmup_steps * init_lr)，
# 10步之后以正常(或衰减)的学习率来学习。
#这样做是为了在开始能够找到一个更佳的学习方向
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps",config.save_checkpoints_steps,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

#guid是该样本的唯一ID，
#text_a表示句子
#如果是test数据集则label统一为0。
class InputExample(object):
  """A single training/test example for simple sequence classification."""
  def __init__(self, guid, text_a,label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

#tokenization过后的样本数据结构，
#input_ids其实就是tokens的索引，input_mask不用解释，
#segment_ids对应模型的token_type_ids以上三者构成模型输入的X，
#label_id是标签，对应Y
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids

#读入数据的基类
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_data(cls, input_file):
    """Reads a BIO data."""
    with open(input_file,'r',encoding='utf-8') as f:
      lines = []
      words = []
      labels = []
      for line in f:
        contends = line.strip()
        word = line.strip().split('_s_')[0]
        label = line.strip().split('_s_')[-1]
        if contends.startswith("-DOCSTART-"):
          words.append('')
          continue
        if len(contends) == 0:
          l = ' '.join([label for label in labels if len(label) > 0])
          w = ''.join([word for word in words if len(word) > 0])
          lines.append([l, w])
          words = []
          labels = []
          continue
        words.append(word)
        labels.append(label)
      return lines

class NerProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "train.txt")), "train"
    )

  def get_dev_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "val.txt")), "dev"
    )

  def get_test_examples(self,data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "val.txt")), "test")

  def get_labels(self):
    return ["O", "B-ASP", "I-ASP", "B-OPI", "I-OPI","[CLS]","[SEP]"]

  def _create_example(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid,text_a=text, label=label))
    return examples
#将样本转换成features
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  labellist = example.label.split(' ')
  #注意这里从1开始，因为句子长度不够时要进行填充其对应的label填充为0
  label_map = {}
  for (i, label) in enumerate(label_list,1):
    label_map[label] = i
  #样本向字id的转换
  if len(example.text_a)>max_seq_length-2:
      example.text_a=example.text_a[0:(max_seq_length-2)]
      labellist=labellist[0:(max_seq_length-2)]
  tokens_a = tokenizer.tokenize(example.text_a)
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  #若输入的是句子对要加入特殊字符
  tokens = []
  segment_ids = []
  label_ids=[]
  tokens.append("[CLS]")
  segment_ids.append(0)
  label_ids.append(label_map["[CLS]"])
  for i,token in enumerate(tokens_a):
    tokens.append(token)
    segment_ids.append(0)
    label_ids.append(label_map[labellist[i]])
  tokens.append("[SEP]")
  segment_ids.append(0)
  label_ids.append(label_map["[SEP]"])

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    # we don't concerned about it!
    label_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
  #创建InputFeatures的一个实例化对象并返回n
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=label_ids)
  return feature
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    #创建一个有序的字典，循环遍历的输出和定义时的顺序相同
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  processors = {
      'ner': NerProcessor,
  }
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))
  tf.gfile.MakeDirs(FLAGS.output_dir)
  task_name = FLAGS.task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))
  processor = processors[task_name]()
  label_list = processor.get_labels()
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    print("训练数据一共有%d条" % len(train_examples))
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    print("验证数据一共有%d条" % len(eval_examples))
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()