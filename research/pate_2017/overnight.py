# this is python code to run all of the teachers in a row,
# and then train the student. just so that i (miguel) can leave
# it running overnight.
# specifically, this trains a student on the svhn dataset with
# 250 teacher models and the LNMax aggregation function. this will
# hopefully be changeable in the future but i want to run a benchmark


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import deep_cnn
import input  # pylint: disable=redefined-builtin
import metrics
import tensorflow.compat.v1 as tf
from train_student import *

tf.disable_eager_execution()

def train_teacher(dataset, nb_teachers, teacher_id):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
  assert input.create_dir_if_needed(FLAGS.data_dir)
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
  elif dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = input.ld_cifar10()
  elif dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = input.ld_mnist()
  else:
    print("Check value of dataset flag")
    return False

  # Retrieve subset of data for this teacher
  data, labels = input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  if FLAGS.deeper:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt'
  else:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
  ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training
  assert deep_cnn.train(data, labels, ckpt_path)

  # Append final step value to checkpoint for evaluation
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  # Retrieve teacher probability estimates on the test data
  teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(teacher_preds, test_labels)
  print('Precision of teacher after training: ' + str(precision))

  return True

def main():
    """
    This function is the main function in this file. This trains an ensemble
    of n_teach teachers, and then uses that to train a student. The main idea
    is just that I want to run the code.
    """
    n_teach = 250
    for i in range(n_teach):
        print(f"running teacher {i} now!")
        train_teacher(dataset='svhn',nb_teachers=n_teach,teacher_id=i)
    print("done with teachers! on to the student!")
    train_student(dataset='svhn',nb_teachers=n_teach)

if __name__ == '__main__':
    main()