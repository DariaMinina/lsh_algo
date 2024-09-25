"""
Программа для тестирования и оценки алгоритмах на данных разного объема. 
(время выполнения по сравнению с размером входного набора данных)
Использует классы, описанные в директории methods для поиска похожих документов.

Выбираем порог сходства s (например, 0,8), 
который показывает, что два документа считаются похожими, 
если коэффициент Джаккарда их наборов шинглов составляет не менее s.
"""


from methods.data_reader import DataReader
from methods.shingling import Shingling
from methods.compare_sets import CompareSets
from methods.min_hashing import MinHashing
from methods.compare_signatures import CompareSignatures
from methods.lsh import LSH

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import combinations
from tqdm import tqdm
import time

class Estimation:

    def __init__(self, small_file, large_file, nr_docs, char_dim, k, threshold, signature_len):
        """
        :param small_file: file path (string) to dataset with a few documents used to test algos on similarity
        :param large_file: file path (string) to large dataset of documents used to test scalability
        :param nr_docs: number of documents to read from the large dataset
        :param char_dim: int > 0, maximum length of documents
        :param k: int > 0, k in k-shingles (number of characters in each shingle)
        :param threshold: similarity threshold, float on interval [0, 1]
        :param signature_len: int > 0, dimension of signatures
        """
        self.large_docs = None
        self.small_docs = None
        self.nr_docs = nr_docs
        self.char_dim = char_dim
        self.k = k
        self.threshold = threshold
        self.signature_len = signature_len

        self.read_files(small_file, large_file)

    def read_files(self, small_file, large_file):
        small_data_loader = DataReader(small_file)
        self.small_docs = small_data_loader.get_documents(10, randomize=False)

        large_data_loader = DataReader(large_file)
        self.large_docs = large_data_loader.get_documents(self.nr_docs, self.char_dim)
        while len(self.large_docs) < self.nr_docs:
            self.large_docs.extend(self.large_docs)

    def find_similar_docs(self):
        """
        Высчитывает similarity всех комбинаций документа в небольшом датасете
        и строит бар чарт для всех трех методов.
        :return:
        """
        print("Finding similar documents: ")
        small_shingling = Shingling(self.small_docs, self.k)
        small_shingling.docs_to_hashed_shingles()
        shingle_sets = small_shingling.hashed_shingles

        min_hashing_object = MinHashing()
        signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets, signature_len=self.signature_len)
        y_sig = self.test_sig_similarity(signature_matrix)

        x, y_jac = self.test_jaccard_similarity(shingle_sets)
        similar_docs_LSH = self.find_similar_pairs_LSH(signature_matrix)[0]

        mean = (np.array(y_jac) - np.array(y_sig)).mean()
        var = (np.array(y_jac) - np.array(y_sig)).var()
        print(
            "\nMean and variance of difference between jaccard and signature similarity:\nMean: {}, variance: {}".format(
                mean, var))
        mean_abs_residual = (np.abs(np.array(y_jac) - np.array(y_sig))).mean()
        print(
            "\nMean absolute residual between jaccard and signature similarity:\nMean: {}".format(
                mean_abs_residual))
        self.bar_chart(x, y_jac, y_sig, similar_docs_LSH)

    def eval_scalability(self, nr_docs_list):
        """
        Берет список интов, nr документов, для тестирования времени работы алгоритмов и строит графики
        по всем трем алгоритмам.
        :param nr_docs_list: список количества документов на каждом шаге
        :return:
        """
        print("Evaluating scalability:")
        LSH_times = []
        min_hash_times = []
        pure_shingling_times = []
        for nr_docs in nr_docs_list:
            print("Starting for nr_docs: " + str(nr_docs))
            signature_matrix, sig_mat_time = self.get_signature_mat(nr_docs)
            LSH_times.append(self.find_similar_pairs_LSH(signature_matrix)[1] + sig_mat_time)
            min_hash_times.append(self.find_similar_pairs_min_hash(signature_matrix) + sig_mat_time)
            pure_shingling_times.append(self.find_similar_pairs_pure(nr_docs))

        self.plot_scalability(nr_docs_list, LSH_times, min_hash_times, pure_shingling_times)

    def compare_signature_lengths(self, sign_len_list):
        """
        Оценка точности приближенной схожести относительно длины сигнатуры.
        :param sign_len_list:  список количества документов на каждом шаге
        :return:
        """
        print("Evaluating approximation precision:")
        signature_len_temp = self.signature_len
        mean_abs_residual_list = [0] * len(sign_len_list)

        small_shingling = Shingling(self.small_docs, self.k)
        small_shingling.docs_to_hashed_shingles()
        shingle_sets = small_shingling.hashed_shingles
        min_hashing_object = MinHashing()

        for i in tqdm(range(len(sign_len_list))):

            signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets,
                                                                            signature_len=sign_len_list[i])
            y_sig = self.test_sig_similarity(signature_matrix)
            y_jac = self.test_jaccard_similarity(shingle_sets)[1]

            mean_abs_residual = (np.abs(np.array(y_jac) - np.array(y_sig))).mean()
            mean_abs_residual_list[i] = mean_abs_residual

        self.signature_len = signature_len_temp
        self.plot_mean(sign_len_list, mean_abs_residual_list)

    def plot_scalability(self, nr_docs_list, LSH_times, min_hash_times, pure_shingling_times):
        """
        Вспомогательная функция для постройки графиков.
        :param nr_docs_list:
        :param LSH_times:
        :param min_hash_times:
        :param pure_shingling_times:
        :return:
        """
        plt.figure()
        plt.plot(nr_docs_list, LSH_times, label='LSH')
        plt.plot(nr_docs_list, min_hash_times, label='MinHash sim')
        plt.plot(nr_docs_list, pure_shingling_times, label='Pure Shingling (Jaccard)')

        plt.title("Время исполнения для каждого из методов", fontsize=18)
        plt.xlabel("Количество документов", fontsize=14)
        plt.ylabel("Время выполнения (sec)", fontsize=14)
        plt.legend()

        plt.show()

    def get_signature_mat(self, nr_docs):
        """
        Вычисляет матрицу сигнатур из большого набора данных и возвращает ее вместе со временем вычисления.
        :param nr_docs: количество документов для включения из большого набора документов. 
        :return: время выполнения
        """
        start_time = time.time()

        large_shingling = Shingling(self.large_docs[:nr_docs], self.k)
        large_shingling.docs_to_hashed_shingles()
        shingle_sets = large_shingling.hashed_shingles

        min_hashing_object = MinHashing()
        signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets, signature_len=self.signature_len)

        return signature_matrix, time.time() - start_time

    def find_similar_pairs_LSH(self, signature_matrix):
        """
        Вспомогательная функция для поиска пар кандидатов, найденных с помощью LSH.
        Также возвращает время вычисления.
        :param signature_matrix: матрица сигнатур
        :return: пары кандидатов и время выполнения
        """
        lsh = LSH(signature_matrix)
        lsh.find_b_and_r(self.threshold, band_method=1, confidence=0.95)

        start_time = time.time()
        candidates_idx = lsh.get_candidate_pairs()
        return candidates_idx, time.time() - start_time

    def find_similar_pairs_min_hash(self, signature_matrix):
        """
        Вовращает пары кандидатов, найденных с помощью minhash
        Также возвращает время вычисления.
        :param signature_matrix: матрица сигнатур
        :return: пары кандидатов и время выполнения
        """
        start_time = time.time()

        indices = list(range(signature_matrix.shape[1]))
        similar_pairs = []
        for pair_idx in combinations(indices, 2):
            similarity = CompareSignatures.similarity(signature_matrix[:, (pair_idx[0], pair_idx[1])])
            if similarity >= self.threshold:
                similar_pairs.append(pair_idx)

        return time.time() - start_time

    def find_similar_pairs_pure(self, nr_docs):
        """ 
        Использование хэшированных шинглов и сравнениe с jaccard similarity
        :param nr_docs: nr документов для сравнения
        :return: время выполнения
        """
        start_time = time.time()

        large_shingling = Shingling(self.large_docs[:nr_docs], self.k)
        large_shingling.docs_to_hashed_shingles()
        shingle_sets = large_shingling.hashed_shingles

        compare_sets = CompareSets()
        similar_pairs = []

        for set_pair in combinations(shingle_sets, 2):
            similarity = compare_sets.get_similarity(set_pair[0], set_pair[1])
            if similarity >= self.threshold:
                similar_pairs.append(set_pair)
        return time.time() - start_time

    def test_sig_similarity(self, signature_matrix):
        """
        Впомогательная функция. Нахождения signature similarity между всеми возможными парами.
        :param signature_matrix: матрица сигнатур
        :return: список "похожестей"
        """
        indices = list(range(signature_matrix.shape[1]))
        y = []
        for pair_idx in combinations(indices, 2):
            similarity = CompareSignatures.similarity(signature_matrix[:, (pair_idx[0], pair_idx[1])])
            y.append(similarity)
        return y

    def test_jaccard_similarity(self, shingle_sets):
        """
        Впомогательная функция. Нахождения jaccard similarity между всеми возможными парами.
        :param shingle_sets: Список множеств шинглов
        :return: x индекс и список "похожестей".
        """
        compare_sets = CompareSets()
        x = list(range(len(list(combinations(shingle_sets, 2)))))
        y = []

        for set_pair in combinations(shingle_sets, 2):
            y.append(compare_sets.get_similarity(set_pair[0], set_pair[1]))
        return x, y

    def bar_chart(self, x, y_jac, y_sig, similar_docs_LSH):
        """
        Вспомогательная функция для построения бар чарта
        :param x:
        :param y_jac:
        :param y_sig:
        :param similar_docs_LSH:
        :return:
        """
        labels = [pair for pair in combinations(range(len(self.small_docs)), 2)]

        plt.bar(x, y_jac, alpha = 0.7, label='Pure shingles')
        plt.bar(x, y_sig, alpha = 0.7, label='MinHash dim ' + str(self.signature_len))
        plt.scatter([labels.index(pair) for pair in similar_docs_LSH], [self.threshold for _ in similar_docs_LSH],
                    marker="+",  edgecolor='black', linewidth=0.5, s=128, color="red", label="LSH кандидат",
                    zorder=3)

        plt.title("Сходство комбинаций пар документов", fontsize=20)
        plt.xlabel("пара", fontsize=14)
        plt.xticks(x, labels, rotation=75)
        plt.ylabel("Сходство", fontsize=14)
        plt.legend()

        plt.show()

    @staticmethod
    def plot_mean(sign_len_list, mean):
        """
        Вспомогательная функция для отображения зависимости длины сигнатуры и среднего ошибки
        :param sign_len_list:
        :param mean:
        :return:
        """
        plt.figure()
        plt.plot(sign_len_list, mean, label='Среднее значение ошибки')

        plt.title("Среднее значение величины ошибки по мере увеличения длины сигнатуры.",
                fontsize=12)
        plt.xlabel("Длина сигнатуры", fontsize=12)
        plt.xscale('log', basex=3)




        plt.ylabel("Ошибка", fontsize=12)

        plt.show()
        return


if __name__ == "__main__":
    """
    Использованные наборы данных: 
        small_file: Путь к файлу с 5-10 документами для оценки сходства.
        large_file: Путь к файлу с большим количеством документов (1000+) для оценки масштабируемости.
    """
    small_file = 'Data/bbc-text-small.csv'
    large_file = 'Data/bbc-text.csv'

    """
    Параметры:
        char_dim: Длина документов
        k: длина шинглов
        threshold: Порог, используемый для алгоритма LSH
        signature_length: Желаемое значение длины сигнатур по умолчанию
        nr_docs_list: Список количества документов, использованных при оценке временной сложности.
        sign_len_list: Список длин сигнатур, используемых для оценки точности приблизительного сходства.
    """
    char_dim = 500
    k = 5
    threshold = 0.6
    signature_len = 100
    nr_docs_list = [5, 10, 20, 50, 100, 500, 1000, 2000, 4000]
    sign_len_list = [int(3**(x*0.1)) for x in range(40, 100, 1)]  # List of signatures to compare the approximation on


    """
    Задача оценки:
        1. Найти и построить сходства между документами.
        2. Оценить масштабируемость по количеству документов.
        3. Сравнить точность приближенного сходства по длине сигнатуры.
    """
    eval = Estimation(small_file, large_file, max(nr_docs_list), char_dim, k, threshold, signature_len)

    print("-----------------------------------------")
    eval.find_similar_docs()
    print("-----------------------------------------")
    eval.eval_scalability(nr_docs_list)
    print("-----------------------------------------")
    eval.compare_signature_lengths(sign_len_list)
    print("-----------------------------------------")
    print("Estimation Done.")






