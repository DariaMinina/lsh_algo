from pathlib import Path
from itertools import combinations
import argparse

from methods.data_reader import DataReader
from methods.shingling import Shingling
from methods.compare_sets import CompareSets
from methods.compare_signatures import CompareSignatures
from methods.min_hashing import MinHashing
from methods.lsh import LSH



def main(args):
    """
    Нахождение похожих документов и вывод их в терминал
    :param args: аргументы
    :return:
    """

    method = methods[args.method.lower()]

    # Загрузка данных 
    data_loader = DataReader(args.filename)
    docs = data_loader.get_documents(args.ndocuments, randomize=False)

    # Создание сетов шинглов
    shingling = Shingling(docs, args.kshingle)
    shingling.docs_to_hashed_shingles()
    shingle_sets = shingling.hashed_shingles

    if method == 0:
        # Реализация LSH

        # Получение сигнатур документов
        min_hashing_object = MinHashing()
        signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets,
                                                                        signature_len=args.signaturelength)
        # Нахождение и вывод похожих документов
        lsh = LSH(signature_matrix)
        lsh.find_b_and_r(args.threshold)
        candidates = lsh.get_candidate_pairs()
        print("Pairs of documents that are estimated to have a similarity score equal to the provided threshold. "
            "(Note: Index staring at 0)\nIdentified candidate pairs:\nPair : similarity")
        for pair in candidates:
            print("{} : {:.2f}+".format(pair, args.threshold))

    elif method == 1:
        # Реализация signature similarity

        # Получение сигнатур документов
        min_hashing_object = MinHashing()
        signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets,
                                                                        signature_len=args.signaturelength)

        # Расчет и вывод скоров
        print("Similarity measure between the signatures. An approximation of the Jaccard similarity between hashed "
            "shingle sets.\n(Note: Only document pairs that are similar larger to the provided threshold is shown "
            "below)"
            "\nPair : similarity")
        indices = list(range(signature_matrix.shape[1]))
        for pair_idx in combinations(indices, 2):
            similarity = CompareSignatures.similarity(signature_matrix[:, (pair_idx[0], pair_idx[1])])
            if similarity >= args.threshold:
                print("{} : {:.2f}".format(pair_idx, similarity))

    elif method == 2:
        # Реализация Jaccard similarity

        compare_sets = CompareSets()

        # Расчет и вывод скоров
        print("Jaccard similarity between hashed shingle sets.\n(Note: Only document pairs that are similar larger to"
            " the provided threshold is shown below)"
            "\nPair : similarity")
        indices = list(range(len(shingle_sets)))
        for pair_idx in combinations(indices, 2):
            similarity = compare_sets.get_similarity(shingle_sets[pair_idx[0]], shingle_sets[ pair_idx[1]])
            if similarity >= args.threshold:
                print("{} : {:.2f}".format(pair_idx, similarity))

    return

# Путь до файла с документами по умолчанию
filename = Path('Data/bbc-text-small.csv')
# Доступные методы
methods = {'lsh': 0, 'signature': 1, 'jaccard': 2}

if __name__ == '__main__':
    # Описание ключей, определяющих параметры работы алгоритма
    parser = argparse.ArgumentParser(description='Tool for measuring similarity between documents.')
    parser.add_argument('-f', '--filename', type=str, default=filename,
                        help='Dataset path. The data set is preferrably a csv file consising of two columns and a '
                            'header. The second column should contain the documents row-wise.')
    parser.add_argument('-m', '--method', type=str, default='LSH', choices=methods.keys(),
                        help='Method for finding similar documents.')
    parser.add_argument('-sl', '--signaturelength', type=int, default=100,
                        help='Length of the signatures after Min Hashing.')
    parser.add_argument('-k', '--kshingle', type=int, default=5, help='Number of characters in each k-shingle.')
    parser.add_argument('-n', '--ndocuments', type=int, default=10, help='Number of documents to consider.')
    parser.add_argument('-c', '--confidence', type=int, default=0.95,
                        help='Portion of how many with the given threshold similarity to include. If LSH is used, bands'
                            ' and rows are selected such that confidence = 1 - (1 - threshold^rows)^bands. Reducing'
                            ' the threshold increases false positives.')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='The similarity threshold for pairs.')

    args = parser.parse_args()

    main(args)
