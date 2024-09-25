"""
Класс MinHashing, который строит сигнатуру minHash (в форме вектора или множества) 
заданной длины n из заданного множества целых чисел (множества хешированных шинглов).
"""
import numpy as np
import sympy


class MinHashing:

    def __init__(self, seed=1337):
        self.seed = seed


    def _get_prime_above(self, n):
        """
        Функция ищет наименьшее простое число, которое превышает заданное число n.
        :param n целое число.
        Функция должна вернуть следующее простое число после n.
        """
        return sympy.nextprime(n)

    def get_signature_matrix_hash(self, shingles, signature_len=100):
        """
        Функция создает матрицу сигнатур.
        Матрица формируется с помощью хеширования.
        :param shingles: список множеств шинглов.
        :param signature_len: длина сигнатуры.
        :return: numpy массив, представляющий матрицу сигнатур.
        Документы расположены в матрице столбцами.
        """
        np.random.seed(self.seed) # set numpy seed

        rows = set.union(*shingles)     # Extract shingle id:s that exist in our documents
        max_shingle_count = max(rows)
        prime = self._get_prime_above(max_shingle_count)
        result = np.ones((signature_len, len(shingles)), dtype=np.int)*prime  # Initialize the signature matrix

        """Create hashes in the form h(r) = (a*r+b) % c"""
        a = np.random.randint(0, np.iinfo(np.int32).max, signature_len)
        b = np.random.randint(0, np.iinfo(np.int32).max, signature_len)
        c = np.ones_like(a) * prime

        # Iterate over each shingle (row in this case)
        for r in rows:
            # Calculate hash of row
            r_hashes = (a * r + b) % c
            # Iterate over each document
            # TODO: Make more efficient using numpy operations
            for doc in range(len(shingles)):
                if r in shingles[doc]:
                    # Update matrix if document contains shingle with id r
                    result[:, doc] = np.where(result[:, doc] < r_hashes, result[:, doc], r_hashes)

        return result

    def get_signature_matrix_permutations(self, shingles, signature_len=100):
        """
        Функция создает матрицу сигнатур.
        Матрица формируется с помощью перестановок.
        :param shingles: список множеств шинглов.
        :param signature_len: длина сигнатуры.
        :return: numpy массив, представляющий матрицу сигнатур.
        Документы расположены в матрице столбцами.
        """
        np.random.seed(self.seed)
        result = np.zeros((signature_len, len(shingles)), dtype=np.int)

        # permutations = [[1,3,7,6,2,5,4], [4,2,1,3,6,7,5], [3,4,7,6,1,2,5]]

        for i in range(signature_len):
            order = np.random.permutation(100)
            # order = permutations[i]
            handle_document = set(range(len(shingles)))
            for idx, j in enumerate(order):
                handle_document_temp = handle_document.copy()
                for doc in handle_document_temp:
                    if j in shingles[doc]:
                        result[i, doc] = int(idx) + 1
                        handle_document.remove(doc)

        return result



def main():
    # Example usage of this class

    min_hashing_object = MinHashing()
    # example_shingles = [{0, 1, 5, 6}, {2, 3, 4}, {0, 5, 6},  {1, 2, 3, 4}]
    # signature_matrix = min_hashing_object.get_signature_matrix_hash(example_shingles)

    from data_reader import DataLoader
    data_loader = DataLoader("Data/bbc-text.csv")
    docs = data_loader.get_documents(nr_docs=20, char_dim=500)
    from shingling import Shingling
    shingling = Shingling(docs, 9)
    shingling.docs_to_hashed_shingles()
    shingle_sets = shingling.hashed_shingles

    signature_matrix = min_hashing_object.get_signature_matrix_hash(shingle_sets, signature_len=100)

    # from lsh import LSH
    # lsh = LSH(signature_matrix)
    # # 20 документов и порог 0.3
    # candidate_pairs = lsh.get_candidate_pairs(0.5, band_method=1)
    # print("Num pairs: ", len(candidate_pairs))
    # print(candidate_pairs)

    return


if __name__ == "__main__":
    main()
