"""
Класс Shingling, который строит k–шинглы заданной длины k (например, 10) 
из заданного документа, вычисляет хеш-значение для каждого уникального шингла,
 и представляет документ в виде множества его хешированных k-шинглов.

Ключевые моменты:
K-шинглы - это подпоследовательности символов заданной длины k.
Класс Shingling создает эти шинглы из входного документа.
Для каждого уникального шингля вычисляется хеш-значение.
Результат представлен как набор хешированных k-шинглов.
"""

class Shingling:
    def __init__(self, documents, k=10, hash_power=32):
        """
        :param documents: список документов (string)
        :param k: int
        :param hash_power: int
        :return список множеств, каждое множество - хешированные k-шинглы документов
        """
        self.documents = documents
        self.hashed_shingles = []
        self.hash_power = hash_power
        self.k = k

    def docs_to_hashed_shingles(self):
        """
        Конвертирует каждый документ в набор сетов хешированных значений, формирует список
        :return:
        """
        for document in self.documents:
            shingle_set = self.doc_to_shingles(document)
            hashed_shingle_set = self.hash_shingles(shingle_set)
            self.hashed_shingles.append(hashed_shingle_set)

    def doc_to_shingles(self, document):
        """
        Создает набор уникальных k-шинглов на уровне символов
        :param document: документ в виде строки
        :return:
        """
        document = document.replace(' ', '_')
        shingles_zip = zip(*[document[i:] for i in range(self.k)])
        shingles_duplicates = [''.join(shingle) for shingle in shingles_zip]
        # set contains only unique shingles
        return set(shingles_duplicates)

    def hash_shingles(self, shingle_set):
        """
        Преобразует строки в целые числа с хешированием в диапазоне [0, 2^hash_power - 1)
        :param shingle_set: Множество шинглов, где каждый шингл представлен в виде хешируемого типа.
        :return:
        """
        hashed_shingle_set = set()
        for shingle in shingle_set:
            hash_value = hash(shingle) % (2 ** self.hash_power - 1)
            hashed_shingle_set.add(hash_value)
        return hashed_shingle_set


if __name__ == "__main__":
    # Пример использование данного класса
    documents = ['name is', 'my name is Dasha']
    k = 4

    shingling = Shingling(documents, k)
    shingling.docs_to_hashed_shingles()
    print(shingling.hashed_shingles)
    # [{2138512553, 2032450338, 528754195, 3190044570},
    #  {2032450338, 1276224325, 1309753989, 1320929415, 3914169448, 2138512553, 
    #       4164588524, 528754195, 1227302452, 3429845621, 600696246, 1730583606, 3190044570}]

