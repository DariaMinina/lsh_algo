"""
Загрузка и чтение датасета с BBC заголовками
https://www.kaggle.com/yufengdev/bbc-fulltext-and-category
"""

import pandas as pd

class DataReader:
    def __init__(self, file_path):
        """
        :param file_path: пути до файла с датасетом
        """
        self.df = pd.read_csv(file_path)

    def get_documents(self, nr_docs, char_dim=None, seed=123, randomize=True):
        """
        Возвращает список nr_docs с длиной char_dim
        :param nr_docs:
        :param char_dim:
        :param seed:
        :param randomize:
        :return:
        """
        if randomize:
            full_docs = self.df.sample(frac=1, random_state=seed)['text'].tolist()[:nr_docs]
        else:
            full_docs = self.df['text'].tolist()[:nr_docs]
        if char_dim is None:
            return full_docs
        return [doc[:char_dim] for doc in full_docs]

if __name__ == "__main__":
    # Пример использования данного класса
    file_path = '../data/bbc-text.csv'
    nr_docs = 20
    char_dim = 10
    data_reader = DataReader(file_path)
    print(data_reader.get_documents(nr_docs, char_dim))