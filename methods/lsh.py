"""
Класс LSH, реализующий технику LSH: при заданной коллекции сигнатуров minhash (векторов целых чисел) и порогового значения t,
класс LSH (используя группировку и хеширование) находит все кандидатные пары сигнатур,
которые согласуются по меньшей мере на долях t своих компонентов.
"""

class LSH:

    def __init__(self, signature_matrix):
        """
        Инициализация с матрицей сигнатуры.
        :param signature_matrix: n*c матрица numpy, где n - длина сигнатуры и c - количество документов.
        """
        self.signature_matrix = signature_matrix
        self.signature_matrix.flags.writeable = False
        self.b = 0
        self.r = 0

    def find_b_and_r(self, threshold, band_method=1, confidence=0.95):
        """
        Найдет и установит числа полос и строк для этого объекта. В этой реализации есть три метода:
            0: Выбирает полосы так, чтобы пороговое значение было равно (1/бандов)^(1/строк).
            1: Выбирает полосы так, чтобы доверие было равно 1 - (1 - порог^строк)^бандов.
            2: Использует принцип клетки и шкафа, чтобы обеспечить, что все сигнатуры, совпадающие на заданный порог.
        Все эти методы гарантируют, что банды*строки=длина сигнатуры.

        :param threshold: Пороговое значение для кандидатных пар
        :param band_method: Метод выбора полос.
        :param confidence: Доля того, сколько с заданным порогом подобранности мы должны включать. Это нужно только если используется метод 1.
        :return:
        """
        if band_method == 0:
            b, r = self._get_best_approx_band(threshold)
        elif band_method == 1:
            b, r = self._get_best_confident_band(threshold, confidence)
        else:
            b, r = self._get_best_confident_band(threshold)
        self.b = b
        self.r = r

    def get_candidate_pairs(self):
        """
        Получает список кандидатных пар. Каждая пара представлена как кортеж индексов.
        :return: список кандидатных пар
        """
        candidate_pairs = set() # Сет кандидатных пар
        for b_idx in range(self.b):
            # Итерация по каждой строке
            boxes = dict() # Создаем пустой словарь, где мы собираем боксы и отслеживаем коллизии
            for doc in range(self.signature_matrix.shape[1]):
                # Итерация по каждому документу для текущей строки и вычисляем, в какой бокс он попадает путем хеширования.
                box = hash(tuple(self.signature_matrix[b_idx*self.r:(b_idx+1)*self.r, doc]))
                if box in boxes:
                    for doc_in_box in boxes[box]:
                        # Создаем кандидатную пару, если два документа помещаются в одну бокс.
                        candidate_pairs.add((doc_in_box, doc))
                    boxes[box].add(doc)
                else:
                    # Создаем бокс в словаре, если документ первый в этом боксе.
                    boxes[box] = {doc}
        return candidate_pairs

    def _get_best_confident_band(self, threshold, confidence=0.95):
        """
        Выбирает количество полос так, чтобы точность была равна или больше заданного доверия
        :param threshold: Пороговое значение подобранности.
        :param confidence: Желаемая точность.
        :return: банды, строки
        """
        # Расчет: confidence = 1 - (1 - t^r)^b
        signature_length = self.signature_matrix.shape[0]
        b = signature_length
        r = signature_length / b
        b_hat = b
        r_hat = r
        confidence_hat = 1 - (1 - threshold**r)**b
        # полный перебор, тестит все пары полос и строк, обеспечивая банды*строки=длина сигнатуры
        while b > 0:
            if signature_length % b != 0:
                b -= 1
                continue
            r = signature_length / b
            confidence_temp = 1 - (1 - threshold**r)**b
            if abs(confidence_hat - confidence) > abs(confidence_temp - confidence) and confidence_temp > confidence:
                b_hat = b
                r_hat = r
                confidence_hat = confidence_temp
            b -= 1

        return b_hat, int(r_hat)

    def _get_best_band_ensuring_t(self, threshold):
        """
        Основан на принципе клетки и шкафа.
        Обеспечивает отсутствие ложноположительных результатов, но склонен включать ложнегативные результаты.
        :param threshold:
        :return: банды, строки
        """
        # Вычисляем наименьшее количество ящиков для обеспечения того, что каждый не может содержать разные элементы
        signature_len = self.signature_matrix.shape[0]
        eps = 1e-9 #Needed for floating point errors
        pigeonholes = int(signature_len * (1 - threshold) + 1 + eps)
        # Получаем наименьшее число ящиков, равное или больше требуемого,
        # при этом оставаясь множителем длины сигнатуры
        while pigeonholes <= signature_len:
            if signature_len % pigeonholes == 0:
                break
            pigeonholes += 1

        return pigeonholes, int(signature_len / pigeonholes)

    def _get_best_approx_band(self, threshold):
        """
        Примерирует количество полос по формуле: threshold = (1/b)^(1/r)
        :param threshold: Пороговое значение
        :return: банды, строки
        """
        # threshold = (1/b)^(1/r)
        # обеспечить n = r * b
        # оптимизация, найти r * b такое, чтобы (t - (1/b)^(1/r)) было минимальным
        signature_length = self.signature_matrix.shape[0]
        b = signature_length
        r = signature_length / b
        b_hat = b
        r_hat = r
        threshold_hat = (1 / b) ** (1 / r)
        # полный перебор, тестит все пары полос и строк, обеспечивая банды*строки=длина сигнатуры
        while b > 0:
            if signature_length % b != 0:
                b -= 1
                continue
            r = signature_length / b
            threshold_temp = (1 / b) ** (1 / r)
            if abs(threshold_hat - threshold) > abs(threshold_temp - threshold):
                b_hat = b
                r_hat = r
                threshold_hat = threshold_temp
            b -= 1
        return b_hat, int(r_hat)


def main():
    # Пример использования этого класса
    from min_hashing import MinHashing
    min_hashing_object = MinHashing()
    example_shingles = [{0, 1, 5, 6}, {2, 3, 4}, {0, 5, 6},  {1, 2, 3, 4}]
    signature_matrix = min_hashing_object.get_signature_matrix_hash(example_shingles, signature_len=100)

    # lsh = LSH(np.zeros((100, 4)), 0.8)
    lsh = LSH(signature_matrix)
    print(lsh._get_best_approx_band(0.8))
    print(lsh._get_best_confident_band(0.8, confidence=0.95))
    print(lsh._get_best_band_ensuring_t(0.8))
    lsh.find_b_and_r(0.8)
    candidates = lsh.get_candidate_pairs()


if __name__ == '__main__':
    main()
