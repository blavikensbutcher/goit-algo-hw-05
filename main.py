import timeit
from typing import List
import requests

from kmp_search import kmp_search
from moore import boyer_moore_search
from rabin import rabin_karp_search


class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def delete(self, key: str):
        key_hash = self.hash_function(key)
        if self.table[key_hash] is not None:
            for pair in self.table[key_hash]:
                if pair[0].lower() == key.lower():
                    return self.table[key_hash].remove(pair)
        else:
            return None

    def insert(self, key, value):
        key_hash = self.hash_function(key)
        key_value = [key, value]

        if self.table[key_hash] is None:
            self.table[key_hash] = list([key_value])
            return True
        else:
            for pair in self.table[key_hash]:
                if pair[0] == key:
                    pair[1] = value
                    return True
            self.table[key_hash].append(key_value)
            return True

    def get(self, key):
        key_hash = self.hash_function(key)
        if self.table[key_hash] is not None:
            for pair in self.table[key_hash]:
                if pair[0] == key:
                    return pair[1]
        return None


print("-------------------------------------")
print("--------------FIRST------------------")
print("-------------------------------------")
# Тестуємо нашу хеш-таблицю:
H = HashTable(5)
H.insert("apple", 10)
H.insert("orange", 20)
H.insert("banana", 30)
H.delete("orange")

print(H.get("apple"))  # Виведе: 10
print(H.get("orange"))  # Виведе: 20
print(H.get("banana"))  # Виведе: 30
print("-------------------------------------")
print("---------------SECOND----------------")
print("-------------------------------------")

array = sorted([0.1, 2.5, 3.7, 4.4, 4.5, 4.6, 5.9, 6.0, 6.1, 7.8, 8.9])


def binary_search(lst: List[float], value: float):
    iteration = 0
    low = 0
    high = len(lst) - 1
    upper_bound = None

    while low <= high:
        iteration += 1
        mid = (low + high) // 2
        if lst[mid] < value:
            low = mid + 1
        else:
            upper_bound = lst[mid]
            high = mid - 1

    return iteration, upper_bound


print(binary_search(array, 4.55))

print("-------------------------------------")
print("---------------THIRD----------------")
print("-------------------------------------")



url_first = (
    "https://drive.google.com/uc?export=download&id=18_R5vEQ3eDuy2VdV3K5Lu-R-B-adxXZh"
)
url_second = (
    "https://drive.google.com/uc?export=download&id=18BfXyQcmuinEI_8KDSnQm4bLx6yIFS_w"
)


def get_article(url: str, encoding: str = "utf-8") -> str | None:
    response = requests.get(url)
    if response.status_code == 200:
        decoded_article = response.content.decode(encoding)
        return decoded_article
    else:
        raise ConnectionError("Connection error")


first_article = get_article(url_first, "cp1251")
second_article = get_article(url_second)

first_article_pattern = "теорії алгоритмів жадібні алгоритми"
second_article_pattern = "Графові моделі СУБД"
not_real_pattern = "Жакуф Петрович"


def run_test():
    karp_position_first = timeit.timeit(
        lambda: rabin_karp_search(first_article, first_article_pattern), number=1
    )
    karp_position_second_real = timeit.timeit(
        lambda: rabin_karp_search(second_article, second_article_pattern), number=1
    )
    karp_position_second_fake = timeit.timeit(
        lambda: rabin_karp_search(second_article, not_real_pattern), number=1
    )

    moore_position_first = timeit.timeit(
        lambda: boyer_moore_search(first_article, first_article_pattern), number=1
    )
    moore_position_second_real = timeit.timeit(
        lambda: boyer_moore_search(second_article, second_article_pattern), number=1
    )
    moore_position_second_fake = timeit.timeit(
        lambda: boyer_moore_search(second_article, not_real_pattern), number=1
    )

    kmp_position_first = timeit.timeit(
        lambda: kmp_search(first_article, first_article_pattern), number=1
    )
    kmp_position_second_real = timeit.timeit(
        lambda: kmp_search(second_article, second_article_pattern), number=1
    )
    kmp_position_second_fake = timeit.timeit(
        lambda: kmp_search(second_article, not_real_pattern), number=1
    )

    return {
        "karp_position_first": karp_position_first,
        "karp_position_second_real": karp_position_second_real,
        "karp_position_second_fake": karp_position_second_fake,
        "moore_position_first": moore_position_first,
        "moore_position_second_real": moore_position_second_real,
        "moore_position_second_fake": moore_position_second_fake,
        "kmp_position_first": kmp_position_first,
        "kmp_position_second_real": kmp_position_second_real,
        "kmp_position_second_fake": kmp_position_second_fake,
    }


def print_results(results):
    print("Час виконання (в секундах):")
    print("\n--- Перша стаття (реальний підрядок) ---")
    print(f"Rabin-Karp: {results['karp_position_first']:.6f}")
    print(f"Boyer-Moore: {results['moore_position_first']:.6f}")
    print(f"KMP: {results['kmp_position_first']:.6f}")

    print("\n--- Друга стаття (реальний підрядок) ---")
    print(f"Rabin-Karp: {results['karp_position_second_real']:.6f}")
    print(f"Boyer-Moore: {results['moore_position_second_real']:.6f}")
    print(f"KMP: {results['kmp_position_second_real']:.6f}")

    print("\n--- Друга стаття (вигаданий підрядок) ---")
    print(f"Rabin-Karp: {results['karp_position_second_fake']:.6f}")
    print(f"Boyer-Moore: {results['moore_position_second_fake']:.6f}")
    print(f"KMP: {results['kmp_position_second_fake']:.6f}")

    print("\n--- Висновки ---")

    first_best = min(
        ("Rabin-Karp", results['karp_position_first']),
        ("Boyer-Moore", results['moore_position_first']),
        ("KMP", results['kmp_position_first']),
        key=lambda x: x[1]
    )
    print(f"Найшвидший для першої статті: {first_best[0]} ({first_best[1]:.6f} сек)")

    second_best_real = min(
        ("Rabin-Karp", results['karp_position_second_real']),
        ("Boyer-Moore", results['moore_position_second_real']),
        ("KMP", results['kmp_position_second_real']),
        key=lambda x: x[1]
    )
    print(f"Найшвидший для другої статті (реальний): {second_best_real[0]} ({second_best_real[1]:.6f} сек)")

    second_best_fake = min(
        ("Rabin-Karp", results['karp_position_second_fake']),
        ("Boyer-Moore", results['moore_position_second_fake']),
        ("KMP", results['kmp_position_second_fake']),
        key=lambda x: x[1]
    )
    print(f"Найшвидший для другої статті (вигаданий): {second_best_fake[0]} ({second_best_fake[1]:.6f} сек)")

    total_avg = {
        "Rabin-Karp": (
            results['karp_position_first'] +
            results['karp_position_second_real'] +
            results['karp_position_second_fake']
        ) / 3,
        "Boyer-Moore": (
            results['moore_position_first'] +
            results['moore_position_second_real'] +
            results['moore_position_second_fake']
        ) / 3,
        "KMP": (
            results['kmp_position_first'] +
            results['kmp_position_second_real'] +
            results['kmp_position_second_fake']
        ) / 3,
    }

    best_overall = min(total_avg.items(), key=lambda x: x[1])
    print(f"Найшвидший загалом: {best_overall[0]} (середній час: {best_overall[1]:.6f} сек)")

# Виклик
results = run_test()
print_results(results)