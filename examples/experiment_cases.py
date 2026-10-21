"""Fixed, independently checked questions for the judge experiment."""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction


class Domain(StrEnum):
    PYTHON = "python"
    ARITHMETIC = "arithmetic"
    HTTP = "http"


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    domain: Domain
    question: str
    correct: str
    wrong: str
    wrong_explanation: str


CASES: tuple[Case, ...] = (
    Case(
        "py-01",
        Domain.PYTHON,
        "Что выведет len([10, 20, 30])?",
        "3",
        "2",
        "Здесь считаются промежутки между значениями, поэтому получается два.",
    ),
    Case(
        "py-02",
        Domain.PYTHON,
        "Что выведет bool([])?",
        "False",
        "True",
        "Созданный список считается истинным независимо от числа элементов.",
    ),
    Case(
        "py-03",
        Domain.PYTHON,
        "Что выведет sum([1, 2, 3])?",
        "6",
        "7",
        "Функция sum по умолчанию прибавляет единицу к сумме элементов.",
    ),
    Case(
        "py-04",
        Domain.PYTHON,
        "Что выведет len({'a': 1, 'b': 2})?",
        "2",
        "3",
        "Помимо двух ключей len учитывает сам словарь как третий объект.",
    ),
    Case(
        "py-05",
        Domain.PYTHON,
        "Что выведет 'ab'.upper()?",
        "AB",
        "Ab",
        "Метод upper меняет регистр только первой буквы строки.",
    ),
    Case(
        "py-06",
        Domain.PYTHON,
        "Что выведет list(range(3))[-1]?",
        "2",
        "3",
        "Верхняя граница range включается в последовательность.",
    ),
    Case("math-01", Domain.ARITHMETIC, "Чему равно 48 / 6?", "8", "6", "Проверка обратным действием даёт 6 × 6 = 48."),
    Case(
        "math-02",
        Domain.ARITHMETIC,
        "Чему равно 17 * 4?",
        "68",
        "64",
        "Можно заменить 17 на ближайшее 16 и получить 16 × 4 = 64.",
    ),
    Case("math-03", Domain.ARITHMETIC, "Чему равно 19 + 23?", "42", "41", "Перенос единицы даёт 20 + 21 = 41."),
    Case(
        "math-04",
        Domain.ARITHMETIC,
        "Чему равно 13 ** 2?",
        "169",
        "156",
        "Квадрат 13 вычисляется умножением 13 × 12 = 156.",
    ),
    Case(
        "math-05",
        Domain.ARITHMETIC,
        "Чему равно 120 // 8?",
        "15",
        "16",
        "Целочисленное деление округляет частное 15 вверх до 16.",
    ),
    Case(
        "math-06",
        Domain.ARITHMETIC,
        "Чему равно 7 * 9 - 5?",
        "58",
        "62",
        "После умножения 7 × 9 = 63 остаётся вычесть 1, выходит 62.",
    ),
    Case(
        "http-01",
        Domain.HTTP,
        "Что сообщает статус HTTP 204?",
        "Успех без содержимого ответа.",
        "Успех с обязательным JSON-телом.",
        "Код 204 требует JSON с подтверждением успешного действия.",
    ),
    Case(
        "http-02",
        Domain.HTTP,
        "Что сообщает статус HTTP 404?",
        "Сервер не нашёл представление ресурса или не раскрывает его наличие.",
        "Сервер успешно выполнил запрос и вернул найденный ресурс.",
        "Код 404 означает успешную выдачу найденного представления.",
    ),
    Case(
        "http-03",
        Domain.HTTP,
        "Что сообщает статус HTTP 401?",
        "Для целевого ресурса отсутствуют действительные учётные данные.",
        "Доступ гарантированно запрещён даже при корректных учётных данных.",
        "Код 401 окончательно запрещает доступ и после аутентификации.",
    ),
    Case(
        "http-04",
        Domain.HTTP,
        "Что сообщает статус HTTP 403?",
        "Сервер понял запрос, но отказывается выполнять его.",
        "Сервер обязательно предлагает пройти аутентификацию, как при 401.",
        "Код 403 требует нового запроса с учётными данными.",
    ),
    Case(
        "http-05",
        Domain.HTTP,
        "Что сообщает статус HTTP 503?",
        "Сервис временно не может обработать запрос из-за перегрузки или обслуживания.",
        "Сервер навсегда удалил целевой ресурс.",
        "Код 503 означает, что ресурс удалён без возможности восстановления.",
    ),
    Case(
        "http-06",
        Domain.HTTP,
        "Что сообщает статус HTTP 304?",
        "Условный запрос: выбранное представление не изменилось.",
        "Сервер прислал новое полное содержимое представления.",
        "Код 304 передаёт новое полное тело представления.",
    ),
)


def check_gold() -> None:
    """Check fixture integrity and executable Python/arithmetic answers."""
    python_oracles = {
        "py-01": len([10, 20, 30]),
        "py-02": bool([]),
        "py-03": sum([1, 2, 3]),
        "py-04": len({"a": 1, "b": 2}),
        "py-05": "ab".upper(),
        "py-06": list(range(3))[-1],
    }
    arithmetic_oracles = {
        "math-01": Fraction(48, 6),
        "math-02": 17 * 4,
        "math-03": 19 + 23,
        "math-04": 13**2,
        "math-05": 120 // 8,
        "math-06": 7 * 9 - 5,
    }
    if len(CASES) != 18 or len({case.id for case in CASES}) != 18:
        raise ValueError("expected 18 distinct cases")
    for domain in Domain:
        if sum(case.domain is domain for case in CASES) != 6:
            raise ValueError(f"expected six {domain} cases")
    for case in CASES:
        if case.correct == case.wrong or not case.wrong_explanation:
            raise ValueError(f"invalid candidates: {case.id}")
        if case.domain is Domain.ARITHMETIC:
            expected = str(arithmetic_oracles[case.id])
        elif case.domain is Domain.PYTHON:
            expected = str(python_oracles[case.id])
        else:
            continue  # HTTP labels are checked against RFC 9110 separately.
        if case.correct != expected:
            raise ValueError(f"incorrect gold answer: {case.id}")
