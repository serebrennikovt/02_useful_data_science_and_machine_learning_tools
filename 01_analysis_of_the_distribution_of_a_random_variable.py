# Анализ распределения случайной величины.
# Генерация распределений - делаем синтаксический датасет по заданным параметрам распределений (пара вариантов из непрерывных функций );
# Построение гистограммы распределения при помощи matplotlib, seaborn;
# Анализ основных метрик распределения с помощью pandas и numpy, выводы по оценкам случайной величины и форме распределения.

# Импорт библиотек.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special as sps
import warnings
from scipy.stats import kurtosis
from scipy.stats import skew

plt.rcParams["figure.figsize"] = (10.0, 7.0)
warnings.filterwarnings("ignore")

# Пусть у нас будут два датасета - прибыль по клиентам разных групп, распределенные по нормальному закону N(500,1000) и
# Гамма закону, параметры которого будет необходимо вычислить на основе средней прибыли = 500 и среднеквадратическому отклонению = 1000.
# Сгенерируем синтетические выборки размера ns = 10000 клиентов каждая.

ns = 10000
n_mean = 500
n_sigma = 1000

# Используем имитацию нормального распределения.

v = np.random.normal(n_mean, n_sigma, ns)
v = pd.DataFrame(v, columns=["volume"])
v.volume = round(v.volume, 0)
print(v.head())

# Для оценки среднего, среднеквадратического отклонения и квантилей можно воспользоваться отдельными методами pandas и numpy.

print(v.volume.mean())  # Среднее значение.
print(v.volume.median())  # Значение медианы.
print(v.volume.value_counts().nlargest(10))  # Значение моды.
print(v.volume.std())  # Среднеквадратическое отклонение.
print(np.percentile(v.volume, 50))  # Квантиль - медиана.
print(np.percentile(v.volume, 75))  # Квантиль - 0.75.

# Воспользуемся встроенным методом pandas describe().

print(v.volume.describe())

# Построим гистограмму распределения - сделаем это двумя способами, при помощи seaborn и встроенного метода pandas hist().

# При помощи seaborn
sns.distplot(v)
plt.title("Распределение прибыли по пользователям группы 1")
plt.show()  # показ гистограммы.

# При помощи pandas
v.volume.hist(bins=100)
plt.title("Распределение прибыли по пользователям группы 1")
plt.show()

print(kurtosis(v.volume))  # Эксцесс.
print(skew(v.volume))  # Ассиметрия.

# Объединим в функцию для удобства работы со второй выборкой.

def my_basic_research(df=v, column="volume"):
    print("Базовые метрики")
    print(df[column].describe())
    print("------------------------------------")

    print("Самые популярные значения метрики, топ 5")
    print(df[column].value_counts().nlargest(5))
    print("------------------------------------")

    print("Эксцесс ", kurtosis(df[column]))
    print("Ассиметрия ", skew(df[column]))

    sns.distplot(df[column])
    plt.title("Распределение прибыли по пользователям")
    plt.show()

my_basic_research()

# Проверим на ассиметричном распределении: Зададим функцию для поиска параметров Гамма распределения по среднему и СКО.

def gamma_params(mean, std):
    shape = round((mean / std) ** 2, 4)
    scale = round((std ** 2) / mean, 4)
    return (shape, scale)

shape, scale = gamma_params(n_mean, n_sigma)
df = np.random.gamma(shape, scale, ns)
df = pd.DataFrame(df, columns=["volume"])
df.volume = round(df.volume, 0)  # округлим до целых.
print(df.head())
my_basic_research(df=df, column="volume")

# Как можно увидеть, теперь мы имеем дело c ассиметричным распределением и все квантили,
# а также коэффициенты ассиметрии и эксцесса поменялись, несмотря на равенство средних и СКО - выводы по этим датасетам получаются абсолютно разные.

# К примеру, сравним долю убыточных клиентов для первого и второго датасетов:
print(v[v.volume < 0].count() / len(v))
print(df[df.volume < 0].count() / len(df))

# А теперь сравним суммарную прибыль по клиентам с прибылью свыше медианы в млн:
print(v[v.volume >= np.percentile(v.volume, 50)].volume.sum() / 10 ** 6)
print(df[df.volume >= np.percentile(df.volume, 50)].volume.sum() / 10 ** 6)

# Как можно видеть, эффект от ассимерии при одинаковых средних и ско существенно меняет выводы.
