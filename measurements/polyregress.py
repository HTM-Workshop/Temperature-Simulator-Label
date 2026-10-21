from dataclasses import dataclass
import numpy
import matplotlib.pyplot as plt


G_TARGET_R2 = 0.9999997
G_MAX_POLY_DEG = 4


ysi400_temp = numpy.array([
    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
])

ysi400_resist = numpy.array([
    6991. , 6646. , 6320. , 6012. , 5721. , 5446. , 5185. , 4939. ,
    4705. , 4484. , 4275. , 4078. , 3888. , 3710. , 3540. , 3380. ,
    3227. , 3083. , 2945. , 2815. , 2691. , 2573. , 2461. , 2354. ,
    2253. , 2157. , 2065. , 1978. , 1894. , 1815. , 1740. , 1668. ,
    1599. , 1534. , 1472. , 1412. , 1355. , 1301. , 1249. , 1200. ,
    1153. , 1108. , 1065. , 1023. ,  984.1,  946.5,  910. ,  876. ,
    843. ,  811.
])

ysi700_temp = numpy.array([
    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
])

ysi700_resist = numpy.array([
    18620, 17700, 16830, 16010, 15240, 14500, 13810, 13150, 12530,
    11940, 11380, 10850, 10350,  9878,  9428,  9000,  8594,  8210,
    7844,  7496,  7166,  6852,  6554,  6270,  6000,  5744,  5500,
    5266,  5046,  4834,  4634,  4442,  4260,  4084,  3918,  3760,
    3610,  3466,  3328,  3196,  3070,  2950,  2836,  2726,  2620,
    2520,  2424,  2334,  2246,  2162
])



@dataclass
class FunctionPack:
    deri: numpy.polynomial.Polynomial
    poly: numpy.ndarray
    r_sq: float
    r_range: numpy.ndarray
    t_range: numpy.ndarray
    poly_degree: int
    def __str__(self):
        out = ""
        for i, v in enumerate(self.poly[::-1]):
            exp = (len(self.poly) - 1) - i
            if exp:
                out = out + f"({v} * (temp**{exp})) + "
            else:
                out = out + f"({v})"
        return out


def calc_r2(a_true: numpy.ndarray, a_pred: numpy.ndarray) -> float:
    ss_res = numpy.sum((a_true - a_pred) ** 2)
    y_mean = numpy.mean(a_true)
    ss_tot = numpy.sum((a_true - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2
    

def compute_formulas(temp: numpy.ndarray, resist: numpy.ndarray, depth = 0) -> list[FunctionPack]:
    if len(temp) != len(resist):
        raise AssertionError(f"Mismatched input array sizes. {len(temp)} vs {len(resist)}")
    degree = 0
    while degree <= G_MAX_POLY_DEG:
        coeff = numpy.polynomial.Polynomial.fit(resist, temp, degree).convert()
        r2 = calc_r2(temp, coeff(resist))
        if r2 >= G_TARGET_R2:
            func = FunctionPack(
                deri = coeff,
                poly = coeff.convert().coef,
                r_sq = r2,
                r_range = resist,
                t_range = temp,
                poly_degree = degree
            )
            return [func]
        else:
            degree += 1
    mid_point = len(temp) // 2
    if mid_point <= 0:
        raise AssertionError(f"No matching function found for given constraints. (MAX_DEGREE: {G_MAX_POLY_DEG}, TARGET_R2: {G_TARGET_R2})")
    func_high = compute_formulas(temp[:mid_point], resist[:mid_point], depth + 1)
    func_low = compute_formulas(temp[mid_point:], resist[mid_point:], depth + 1)
    return func_low + func_high



def codegen(functions: list[FunctionPack]) -> None:
    last_r = 0
    for i, func in enumerate(functions):
        r_high = int(func.r_range[0])
        if i == 0:
            print(f"if(temp > {r_high}):")
            print(f"    resistance = {func}\n")
        elif i == len(functions) - 1:
            print("else:")
            print(f"    resistance = {func}\n")
        else:
            print(f"elif(temp <= {last_r} and temp > {r_high}):")
            print(f"    resistance = {func}\n")
        last_r = int(func.r_range[0])
    print("return resistance")


    
def main():
    ysi400_functions = compute_formulas(ysi400_resist, ysi400_temp)
    ysi700_functions = compute_formulas(ysi700_resist, ysi700_temp)

    fig, ax = plt.subplots()
    for i in ysi400_functions:
        ax.plot(i.r_range, i.deri(i.r_range))
    ax.grid()
    for i in ysi700_functions:
        ax.plot(i.r_range, i.deri(i.r_range))
    plt.show()

    print("\nYSI400 -----------------")
    for f in ysi400_functions:
        print(f"\nFor range: {f.r_range[-1]} - {f.r_range[0]} | (R2: {f.r_sq} | POLY_DEGREE: {f.poly_degree}):")
        print(f)

    print("\n\n\nYSI700 -----------------")
    for f in ysi700_functions:
        print(f"\nFor range: {f.r_range[-1]} - {f.r_range[0]}  (R2: {f.r_sq} | POLY_DEGREE: {f.poly_degree}):")
        print(f)


    print("\n  ###   YSI400 Code:")
    codegen(ysi400_functions) 

    print("\n  ###   YSI700 Code:")
    codegen(ysi700_functions)

if __name__ == "__main__":
    main()