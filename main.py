
import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
import math


def polyfit_approximation(X, Y, degree, x):
    n = len(X)
    if n != len(Y):
        raise ValueError("Długość zbiorów X i Y musi być taka sama.")

    A = []
    for i in range(degree + 1):
        A.append([xi ** i for xi in X])

    A = np.array(A)
    B = np.array(Y)

    # Rozwiązanie układu równań liniowych A * coeffs = B
    coeffs = np.linalg.solve(A.dot(A.T), A.dot(B))

    result = 0.0
    for i in range(degree + 1):
        result += coeffs[i] * (x ** i)

    return result

def pokaz_wykres(miasto,co_pokazac,nazwa_miasta,co_pokazac_nazwa):
    pokazac = miasto[co_pokazac].tolist()
    data = miasto['date'].tolist()
    pyplot.plot(data,  pokazac)
    pyplot.xticks(range(0,150, 40))
    pyplot.xlabel('Oś  {}'.format('data'))
    pyplot.ylabel('Oś  {}'.format(co_pokazac_nazwa))
    pyplot.title('Wykres miasta {}: {} od {} '.format(nazwa_miasta,co_pokazac_nazwa,'data'))
    pyplot.show()

def Lagrange_interpolation(X,Y,xp,a,b):
    if b > len(X):
        b = len(X)
    yp = 0
    for i in range(a,b):
        if(pd.isna(Y[i])):
            k = Y[i+1]
        else:
            k = Y[i]
        for j in range(a,b):
            if j != i:
                k = k * (xp - X[j]) / (X[i] - X[j])
        yp += k


    return yp

def popraw_dane(miasto,co_poprawic):
    X = miasto['day_of_year'].tolist()
    Y = miasto[co_poprawic].tolist()



    for i in range(0,len(X)):

        if math.isnan(Y[i]) or Y[i] == 'nan' or pd.isna(Y[i]):

            Y[i] = Lagrange_interpolation(X,Y,i,i-3,i+3)

    miasto[co_poprawic] = Y
    return miasto

def aproksymuj_i_pokaz(miasto,co_aprox,tytul,os):
    X = miasto['day_of_year'].tolist()
    Y = miasto[co_aprox].tolist()
    degree = 2
    coeffs = np.polyfit(X, Y, degree)
    poly = np.poly1d(coeffs)
    pyplot.xticks(range(0, 150, 40))
    # Wygenerowanie punktów dla wykresu
    x_fit = np.linspace(np.min(X), np.max(X), 100)
    y_fit = poly(x_fit)

    # Wykres danych i funkcji aproksymującej
    pyplot.scatter(miasto['date'].tolist(), Y, label='Dane')
    pyplot.plot(x_fit, y_fit, label='Aproksymacja')
    pyplot.xlabel('X')
    pyplot.ylabel(os)
    pyplot.title('Aproksymacja funkcji {} dla {}'.format(os,tytul))
    pyplot.legend()
    pyplot.grid(True)
    pyplot.show()

if __name__ == '__main__':
    deblin = pd.read_excel("Deblin.xlsx", engine="openpyxl")
    warszawa = pd.read_excel("warszawa-okecie.xlsx", engine="openpyxl")
    siedlce = pd.read_excel("Siedlce.xlsx", engine="openpyxl")

# ----------------------- Deblin
    poprawione_deblin = popraw_dane(deblin,'tavg')
    poprawione_deblin = popraw_dane(poprawione_deblin,'wspd')
    poprawione_deblin = popraw_dane(poprawione_deblin, 'pres')
    print(poprawione_deblin)
    #  pokaz_wykres(deblin,'tavg','Deblin','temperatura')
    #  pokaz_wykres(deblin, 'wspd', 'Deblin', 'predkosc wiatru')
    #  pokaz_wykres(deblin, 'pres', 'Deblin', 'cisnienie')

    #  pokaz_wykres(poprawione_deblin, 'tavg', 'Deblin', 'temperatura')
    #  pokaz_wykres(poprawione_deblin, 'wspd', 'Deblin', 'predkosc wiatru')
    #  pokaz_wykres(poprawione_deblin, 'pres', 'Deblin', 'cisnienie')

# ----------------------- Warszawa
#  pokaz_wykres(warszawa,'tavg','Warszawa','temperatura')
#  pokaz_wykres(warszawa, 'wspd', 'Warszawa', 'predkosc wiatru')
#  pokaz_wykres(warszawa, 'pres', 'Warszawa', 'cisnienie')

    poprawione_warszawa = popraw_dane(warszawa, 'tavg')
    poprawione_warszawa = popraw_dane(poprawione_warszawa, 'wspd')
    poprawione_warszawa = popraw_dane(poprawione_warszawa, 'pres')

#    pokaz_wykres(poprawione_warszawa, 'tavg', 'Warszawa', 'temperatura')
#    pokaz_wykres(poprawione_warszawa, 'wspd', 'Warszawa', 'predkosc wiatru')
#    pokaz_wykres(poprawione_warszawa, 'pres', 'Warszawa', 'cisnienie')

# ----------------------- Siedlce
#pokaz_wykres(siedlce,'tavg','Siedlce','temperatura')
#pokaz_wykres(siedlce, 'wspd', 'Siedlce', 'predkosc wiatru')
#pokaz_wykres(siedlce, 'pres', 'Siedlce', 'cisnienie')

poprawione_siedlce = popraw_dane(siedlce, 'tavg')
poprawione_siedlce = popraw_dane(poprawione_siedlce, 'wspd')
poprawione_siedlce = popraw_dane(poprawione_siedlce, 'pres')

#pokaz_wykres(poprawione_siedlce, 'tavg', 'Siedlce', 'temperatura')
#pokaz_wykres(poprawione_siedlce, 'wspd', 'Siedlce', 'predkosc wiatru')
#pokaz_wykres(poprawione_siedlce, 'pres', 'Siedlce', 'cisnienie')

# ----------------------- Aproksymacja dalej

#aproksymuj_i_pokaz(deblin,'tavg','Deblin','temperatura')
#aproksymuj_i_pokaz(deblin,'wspd','Deblin','predkosc wiatru')
#aproksymuj_i_pokaz(deblin,'pres','Deblin','cisnienie')

#aproksymuj_i_pokaz(poprawione_warszawa,'tavg','Warszawa','temperatura')
#aproksymuj_i_pokaz(poprawione_warszawa,'wspd','Warszawa','predkosc wiatru')
#aproksymuj_i_pokaz(poprawione_warszawa,'pres','Warszawa','cisnienie')

#aproksymuj_i_pokaz(poprawione_siedlce,'tavg','Siedlce','temperatura')
#aproksymuj_i_pokaz(poprawione_siedlce,'wspd','Siedlce','predkosc wiatru')
#aproksymuj_i_pokaz(poprawione_siedlce,'pres','Siedlce','cisnienie')
