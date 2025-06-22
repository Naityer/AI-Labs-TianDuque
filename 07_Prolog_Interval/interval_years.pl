% interval_2024.pl
% Variante 2: calcular número de días entre dos fechas en el año 2024

% Días en cada mes (considerando 2024 como año bisiesto)
days_in_month(1, 31).  % Enero
days_in_month(2, 29).  % Febrero (bisiesto)
days_in_month(3, 31).
days_in_month(4, 30).
days_in_month(5, 31).
days_in_month(6, 30).
days_in_month(7, 31).
days_in_month(8, 31).
days_in_month(9, 30).
days_in_month(10, 31).
days_in_month(11, 30).
days_in_month(12, 31).

% Convertir una fecha "DDMM" a número de días desde el 1 de enero
date_to_day_of_year(DateStr, DayOfYear) :-
    string_chars(DateStr, [D1,D2,M1,M2]),
    number_chars(Day, [D1,D2]),
    number_chars(Month, [M1,M2]),
    sum_days_before_month(Month, DaysBefore),
    DayOfYear is DaysBefore + Day.

% Sumar los días de todos los meses anteriores
sum_days_before_month(1, 0).
sum_days_before_month(M, TotalDays) :-
    M > 1,
    M1 is M - 1,
    sum_days(1, M1, TotalDays).

sum_days(Min, Max, 0) :- Min > Max.
sum_days(Min, Max, Total) :-
    Min =< Max,
    days_in_month(Min, D),
    Next is Min + 1,
    sum_days(Next, Max, Rest),
    Total is D + Rest.

% Calcular el intervalo entre dos fechas
interval(Date1, Date2, Days) :-
    date_to_day_of_year(Date1, D1),
    date_to_day_of_year(Date2, D2),
    Days is abs(D2 - D1).
