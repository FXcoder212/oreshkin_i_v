# Telescope observation planning

Решение: `submission_telescope.csv`.

1. `models.py`: модель вероятности успеха (LightGBM без погоды, т.к. в тесте её нет; AUC 0.915 по CV) и модель изменения температуры за шаг.
2. `sim.py`: таблицы p(T) и ΔT(T) для каждого шага/объекта на сетке температур, Монте-Карло открытого плана с учётом принудительного отдыха; тест обнуляется, если отдыхов (REST+REST_FORCED) больше 45.
3. `opt.py` + `make_submission.py`: жадный старт и покоординатный подъём по ожидаемому баллу.

Запуск: `python3 prep.py && python3 opt.py && python3 make_submission.py`.
