from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import tkinter as tk
import numpy as np
from PIL import Image
from AKG_2 import DEFAULTS, EXAMPLES, App, SphereBrightnessModel


def model(changes=None):
    m = SphereBrightnessModel(DEFAULTS | (changes or {}))
    m.compute()
    assert np.isfinite(m.L).all() and (m.L >= 0).all()
    return m


def verify():
    checks = []
    base = model()
    assert np.isclose(base.L_max, 2.116825051444181e-4, rtol=1e-11)
    checks.append('Исходный расчет совпадает с отчетом')

    for name, changes in EXAMPLES.items():
        m = model(changes)
        assert len(m.sample_points()) == 3
        if m.L_max > 0:
            assert m.image_u8.max() == 255
    checks.append('Пять примеров дают конечные значения и три разные точки')

    under = model(EXAMPLES['Под наблюдателем'])
    P = np.array([[0., 0., 2100.]])
    N = np.array([[0., 0., 1.]])
    d = np.array([[0., 0., -1.]])
    expected = 3000 * (1 / 900**2 + 1 / 1400**2) * 1.6 / np.pi
    assert np.isclose(under._shade_point(P, N, d)[0], expected, rtol=1e-12)
    checks.append('Источники под наблюдателем совпадают с аналитическим расчетом полюса')

    at_eye = model(dict(xL1=0, yL1=0, zL1=4000, xL2=0, yL2=0, zL2=4000,
                       xT1=0, yT1=0, zT1=1500, xT2=0, yT2=0, zT2=1500))
    expected = 4500 / 1900**2 * 1.6 / np.pi
    assert np.isclose(at_eye._shade_point(P, N, d)[0], expected, rtol=1e-12)
    checks.append('Источник может совпадать с точкой наблюдения')

    below = model(EXAMPLES['Под сферой'])
    assert not below.L.any() and not below.image_u8.any()
    bottom = np.array([0., 0., 900.])
    to_light = below.lights[0]['pos'] - bottom
    assert np.dot([0, 0, -1], to_light / np.linalg.norm(to_light)) > 0
    checks.append('Лампы снизу освещают нижнюю сторону, но видимая сверху поверхность темная')

    side = model(EXAMPLES['Симметрично по бокам'])
    assert side.L_max > 0
    assert np.allclose(side.L, np.fliplr(side.L), atol=1e-14)
    assert np.allclose(side.L, np.flipud(side.L), atol=1e-14)
    checks.append('Сфера и боковые лампы при z=0 дают симметричное освещение')

    away = model(EXAMPLES['Направлены от сферы'])
    assert not away.L.any()
    checks.append('При развороте обеих ламп от сферы освещение исчезает')

    one_away = model(EXAMPLES['Под наблюдателем'] | dict(zT1=5000))
    expected = 3000 / 1400**2 * 1.6 / np.pi
    assert np.isclose(one_away._shade_point(P, N, d)[0], expected, rtol=1e-12)
    checks.append('Поворот первой лампы не меняет вклад второй')

    inside = model(dict(xL1=0, yL1=0, zL1=1400, xL2=0, yL2=0, zL2=1600,
                        xT1=0, yT1=0, zT1=1500, xT2=0, yT2=0, zT2=1500))
    on_surface = model(dict(xL1=0, yL1=0, zL1=2100, xL2=0, yL2=0, zL2=900,
                            xT1=0, yT1=0, zT1=1500, xT2=0, yT2=0, zT2=1500))
    assert not inside.L.any() and not on_surface.L.any()
    checks.append('Лампы внутри и на непрозрачной сфере разрешены, внешнего освещения от них нет')

    far = model(dict(xL1=-20000, yL1=0, zL1=-20000, xL2=20000, yL2=0, zL2=20000,
                     xT1=0, yT1=0, zT1=1500, xT2=0, yT2=0, zT2=1500))
    assert far.L_max > 0
    checks.append('Координаты ламп за прежними границами ±10000 разрешены')

    doubled = model(dict(I01=6000, I02=3000))
    assert np.allclose(doubled.L, 2 * base.L, rtol=1e-12, atol=0)
    assert np.array_equal(doubled.image_u8, base.image_u8)
    checks.append('Удвоение сил ламп удваивает яркость и сохраняет нормированное изображение')

    assert np.allclose(np.linalg.norm(base.P[base.hit_mask] - base.C, axis=1), base.R)
    assert np.allclose(np.linalg.norm(base.N[base.hit_mask], axis=1), 1)
    moved = model(dict(W=3000, H=3000, xC=250, yC=250))
    for m in (base, moved):
        points = m.sample_points()
        assert len({(p['row'], p['col']) for p in points}) == 3
        for p in points:
            assert np.allclose(m.project_to_screen(p['point']),
                               [m.X[p['row'], p['col']], m.Y[p['row'], p['col']]], atol=1e-9)
    checks.append('Точки лежат на сфере, нормали единичные, маркеры совпадают с проекциями')

    with TemporaryDirectory() as folder:
        path = Path(folder) / 'image.png'
        base.save_image(path)
        assert np.array_equal(np.array(Image.open(path)), np.flipud(base.image_u8))
    checks.append('PNG ориентирован как изображение на экране')

    for changes in (dict(xT1=-1500, yT1=-1000, zT1=3000), dict(xL1=float('nan')),
                    dict(Wres=400.5), dict(H=100), dict(zO=2000)):
        try:
            model(changes)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Некорректные параметры приняты: {changes}')
    checks.append('Ошибочные числа, нулевое направление и неверная геометрия отклоняются')

    interpreter = tk.Tcl()
    app = SimpleNamespace(vars={k: tk.StringVar(master=interpreter, value='invalid') for k in DEFAULTS},
                          example=tk.StringVar(master=interpreter, value='Под сферой'), calls=0)
    app.set_params = lambda values: App.set_params(app, values)
    def calculate():
        app.calls += 1
        SphereBrightnessModel({k: var.get() for k, var in app.vars.items()})
    app.calculate = calculate
    App.reset_params(app)
    assert all(float(app.vars[k].get()) == v for k, v in DEFAULTS.items()) and app.calls == 1
    app.example.set('Под сферой')
    App.apply_example(app)
    assert float(app.vars['zL2'].get()) == -1500 and app.calls == 2
    checks.append('Сброс и выбор примера обновляют все поля и запускают расчет')
    return checks


if __name__ == '__main__':
    for i, check in enumerate(verify(), 1):
        print(f'{i:02d}. OK: {check}')
