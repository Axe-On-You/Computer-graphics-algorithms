import argparse
import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


DEFAULTS = dict(W=2000, H=2000, Wres=400, zO=4000,
                xC=0, yC=0, zC=1500, R=600,
                xL1=-1500, yL1=-1000, zL1=3000, I01=3000,
                xL2=1500, yL2=1000, zL2=3000, I02=1500,
                xT1=-1500, yT1=-1000, zT1=2000,
                xT2=1500, yT2=1000, zT2=2000,
                kd=0.7, ks=0.9, ke=40)
EPS = 1e-9
EXAMPLES = {
    'Исходная сцена': {},
    'Под наблюдателем': dict(xL1=0, yL1=0, zL1=3000, xL2=0, yL2=0, zL2=3500,
                            xT1=0, yT1=0, zT1=1500, xT2=0, yT2=0, zT2=1500, I02=3000),
    'Под сферой': dict(xL1=0, yL1=0, zL1=0, xL2=0, yL2=0, zL2=-1500,
                      xT1=0, yT1=0, zT1=1500, xT2=0, yT2=0, zT2=1500, I02=3000),
    'Симметрично по бокам': dict(zC=0, xL1=-1500, yL1=0, zL1=0,
                               xL2=1500, yL2=0, zL2=0, I02=3000,
                               xT1=0, yT1=0, zT1=0, xT2=0, yT2=0, zT2=0),
    'Направлены от сферы': dict(zT1=4000, zT2=4000),
}


def validate_params(values):
    v = {key: float(values[key]) for key in DEFAULTS}
    if not all(np.isfinite(x) for x in v.values()):
        raise ValueError('Все параметры должны быть конечными числами.')
    limits = {'W': (100, 10000), 'H': (100, 10000), 'Wres': (200, 800),
              'xC': (-10000, 10000), 'yC': (-10000, 10000), 'zC': (-10000, 10000)}
    for i in (1, 2):
        limits[f'I0{i}'] = (0.01, 10000)
    for key, (lo, hi) in limits.items():
        if not lo <= v[key] <= hi:
            raise ValueError(f'{key}: допустимый диапазон [{lo}, {hi}].')
    if not v['Wres'].is_integer():
        raise ValueError('Wres должно быть целым числом.')
    v['Wres'] = int(v['Wres'])
    px = v['W'] / v['Wres']
    hres = round(v['H'] / px)
    if not 200 <= hres <= 800:
        raise ValueError(f'Hres = {hres}; требуется 200..800. Измените W, H или Wres.')
    if v['R'] <= EPS or v['zO'] <= 0:
        raise ValueError('R должно быть больше 1e-9 мм, zO должно быть положительным.')
    if v['kd'] < 0 or v['ks'] < 0 or v['ke'] <= 0:
        raise ValueError('Требуется kd ≥ 0, ks ≥ 0, ke > 0.')
    depth = v['zO'] - v['zC']
    if depth <= v['R']:
        raise ValueError('Наблюдатель должен быть выше всей сферы: zO > zC + R.')
    # Расстояния центра сферы до четырех боковых плоскостей поля зрения.
    for size, offset in ((v['W'], v['xC']), (hres * px, v['yC'])):
        slope = size / (2 * v['zO'])
        margin = (slope * depth - abs(offset)) / np.hypot(1, slope)
        if margin < v['R']:
            raise ValueError('Сфера не помещается в кадр. Увеличьте экран или уменьшите R/смещение.')
    for i in (1, 2):
        light = np.array([v[f'xL{i}'], v[f'yL{i}'], v[f'zL{i}']])
        target = np.array([v[f'xT{i}'], v[f'yT{i}'], v[f'zT{i}']])
        length = np.linalg.norm(target - light)
        if not np.isfinite(length):
            raise ValueError(f'Слишком большие координаты источника {i} или точки прицеливания.')
        if length <= EPS:
            raise ValueError(f'Точка прицеливания источника {i} должна отличаться от его положения.')
    return v


class SphereBrightnessModel:
    def __init__(self, values):
        self.params = v = validate_params(values)
        self.W, self.H = v['W'], v['H']
        self.Wres = v['Wres']
        self.px = self.W / self.Wres
        self.Hres = round(self.H / self.px)
        self.H_eff = self.Hres * self.px
        self.eye = np.array([0.0, 0.0, v['zO']])
        self.C = np.array([v['xC'], v['yC'], v['zC']])
        self.R = v['R']
        self.kd, self.ks, self.ke = v['kd'], v['ks'], v['ke']
        self.lights = []
        for i in (1, 2):
            pos = np.array([v[f'xL{i}'], v[f'yL{i}'], v[f'zL{i}']])
            target = np.array([v[f'xT{i}'], v[f'yT{i}'], v[f'zT{i}']])
            axis = target - pos
            axis /= np.linalg.norm(axis)
            self.lights.append({'pos': pos, 'I0': v[f'I0{i}'], 'axis': axis})
        xs = (np.arange(self.Wres) + 0.5) * self.px - self.W / 2
        ys = (np.arange(self.Hres) + 0.5) * self.px - self.H_eff / 2
        self.X, self.Y = np.meshgrid(xs, ys)
        self.S = np.stack([self.X, self.Y, np.zeros_like(self.X)], axis=-1)

    def _intersect_sphere(self):
        d = self.S - self.eye
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        oc = self.eye - self.C
        b = 2 * np.sum(d * oc, axis=-1)
        c = np.dot(oc, oc) - self.R ** 2
        disc = b * b - 4 * c
        t = (-b - np.sqrt(np.maximum(disc, 0))) / 2
        mask = (disc >= 0) & (t > EPS)
        return mask, self.eye + t[..., None] * d, d

    def _shade_point(self, P, N, d):
        v = -d
        total = np.zeros(P.shape[:-1])
        for light in self.lights:
            # Непрозрачная сфера не пропускает свет изнутри на внешнюю сторону.
            if np.linalg.norm(light['pos'] - self.C) <= self.R:
                continue
            s = P - light['pos']
            r = np.maximum(np.linalg.norm(s, axis=-1), EPS)
            l = -s / r[..., None]
            cos_theta = np.clip(np.sum(s * light['axis'], axis=-1) / r, 0, 1)
            cos_sigma = np.clip(np.sum(N * l, axis=-1), 0, 1)
            E = light['I0'] * cos_theta * cos_sigma / r ** 2
            h = v + l
            h /= np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), EPS)
            spec = np.clip(np.sum(N * h, axis=-1), 0, 1) ** self.ke
            total += E * (self.kd + self.ks * spec) / np.pi
        return total

    def compute(self):
        self.hit_mask, self.P, d = self._intersect_sphere()
        self.N = np.zeros_like(self.P)
        self.N[self.hit_mask] = (self.P[self.hit_mask] - self.C) / self.R
        self.L = np.zeros(self.hit_mask.shape)
        self.L[self.hit_mask] = self._shade_point(
            self.P[self.hit_mask], self.N[self.hit_mask], d[self.hit_mask])
        if not np.isfinite(self.L).all():
            raise ValueError('Слишком большие значения параметров: расчет переполнен.')
        visible = self.L[self.hit_mask]
        self.L_max = float(visible.max()) if visible.size else 0.0
        self.L_min = float(visible.min()) if visible.size else 0.0
        normalized = self.L / self.L_max if self.L_max > 0 else np.zeros_like(self.L)
        self.image_u8 = np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        return self.image_u8

    def sample_points(self):
        rows, cols = np.where(self.hit_mask)
        if not len(rows):
            return []
        xc = self.eye[2] * self.C[0] / (self.eye[2] - self.C[2])
        yc = self.eye[2] * self.C[1] / (self.eye[2] - self.C[2])
        xs, ys = self.X[rows, cols], self.Y[rows, cols]
        targets = [(xc, yc), ((xs.min() + xc) / 2, yc), ((xs.max() + xc) / 2, yc)]
        used = np.zeros(len(rows), dtype=bool)
        result = []
        for name, (x, y) in zip(('A', 'B', 'C'), targets):
            if used.all():
                break
            dist = np.where(used, np.inf, (xs - x) ** 2 + (ys - y) ** 2)
            idx = int(np.argmin(dist))
            used[idx] = True
            row, col = int(rows[idx]), int(cols[idx])
            result.append({'name': name, 'row': row, 'col': col,
                           'point': self.P[row, col], 'L': float(self.L[row, col])})
        return result

    def project_to_screen(self, point):
        return self.eye[2] * np.asarray(point)[:2] / (self.eye[2] - point[2])

    def save_image(self, path):
        # В массиве y растет вверх, в файле первая строка расположена сверху.
        Image.fromarray(np.flipud(self.image_u8)).save(path)

    def save_values(self, path):
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['point', 'row_from_bottom', 'col_from_left', 'x_mm', 'y_mm',
                             'z_mm', 'L_W_per_sr_mm2'])
            for p in self.sample_points():
                writer.writerow([p['name'], p['row'], p['col'], *p['point'], p['L']])
            writer.writerow(['max_visible_grid', '', '', '', '', '', self.L_max])
            writer.writerow(['min_visible_grid', '', '', '', '', '', self.L_min])


class App(tk.Tk):
    def __init__(self, initial=None):
        super().__init__()
        self.title('ЛР 2 — Яркость сферы — Михайлов, Гришин — АКГ 1.2')
        self.geometry('1200x780')
        self.minsize(980, 700)
        self.model = None
        self.vars = {}
        self._build_ui()
        if initial is not None:
            self.set_params(initial)
            if initial != DEFAULTS:
                self.example.set('Сцена из файла')
        self.calculate()

    def _build_ui(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side='left', fill='y')
        right = ttk.Frame(self, padding=8)
        right.pack(side='left', fill='both', expand=True)
        ttk.Label(left, text='Примеры расположения света').pack(anchor='w')
        self.example = tk.StringVar(value='Исходная сцена')
        selector = ttk.Combobox(left, textvariable=self.example, values=list(EXAMPLES), state='readonly')
        selector.pack(fill='x', pady=(2, 6))
        selector.bind('<<ComboboxSelected>>', self.apply_example)
        tabs = ttk.Notebook(left)
        tabs.pack(fill='x')
        groups = [
            ('Геометрия', [('W', 'Ширина экрана, мм'), ('H', 'Высота экрана, мм'),
                          ('Wres', 'Ширина, пикселей'), ('zO', 'z наблюдателя, мм'),
                          ('xC', 'x центра сферы, мм'), ('yC', 'y центра сферы, мм'),
                          ('zC', 'z центра сферы, мм'), ('R', 'Радиус, мм')]),
            ('Источник 1', [(f'{axis}L1', f'{axis} источника, мм') for axis in 'xyz'] +
                           [('I01', 'I₀, Вт/ср')] +
                           [(f'{axis}T1', f'{axis} точки прицеливания, мм') for axis in 'xyz']),
            ('Источник 2', [(f'{axis}L2', f'{axis} источника, мм') for axis in 'xyz'] +
                           [('I02', 'I₀, Вт/ср')] +
                           [(f'{axis}T2', f'{axis} точки прицеливания, мм') for axis in 'xyz']),
            ('Материал', [('kd', 'kd — диффузный вес'), ('ks', 'ks — зеркальный вес'),
                          ('ke', 'ke — резкость блика')])]
        for title, fields in groups:
            frame = ttk.Frame(tabs, padding=8)
            tabs.add(frame, text=title)
            for row, (key, label) in enumerate(fields):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', pady=4)
                self.vars[key] = tk.StringVar(value=str(DEFAULTS[key]))
                ttk.Entry(frame, textvariable=self.vars[key], width=10).grid(row=row, column=1)
            if title in ('Источник 1', 'Источник 2'):
                i = 1 if title == 'Источник 1' else 2
                ttk.Label(frame, text='Лампа светит от своего положения\nк точке прицеливания.').grid(
                    row=len(fields), column=0, columnspan=2, sticky='w', pady=8)
                ttk.Button(frame, text='Направить в центр сферы',
                           command=lambda i=i: self.aim_at_center(i)).grid(
                    row=len(fields) + 1, column=0, columnspan=2, sticky='ew')
        for label, command in [('Рассчитать', self.calculate),
                               ('Исходные параметры', self.reset_params),
                               ('Сохранить PNG', self.save_image),
                               ('Сохранить значения CSV', self.save_values)]:
            ttk.Button(left, text=label, command=command).pack(fill='x', pady=4)
        self.info = ttk.Label(left, text='', wraplength=300)
        self.info.pack(fill='x', pady=6)
        ttk.Label(left, text='P — точка на сфере (x, y, z), мм\nS — ее проекция на экран (x, y), мм\nЯркость L, Вт/(ср·мм²)').pack(anchor='w')
        result_frame = ttk.Frame(left)
        result_frame.pack(fill='both', expand=True)
        self.result_text = tk.Text(result_frame, width=38, height=14, state='disabled', font=('TkFixedFont', 10))
        self.result_text.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.result_text.config(yscrollcommand=scrollbar.set)
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    def set_params(self, values):
        for key, value in values.items():
            self.vars[key].set(str(value))

    def reset_params(self):
        self.example.set('Исходная сцена')
        self.set_params(DEFAULTS)
        self.calculate()

    def apply_example(self, event=None):
        self.set_params(DEFAULTS | EXAMPLES[self.example.get()])
        self.calculate()

    def aim_at_center(self, i):
        for axis in 'xyz':
            self.vars[f'{axis}T{i}'].set(self.vars[f'{axis}C'].get())
        self.calculate()

    def calculate(self):
        try:
            values = {k: float(var.get().replace(',', '.')) for k, var in self.vars.items()}
            model = SphereBrightnessModel(values)
            model.compute()
        except (ValueError, OverflowError) as e:
            messagebox.showerror('Ошибка параметров', str(e))
            return
        self.model = m = model
        self.info.config(text=f'{m.Wres} × {m.Hres} пикселей; шаг {m.px:g} мм\n'
                              f'Высота сетки: {m.H_eff:g} мм (задано {m.H:g})')
        self.ax.clear()
        self.ax.imshow(m.image_u8, cmap='gray', origin='lower', interpolation='nearest',
                       extent=[-m.W/2, m.W/2, -m.H_eff/2, m.H_eff/2], vmin=0, vmax=255)
        self.ax.set(xlabel='X проекции на экран z = 0, мм', ylabel='Y проекции на экран z = 0, мм',
                    title='Нормированная яркость 0–255 · маркеры — проекции точек')
        lines = []
        for p in m.sample_points():
            x, y, z = p['point']
            sx, sy = m.project_to_screen(p['point'])
            lines.append(f"{p['name']}: P = ({x:.2f}; {y:.2f}; {z:.2f})\n"
                         f"   S = ({sx:.2f}; {sy:.2f})\nL = {p['L']:.6e}")
            self.ax.plot(sx, sy, '+', color='orange', markersize=7)
            self.ax.annotate(p['name'], (sx, sy), xytext=(6, 6), textcoords='offset points', color='orange')
        lines.extend([f'Max по видимой сетке: {m.L_max:.6e}', f'Min по видимой сетке: {m.L_min:.6e}'])
        if not m.hit_mask.any():
            lines = ['Сфера меньше шага сетки: лучи прошли мимо.\nУвеличьте R или разрешение.']
        elif len(m.sample_points()) < 3:
            lines.append('Меньше трех видимых пикселей.\nУвеличьте R или разрешение.')
        elif m.L_max == 0:
            lines.append('Видимая поверхность не освещена.')
        for i, light in enumerate(m.lights, 1):
            if np.linalg.norm(light['pos'] - m.C) <= m.R:
                lines.append(f'Источник {i} внутри/на сфере:\nвнешняя поверхность от него не освещена.')
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('end', '\n\n'.join(lines))
        self.result_text.config(state='disabled')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _save(self, extension, label, action):
        if self.model is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=extension, filetypes=[(label, '*' + extension)])
        if path:
            try:
                action(path)
            except OSError as e:
                messagebox.showerror('Ошибка сохранения', str(e))

    def save_image(self):
        self._save('.png', 'PNG', self.model.save_image)

    def save_values(self):
        self._save('.csv', 'CSV', self.model.save_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ЛР 2. Расчет яркости сферы')
    parser.add_argument('--scene', metavar='JSON', help='загрузить параметры сцены из JSON')
    parser.add_argument('--export', metavar='DIR', help='сохранить расчет без открытия окна')
    args = parser.parse_args()
    values = DEFAULTS.copy()
    if args.scene:
        try:
            scene = json.loads(Path(args.scene).read_text(encoding='utf-8'))
            if not isinstance(scene, dict) or set(scene) - DEFAULTS.keys():
                raise ValueError('JSON должен содержать объект с известными параметрами сцены.')
            values = validate_params(DEFAULTS | scene)
        except (OSError, ValueError, TypeError) as e:
            parser.error(str(e))
    if args.export:
        folder = Path(args.export)
        folder.mkdir(parents=True, exist_ok=True)
        model = SphereBrightnessModel(values)
        model.compute()
        model.save_image(folder / 'result_sphere.png')
        model.save_values(folder / 'control_points.csv')
        (folder / 'scene.json').write_text(json.dumps(model.params, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Результат сохранен: {folder.resolve()}')
    else:
        App(values).mainloop()
