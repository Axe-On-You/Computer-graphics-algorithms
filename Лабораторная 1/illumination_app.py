import math
import os
import sys
import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
from PIL import Image

# В обычном оконном запуске используется TkAgg; при автоматической проверке
# в headless-среде Matplotlib может безопасно переключиться на Agg.
if os.environ.get("DISPLAY") or sys.platform.startswith(("win", "darwin")):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# -----------------------------------------------------------------------------
# Ограничения из задания
# -----------------------------------------------------------------------------
PARAM_LIMITS = {
    "W": (100.0, 10000.0),
    "H": (100.0, 10000.0),
    "Wres": (200, 800),
    "xL": (-10000.0, 10000.0),
    "yL": (-10000.0, 10000.0),
    "zL": (100.0, 10000.0),
    "I0": (0.01, 10000.0),
}


class IlluminationModel:
    """Расчет энергетической освещенности от точечного ламбертовского источника."""

    def __init__(self, W, H, Wres, xL, yL, zL, I0, cx, cy, R):
        self.W = float(W)
        self.H = float(H)
        self.Wres = int(Wres)
        self.xL = float(xL)
        self.yL = float(yL)
        self.zL = float(zL)
        self.I0 = float(I0)
        self.cx = float(cx)
        self.cy = float(cy)
        self.R = float(R)

        # Задаем один общий шаг по X и Y -> пиксели квадратные.
        self.px = self.W / self.Wres
        self.Hres = round(self.H / self.px)
        if not (200 <= self.Hres <= 800):
            raise ValueError(
                "При выбранных W, H и Wres автоматический Hres выходит "
                "за допустимый диапазон 200..800. Измените размеры области "
                "или разрешение по ширине."
            )

        self.H_effective = self.Hres * self.px

        # Координаты центров квадратных пикселей.
        xs = (np.arange(self.Wres) + 0.5) * self.px - self.W / 2.0
        ys = (np.arange(self.Hres) + 0.5) * self.px - self.H_effective / 2.0
        self.X, self.Y = np.meshgrid(xs, ys)

        self.mask = (self.X - self.cx) ** 2 + (self.Y - self.cy) ** 2 <= self.R ** 2
        if not np.any(self.mask):
            raise ValueError("Внутри круга не оказалось ни одного центра пикселя.")

        self.E = None
        self.E_max = None
        self.E_min = None
        self.E_mean = None
        self.image_u8 = None

    # ------------------------------------------------------------------
    # Математическая модель
    # ------------------------------------------------------------------
    def illuminance(self, x, y):
        """
        Энергетическая освещенность на плоскости z=0.

        Для ламбертовского источника:
            I(theta) = I0 * cos(theta)

        Для плоскости:
            E = I(theta) * cos(theta) / r^2
              = I0 * cos(theta)^2 / r^2
              = I0 * zL^2 / r^4.
        """
        dx = self.xL - x
        dy = self.yL - y
        r2 = dx * dx + dy * dy + self.zL * self.zL

        # zL > 0 по ограничениям задания, поэтому r2 > 0.
        return self.I0 * (self.zL ** 2) / (r2 ** 2)

    def compute(self):
        """Вычисляет распределение, статистики и 8-битное изображение."""
        E_full = self.illuminance(self.X, self.Y)
        self.E = np.where(self.mask, E_full, 0.0)

        inside = self.E[self.mask]
        self.E_max = float(np.max(inside))
        self.E_min = float(np.min(inside))
        self.E_mean = float(np.mean(inside))

        # По заданию: нормировка на максимум внутри круга и перевод в 0..255.
        normalized = np.zeros_like(self.E, dtype=np.float64)
        normalized[self.mask] = self.E[self.mask] / self.E_max * 255.0
        normalized = np.clip(normalized, 0.0, 255.0)
        self.image_u8 = np.rint(normalized).astype(np.uint8)

        return self.image_u8

    # ------------------------------------------------------------------
    # Контрольные точки
    # ------------------------------------------------------------------
    def _axis_intersections(self):
        """Находит реальные пересечения окружности с глобальными осями X и Y."""
        result = []
        eps = 1e-12

        # Ось X: y = 0.
        disc_x = self.R ** 2 - self.cy ** 2
        if disc_x >= -eps:
            d = math.sqrt(max(0.0, disc_x))
            x1 = self.cx - d
            x2 = self.cx + d
            result.append(("Ось X, точка 1", x1, 0.0))
            if abs(x2 - x1) > eps:
                result.append(("Ось X, точка 2", x2, 0.0))

        # Ось Y: x = 0.
        disc_y = self.R ** 2 - self.cx ** 2
        if disc_y >= -eps:
            d = math.sqrt(max(0.0, disc_y))
            y1 = self.cy - d
            y2 = self.cy + d
            result.append(("Ось Y, точка 1", 0.0, y1))
            if abs(y2 - y1) > eps:
                result.append(("Ось Y, точка 2", 0.0, y2))

        return result

    def control_points(self):
        """
        Возвращает контрольные значения.

        Для варианта cx=cy=0 получаются ровно пять требуемых геометрических
        точек: центр, два пересечения с X и два пересечения с Y.
        Для смещенного круга возвращаются реальные пересечения с глобальными
        осями; если пересечения нет, оно не существует геометрически.
        """
        result = OrderedDict()
        result["Центр круга"] = ((self.cx, self.cy), float(self.illuminance(self.cx, self.cy)))

        for name, x, y in self._axis_intersections():
            result[name] = ((x, y), float(self.illuminance(x, y)))

        result["Максимум по сетке"] = (None, self.E_max)
        result["Минимум по сетке"] = (None, self.E_min)
        result["Среднее по сетке"] = (None, self.E_mean)
        return result

    # ------------------------------------------------------------------
    # Точные сечения через центр круга
    # ------------------------------------------------------------------
    def cross_section(self, axis="X", samples=1200):
        """
        Точное математическое сечение через центр круга.

        X: y = cy, x in [cx-R, cx+R]
        Y: x = cx, y in [cy-R, cy+R]

        Здесь не выбирается "ближайшая строка пикселей": функция E(x, y)
        вычисляется непосредственно на требуемой линии.
        """
        if axis not in ("X", "Y"):
            raise ValueError("Ось должна быть X или Y.")

        coord = np.linspace(-self.R, self.R, samples)
        if axis == "X":
            x = self.cx + coord
            y = np.full_like(coord, self.cy)
            values = self.illuminance(x, y)
        else:
            x = np.full_like(coord, self.cx)
            y = self.cy + coord
            values = self.illuminance(x, y)

        return coord, values


class App(tk.Tk):
    """Графическое приложение лабораторной работы №1."""

    BG = "#f4f6f8"
    CARD = "#ffffff"
    TEXT = "#20252b"
    MUTED = "#65707b"
    ACCENT = "#2563eb"

    def __init__(self):
        super().__init__()
        self.title("АКГ — Лабораторная работа №1")
        self.geometry("1320x840")
        self.minsize(1120, 720)
        self.configure(bg=self.BG)

        self.model = None
        self.vars = {}
        self.axis_var = tk.StringVar(value="X")
        self.status_var = tk.StringVar(value="Введите параметры и нажмите «Рассчитать».")
        self.grid_info_var = tk.StringVar(value="Сетка: —")

        self._configure_style()
        self._build_ui()
        self._set_defaults()

    # ------------------------------------------------------------------
    # Стиль
    # ------------------------------------------------------------------
    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TFrame",
            background=self.BG,
        )
        style.configure(
            "Card.TFrame",
            background=self.CARD,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Field.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Hint.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
        )
        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 9),
            padding=(10, 6),
        )
        style.configure(
            "TRadiobutton",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Segoe UI", 9),
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))

        ttk.Label(
            header,
            text="Лабораторная работа №1 — Расчет освещенности",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Точечный источник света · ламбертовская диаграмма · плоскость z = 0",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="Card.TFrame", padding=14)
        right = ttk.Frame(body, style="Card.TFrame", padding=10)
        body.add(left, weight=0)
        body.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

        status = ttk.Frame(outer)
        status.pack(fill="x", pady=(10, 0))
        ttk.Label(status, textvariable=self.status_var, style="Subtitle.TLabel").pack(side="left")
        ttk.Label(status, textvariable=self.grid_info_var, style="Subtitle.TLabel").pack(side="right")

    def _build_left_panel(self, parent):
        ttk.Label(parent, text="Параметры сцены", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Размеры — в мм; I0 — в Вт/ср.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(1, 10))

        params_frame = ttk.Frame(parent, style="Card.TFrame")
        params_frame.pack(fill="x")

        fields = [
            ("W", "Ширина области, мм", 3000),
            ("H", "Высота области, мм", 2000),
            ("Wres", "Разрешение по ширине, px", 600),
            ("xL", "X источника, мм", 500),
            ("yL", "Y источника, мм", 300),
            ("zL", "Z источника, мм", 1200),
            ("I0", "Сила излучения, Вт/ср", 2000),
        ]

        for row, (key, label, default) in enumerate(fields):
            self._add_entry(params_frame, row, key, label, default)

        ttk.Separator(parent).pack(fill="x", pady=12)

        ttk.Label(parent, text="Расчетный круг", style="CardTitle.TLabel").pack(anchor="w")
        circle_frame = ttk.Frame(parent, style="Card.TFrame")
        circle_frame.pack(fill="x", pady=(8, 0))

        circle_fields = [
            ("cx", "Центр X, мм", 0),
            ("cy", "Центр Y, мм", 0),
            ("R", "Радиус, мм", 900),
        ]
        for row, (key, label, default) in enumerate(circle_fields):
            self._add_entry(circle_frame, row, key, label, default)

        ttk.Label(
            parent,
            textvariable=self.grid_info_var,
            style="Hint.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

        ttk.Separator(parent).pack(fill="x", pady=12)

        ttk.Label(parent, text="Сечение", style="CardTitle.TLabel").pack(anchor="w")
        axis_frame = ttk.Frame(parent, style="Card.TFrame")
        axis_frame.pack(fill="x", pady=(5, 0))
        ttk.Radiobutton(
            axis_frame,
            text="X: y = cy",
            variable=self.axis_var,
            value="X",
            command=self._redraw_section,
        ).pack(side="left", padx=(0, 14))
        ttk.Radiobutton(
            axis_frame,
            text="Y: x = cx",
            variable=self.axis_var,
            value="Y",
            command=self._redraw_section,
        ).pack(side="left")

        buttons = ttk.Frame(parent, style="Card.TFrame")
        buttons.pack(fill="x", pady=(14, 0))

        ttk.Button(
            buttons,
            text="Рассчитать",
            style="Primary.TButton",
            command=self.calculate,
        ).pack(fill="x", pady=3)
        ttk.Button(
            buttons,
            text="Восстановить пример",
            style="Secondary.TButton",
            command=self._set_defaults,
        ).pack(fill="x", pady=3)
        ttk.Button(
            buttons,
            text="Сохранить карту PNG",
            style="Secondary.TButton",
            command=self.save_image,
        ).pack(fill="x", pady=3)
        ttk.Button(
            buttons,
            text="Сохранить сечение PNG",
            style="Secondary.TButton",
            command=self.save_section,
        ).pack(fill="x", pady=3)

    def _add_entry(self, parent, row, key, label, default):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=4,
        )
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(parent, textvariable=var, width=14, justify="right")
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        self.vars[key] = var
        parent.columnconfigure(1, weight=1)

    def _build_right_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)
        parent.rowconfigure(2, weight=2)

        self.fig = Figure(figsize=(8.5, 8), dpi=100)
        self.ax_img = self.fig.add_subplot(211)
        self.ax_section = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.35, left=0.09, right=0.96, top=0.96, bottom=0.08)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, sticky="nsew")

        toolbar_frame = ttk.Frame(parent, style="Card.TFrame")
        toolbar_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left")

        results = ttk.LabelFrame(parent, text="Контрольные значения", padding=6)
        results.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(8, 0))
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            results,
            columns=("point", "coord", "value"),
            show="headings",
            height=18,
        )
        self.tree.heading("point", text="Точка")
        self.tree.heading("coord", text="Координаты")
        self.tree.heading("value", text="E, Вт/мм²")
        self.tree.column("point", width=170, anchor="w")
        self.tree.column("coord", width=150, anchor="center")
        self.tree.column("value", width=145, anchor="e")
        self.tree.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(results, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

        self._clear_plots()

    # ------------------------------------------------------------------
    # Данные / валидация
    # ------------------------------------------------------------------
    def _set_defaults(self):
        defaults = {
            "W": 3000,
            "H": 2000,
            "Wres": 600,
            "xL": 500,
            "yL": 300,
            "zL": 1200,
            "I0": 2000,
            "cx": 0,
            "cy": 0,
            "R": 900,
        }
        for key, value in defaults.items():
            self.vars[key].set(str(value))
        self.status_var.set("Параметры восстановлены. Нажмите «Рассчитать».")

    def _read_params(self):
        try:
            values = {key: float(var.get().replace(",", ".")) for key, var in self.vars.items()}
        except ValueError as exc:
            raise ValueError("Все параметры должны быть числами.") from exc

        for key, (lo, hi) in PARAM_LIMITS.items():
            if not (lo <= values[key] <= hi):
                raise ValueError(
                    f"Параметр {key} должен находиться в диапазоне [{lo:g}, {hi:g}]."
                )

        if not math.isclose(values["Wres"], round(values["Wres"]), abs_tol=1e-9):
            raise ValueError("Wres должно быть целым числом пикселей.")
        values["Wres"] = round(values["Wres"])

        if values["R"] <= 0:
            raise ValueError("Радиус R должен быть положительным.")

        W = values["W"]
        H = values["H"]
        cx = values["cx"]
        cy = values["cy"]
        R = values["R"]

        if abs(cx) + R > W / 2.0 or abs(cy) + R > H / 2.0:
            raise ValueError(
                "Круг должен целиком находиться внутри прямоугольной области W×H."
            )

        # Проверяем результирующее Hres до построения модели.
        px = W / values["Wres"]
        Hres = round(H / px)
        if not (200 <= Hres <= 800):
            raise ValueError(
                f"Для выбранных W={W:g}, H={H:g}, Wres={values['Wres']} "
                f"получается Hres={Hres}. Требуется 200..800. "
                "Подберите совместимые W, H и Wres."
            )

        return values

    # ------------------------------------------------------------------
    # Расчет / визуализация
    # ------------------------------------------------------------------
    def calculate(self):
        try:
            values = self._read_params()
            self.model = IlluminationModel(**values)
            self.model.compute()
        except ValueError as exc:
            self.model = None
            messagebox.showerror("Ошибка ввода", str(exc))
            self.status_var.set("Ошибка: проверьте входные параметры.")
            return
        except Exception as exc:  # noqa: BLE001
            self.model = None
            messagebox.showerror("Ошибка", f"Не удалось выполнить расчет:\n{exc}")
            self.status_var.set("Внутренняя ошибка расчета.")
            return

        self.grid_info_var.set(
            f"Сетка: {self.model.Wres}×{self.model.Hres} px · "
            f"шаг = {self.model.px:.6g} мм · Hэфф = {self.model.H_effective:.6g} мм"
        )
        self._draw_image()
        self._redraw_section()
        self._show_control_points()

        self.status_var.set(
            "Расчет выполнен. Формула: E = I0·zL² / r⁴. "
            "Максимум/минимум/среднее — по центрам пикселей внутри круга."
        )

    def _clear_plots(self):
        self.ax_img.clear()
        self.ax_img.set_title("Распределение освещенности")
        self.ax_img.set_xlabel("X, мм")
        self.ax_img.set_ylabel("Y, мм")
        self.ax_img.text(
            0.5,
            0.5,
            "После расчета здесь появится карта",
            transform=self.ax_img.transAxes,
            ha="center",
            va="center",
        )

        self.ax_section.clear()
        self.ax_section.set_title("Сечение через центр круга")
        self.ax_section.set_xlabel("Смещение от центра, мм")
        self.ax_section.set_ylabel("E, Вт/мм²")
        self.ax_section.text(
            0.5,
            0.5,
            "Выберите X или Y",
            transform=self.ax_section.transAxes,
            ha="center",
            va="center",
        )
        self.canvas.draw_idle()

    def _draw_image(self):
        m = self.model
        self.ax_img.clear()

        extent = [
            -m.W / 2.0,
            m.W / 2.0,
            -m.H_effective / 2.0,
            m.H_effective / 2.0,
        ]
        self.ax_img.imshow(
            m.image_u8,
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=255,
            aspect="equal",
        )
        circle = Circle(
            (m.cx, m.cy),
            m.R,
            fill=False,
            linewidth=1.5,
            edgecolor="tab:red",
        )
        self.ax_img.add_patch(circle)

        # Проекция источника на плоскость — полезная визуальная подсказка.
        if -m.W / 2 <= m.xL <= m.W / 2 and -m.H_effective / 2 <= m.yL <= m.H_effective / 2:
            self.ax_img.plot(m.xL, m.yL, marker="+", markersize=10, markeredgewidth=1.5)

        self.ax_img.set_title("Нормированное распределение освещенности (0–255)")
        self.ax_img.set_xlabel("X, мм")
        self.ax_img.set_ylabel("Y, мм")
        self.ax_img.grid(alpha=0.15)

        self.canvas.draw_idle()

    def _redraw_section(self):
        if self.model is None:
            self.ax_section.clear()
            self.ax_section.set_title("Сечение через центр круга")
            self.ax_section.set_xlabel("Смещение от центра, мм")
            self.ax_section.set_ylabel("E, Вт/мм²")
            self.canvas.draw_idle()
            return

        axis = self.axis_var.get()
        coord, values = self.model.cross_section(axis=axis)

        self.ax_section.clear()
        self.ax_section.plot(coord, values, linewidth=1.7)
        self.ax_section.axvline(0.0, linewidth=0.8, alpha=0.5)
        self.ax_section.grid(True, alpha=0.25)
        self.ax_section.set_title(
            "Сечение через центр круга: "
            + ("y = cy" if axis == "X" else "x = cx")
        )
        self.ax_section.set_xlabel("Смещение от центра, мм")
        self.ax_section.set_ylabel("E, Вт/мм²")
        self.canvas.draw_idle()

    def _show_control_points(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for name, (coord, value) in self.model.control_points().items():
            if coord is None:
                coord_text = "—"
            else:
                coord_text = f"({coord[0]:.6g}; {coord[1]:.6g})"
            self.tree.insert(
                "",
                "end",
                values=(name, coord_text, f"{value:.8e}"),
            )

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    def save_image(self):
        if self.model is None or self.model.image_u8 is None:
            messagebox.showwarning("Нет результата", "Сначала выполните расчет.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить карту освещенности",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return

        try:
            Image.fromarray(self.model.image_u8).save(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка сохранения", str(exc))
            return

        messagebox.showinfo("Готово", f"Изображение сохранено:\n{path}")

    def save_section(self):
        if self.model is None:
            messagebox.showwarning("Нет результата", "Сначала выполните расчет.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить график сечения",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return

        axis = self.axis_var.get()
        coord, values = self.model.cross_section(axis=axis)

        figure = Figure(figsize=(8, 4.8), dpi=130)
        ax = figure.add_subplot(111)
        ax.plot(coord, values, linewidth=1.8)
        ax.axvline(0.0, linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            "Сечение распределения освещенности через центр круга: "
            + ("X (y = cy)" if axis == "X" else "Y (x = cx)")
        )
        ax.set_xlabel("Смещение от центра, мм")
        ax.set_ylabel("E, Вт/мм²")
        figure.tight_layout()

        try:
            figure.savefig(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка сохранения", str(exc))
            return
        finally:
            # Явно закрываем отдельный Figure, чтобы не накапливать объекты Matplotlib.
            import matplotlib.pyplot as plt
            plt.close(figure)

        messagebox.showinfo("Готово", f"График сечения сохранен:\n{path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
