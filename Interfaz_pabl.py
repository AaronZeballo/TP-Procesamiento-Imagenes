import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
from PIL import ImageTk, Image
import numpy as np
import os
import matplotlib.pyplot as plt
import funciones  # Módulo de funciones propio

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------------------
# Utilidades: conversión entre PIL y NumPy (funciones.py usa matrices NumPy)
# ---------------------------------------------------------------------------

def _numpy_a_pil(matriz):
    """Convierte una matriz NumPy (uint8, 2D o 3D) a imagen PIL."""
    if matriz.ndim == 2:
        return Image.fromarray(matriz, mode="L")
    else:
        return Image.fromarray(matriz, mode="RGB")

def _pil_a_numpy(img_pil):
    """Convierte una imagen PIL a matriz NumPy uint8."""
    return np.array(img_pil)


# ---------------------------------------------------------------------------
# Ventana de Selección de Región (TP0)
# ---------------------------------------------------------------------------

class VentanaSeleccion(ctk.CTkToplevel):
    """Ventana de selección 1:1 con scrollbars y precisión (TP0)."""
    def __init__(self, master, matriz_numpy):
        super().__init__(master)
        self.title("Selección de Región — TP0")
        self.matriz = matriz_numpy
        pil_image = _numpy_a_pil(matriz_numpy)
        self.pil_image = pil_image
        w, h = pil_image.size
        self.geometry(f"{min(w+60, 1000)}x{min(h+160, 800)}")

        inner = ctk.CTkFrame(self, fg_color="#1a1a1a")
        inner.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = Canvas(inner, bg="#111111", highlightthickness=0, cursor="crosshair")
        vbar = ctk.CTkScrollbar(inner, orientation="vertical", command=self.canvas.yview)
        hbar = ctk.CTkScrollbar(inner, orientation="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        hbar.pack(side="bottom", fill="x")
        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.lbl_info = ctk.CTkLabel(self, text="Arrastrá el mouse para medir la región.",
                                     font=("Consolas", 14), text_color="#cc6a00")
        self.lbl_info.pack(pady=10)

        self._photo = ImageTk.PhotoImage(self.pil_image)
        self.canvas.config(scrollregion=(0, 0, w, h))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self._start = None
        self._rect_id = None
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _coords(self, event):
        return int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))

    def _on_press(self, event):
        self._start = self._coords(event)
        if self._rect_id:
            self.canvas.delete(self._rect_id)

    def _on_drag(self, event):
        if not self._start:
            return
        x0, y0 = self._start
        x1, y1 = self._coords(event)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2, dash=(4, 4))

    def _on_release(self, event):
        if not self._start:
            return
        x0, y0 = self._start
        x1, y1 = self._coords(event)
        lx, rx = sorted([x0, x1])
        ty, by = sorted([y0, y1])
        iw, ih = self.pil_image.size
        lx, rx = max(0, lx), min(iw, rx)
        ty, by = max(0, ty), min(ih, by)

        if rx <= lx or by <= ty:
            self.lbl_info.configure(text="Región inválida.")
            return

        try:
            # funciones.obtener_estadisticas_region(matriz, x1, y1, x2, y2)
            n, prom = funciones.obtener_estadisticas_region(self.matriz, lx, ty, rx, by)
            if isinstance(prom, np.ndarray):
                prom_str = f"RGB({prom[0]}, {prom[1]}, {prom[2]})"
            else:
                prom_str = f"{prom}"
            self.lbl_info.configure(
                text=f"Área: ({lx},{ty}) a ({rx},{by}) | Pxs: {n} | Promedio: {prom_str}"
            )
        except Exception as e:
            self.lbl_info.configure(text=f"Error: {e}")


# ---------------------------------------------------------------------------
# Aplicación principal
# ---------------------------------------------------------------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Procesamiento de Imágenes — UNAHUR")
        self.geometry("1250x850")

        # Las imágenes se guardan como matrices NumPy
        self._mat_a: np.ndarray | None = None
        self._mat_b: np.ndarray | None = None

        # --- TOOLBAR SUPERIOR ---
        self.toolbar = ctk.CTkFrame(self, height=65)
        self.toolbar.pack(fill="x", side="top", padx=10, pady=10)

        ctk.CTkButton(self.toolbar, text="Abrir Imagen 1", command=self._load_a, width=120).pack(side="left", padx=5)
        ctk.CTkButton(self.toolbar, text="Abrir Imagen 2", command=self._load_b, width=120).pack(side="left", padx=5)
        ctk.CTkButton(self.toolbar, text="Guardar Imagen 1", command=self._guardar_a, width=130).pack(side="left", padx=5)

        # Menú TP0
        self.menu_tp0 = ctk.CTkOptionMenu(
            self.toolbar,
            values=["TP0: Opciones", "Restar A-B", "Obtener Píxel", "Modificar Píxel", "Seleccionar Región"],
            command=self._handle_tp0, width=160
        )
        self.menu_tp0.pack(side="left", padx=10)
        self.menu_tp0.set("TP0")

        # Menú TP1
        self.menu_tp1 = ctk.CTkOptionMenu(
            self.toolbar,
            values=[
                "TP1: Opciones", "Función Gamma", "Negativo", "Histograma", "Ecualización",
                "Umbralización", "Ruido Gaussiano (Aditivo)", "Ruido Exponencial (Multiplicativo)",
                "Ruido Sal y Pimienta", "Filtro de Media", "Filtro de Mediana",
                "Filtro de Mediana Ponderada", "Filtro Gaussiano", "Filtro Realce de Bordes"
            ],
            command=self._handle_tp1, width=200
        )
        self.menu_tp1.pack(side="left", padx=10)
        self.menu_tp1.set("TP1")

        # --- PANELES DE IMAGEN ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.panel_a = ctk.CTkLabel(self.main_frame, text="Imagen A (Vacío)", bg_color="#1a1a1a", corner_radius=8)
        self.panel_a.pack(side="left", fill="both", expand=True, padx=5)

        self.panel_b = ctk.CTkLabel(self.main_frame, text="Imagen B (Vacío)", bg_color="#1a1a1a", corner_radius=8)
        self.panel_b.pack(side="right", fill="both", expand=True, padx=5)

        # Barra de estado
        self.status = ctk.StringVar(value="Listo.")
        ctk.CTkLabel(self, textvariable=self.status, anchor="w", fg_color="#111111", padx=10).pack(fill="x", side="bottom")

    # -----------------------------------------------------------------------
    # Helpers de visualización
    # -----------------------------------------------------------------------

    def _actualizar_paneles(self):
        if self._mat_a is not None:
            self._mostrar_en_panel(self.panel_a, self._mat_a)
        if self._mat_b is not None:
            self._mostrar_en_panel(self.panel_b, self._mat_b)

    def _mostrar_en_panel(self, panel, matriz):
        img_pil = _numpy_a_pil(matriz)
        img_pil.thumbnail((550, 550))
        ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
        panel.configure(image=ctk_img, text="")
        panel.image = ctk_img  # Evita que el GC elimine la referencia

    def _pedir_entero(self, prompt, titulo=""):
        val = ctk.CTkInputDialog(text=prompt, title=titulo).get_input()
        if val is None:
            return None
        return int(val.strip())

    def _pedir_float(self, prompt, titulo=""):
        val = ctk.CTkInputDialog(text=prompt, title=titulo).get_input()
        if val is None:
            return None
        return float(val.replace(',', '.').strip())

    # -----------------------------------------------------------------------
    # Carga y guardado
    # -----------------------------------------------------------------------

    def _load_a(self):
        p = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.raw *.jpg *.jpeg *.png *.bmp *.tiff"), ("RAW", "*.raw"), ("Todos", "*.*")]
        )
        if not p:
            return
        try:
            ext = os.path.splitext(p)[1].upper()
            if ext == ".RAW":
                w = self._pedir_entero("Ancho (px):", "RAW – Imagen A")
                h = self._pedir_entero("Alto (px):", "RAW – Imagen A")
                if w is None or h is None:
                    return
                self._mat_a = funciones.cargar_imagen(p, w, h)
            else:
                self._mat_a = funciones.cargar_imagen(p)
            self._actualizar_paneles()
            self.status.set(f"Imagen A cargada: {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def _load_b(self):
        p = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.raw *.jpg *.jpeg *.png *.bmp *.tiff"), ("RAW", "*.raw"), ("Todos", "*.*")]
        )
        if not p:
            return
        try:
            ext = os.path.splitext(p)[1].upper()
            if ext == ".RAW":
                w = self._pedir_entero("Ancho (px):", "RAW – Imagen B")
                h = self._pedir_entero("Alto (px):", "RAW – Imagen B")
                if w is None or h is None:
                    return
                self._mat_b = funciones.cargar_imagen(p, w, h)
            else:
                self._mat_b = funciones.cargar_imagen(p)
            self._actualizar_paneles()
            self.status.set(f"Imagen B cargada: {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def _guardar_a(self):
        if self._mat_a is None:
            messagebox.showwarning("Guardar", "No hay imagen A para guardar.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("RAW", "*.raw"), ("Todos", "*.*")]
        )
        if not p:
            return
        try:
            funciones.guardar_imagen(self._mat_a, p)
            self.status.set(f"Imagen A guardada en: {os.path.basename(p)}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

    # -----------------------------------------------------------------------
    # TP0
    # -----------------------------------------------------------------------

    def _handle_tp0(self, choice):
        try:
            if choice == "Restar A-B":
                if self._mat_a is None or self._mat_b is None:
                    messagebox.showwarning("Restar", "Se necesitan ambas imágenes (A y B).")
                    return
                resultado = funciones.restar_imagenes(self._mat_a, self._mat_b)
                self._mat_b = resultado
                self._actualizar_paneles()
                self.status.set("Resta A-B aplicada → Imagen B.")

            elif choice == "Obtener Píxel":
                if self._mat_a is None:
                    return
                x = self._pedir_entero("Coordenada X:", "Obtener Píxel")
                y = self._pedir_entero("Coordenada Y:", "Obtener Píxel")
                if x is None or y is None:
                    return
                val = funciones.obtener_pixel(self._mat_a, x, y)
                messagebox.showinfo("Valor del Píxel", f"Píxel en ({x}, {y}): {val}")

            elif choice == "Modificar Píxel":
                if self._mat_a is None:
                    return
                x = self._pedir_entero("Coordenada X:", "Modificar Píxel")
                y = self._pedir_entero("Coordenada Y:", "Modificar Píxel")
                if x is None or y is None:
                    return
                # Soporte para imágenes grises (un valor) y RGB (tres valores)
                if self._mat_a.ndim == 2:
                    v = self._pedir_entero("Nuevo valor (0-255):", "Modificar Píxel")
                    if v is None:
                        return
                    nuevo_valor = np.uint8(np.clip(v, 0, 255))
                else:
                    r = self._pedir_entero("Canal R (0-255):", "Modificar Píxel")
                    g = self._pedir_entero("Canal G (0-255):", "Modificar Píxel")
                    b = self._pedir_entero("Canal B (0-255):", "Modificar Píxel")
                    if any(c is None for c in [r, g, b]):
                        return
                    nuevo_valor = np.array([r, g, b], dtype=np.uint8)
                self._mat_a = funciones.modificar_pixel(self._mat_a, x, y, nuevo_valor)
                self._actualizar_paneles()
                self.status.set(f"Píxel ({x},{y}) modificado.")

            elif choice == "Seleccionar Región":
                if self._mat_a is None:
                    return
                VentanaSeleccion(self, self._mat_a)

        except Exception as e:
            messagebox.showerror("Error TP0", str(e))
        finally:
            self.menu_tp0.set("TP0")

    # -----------------------------------------------------------------------
    # TP1
    # -----------------------------------------------------------------------

    def _handle_tp1(self, choice):
        if self._mat_a is None:
            messagebox.showwarning("TP1", "Primero cargá la Imagen A.")
            self.menu_tp1.set("TP1")
            return

        try:
            if choice == "Función Gamma":
                gamma = self._pedir_float("Gamma (ej: 0.5 o 1.5):", "Función Gamma")
                if gamma is None:
                    return
                self._mat_a = funciones.aplicar_transformacion_potencia(self._mat_a, gamma)
                self.status.set(f"Función Gamma (γ={gamma}) aplicada.")

            elif choice == "Negativo":
                self._mat_a = funciones.obtener_negativo(self._mat_a)
                self.status.set("Negativo aplicado.")

            elif choice == "Histograma":
                hist = funciones.obtener_histograma_gris(self._mat_a)
                plt.figure("Histograma – TP1")
                plt.title("Frecuencia de Niveles de Gris")
                plt.bar(range(256), hist, color="steelblue", width=1.0)
                plt.xlim([0, 256])
                plt.xlabel("Nivel de gris")
                plt.ylabel("Frecuencia")
                plt.tight_layout()
                plt.show()
                return  # No actualiza paneles

            elif choice == "Ecualización":
                self._mat_a = funciones.ecualizar_histograma(self._mat_a)
                self.status.set("Ecualización de histograma aplicada.")

            elif choice == "Umbralización":
                umbral = self._pedir_entero("Umbral (0-255):", "Umbralización")
                if umbral is None:
                    return
                self._mat_a = funciones.obtener_umbralizacion(self._mat_a, umbral)
                self.status.set(f"Umbralización (umbral={umbral}) aplicada.")

            elif choice == "Ruido Gaussiano (Aditivo)":
                densidad = self._pedir_float("Densidad de píxeles afectados (0-100):", "Ruido Gaussiano")
                if densidad is None:
                    return
                sigma = self._pedir_float("Desviación estándar (sigma):", "Ruido Gaussiano")
                if sigma is None:
                    return
                self._mat_a = funciones.aplicar_ruido_aditivo_gaussiano(self._mat_a, densidad, sigma)
                self.status.set(f"Ruido Gaussiano aditivo aplicado (densidad={densidad}%, σ={sigma}).")

            elif choice == "Ruido Exponencial (Multiplicativo)":
                porcentaje = self._pedir_float("Porcentaje de píxeles afectados (0-100):", "Ruido Exponencial")
                if porcentaje is None:
                    return
                lambd = self._pedir_float("Parámetro lambda (> 0):", "Ruido Exponencial")
                if lambd is None:
                    return
                self._mat_a = funciones.aplicar_ruido_multiplicativo_exponencial(self._mat_a, porcentaje, lambd)
                self.status.set(f"Ruido Exponencial multiplicativo aplicado (p={porcentaje}%, λ={lambd}).")

            elif choice == "Ruido Sal y Pimienta":
                densidad = self._pedir_float("Densidad total de contaminación (0-100):", "Sal y Pimienta")
                if densidad is None:
                    return
                self._mat_a = funciones.aplicar_ruido_sal_y_pimienta(self._mat_a, densidad)
                self.status.set(f"Ruido Sal y Pimienta aplicado (densidad={densidad}%).")

            elif choice == "Filtro de Media":
                tamano = self._pedir_entero("Tamaño de máscara (impar, ej: 3, 5, 7):", "Filtro de Media")
                if tamano is None:
                    return
                self._mat_a = funciones.aplicar_filtro_media(self._mat_a, tamano)
                self.status.set(f"Filtro de Media ({tamano}×{tamano}) aplicado.")

            elif choice == "Filtro de Mediana":
                tamano = self._pedir_entero("Tamaño de máscara (impar, ej: 3, 5, 7):", "Filtro de Mediana")
                if tamano is None:
                    return
                self._mat_a = funciones.aplicar_filtro_mediana(self._mat_a, tamano)
                self.status.set(f"Filtro de Mediana ({tamano}×{tamano}) aplicado.")

            elif choice == "Filtro de Mediana Ponderada":
                tamano = self._pedir_entero("Tamaño de máscara (impar, ej: 3, 5, 7):", "Filtro Mediana Ponderada")
                if tamano is None:
                    return
                self._mat_a = funciones.aplicar_filtro_mediana_ponderada(self._mat_a, tamano)
                self.status.set(f"Filtro de Mediana Ponderada ({tamano}×{tamano}) aplicado.")

            elif choice == "Filtro Gaussiano":
                sigma = self._pedir_float("Valor de Sigma (ej: 1, 2, 3):", "Filtro Gaussiano")
                if sigma is None:
                    return
                self._mat_a = funciones.aplicar_filtro_gaussiano(self._mat_a, sigma)
                self.status.set(f"Filtro Gaussiano (σ={sigma}) aplicado.")

            elif choice == "Filtro Realce de Bordes":
                tamano = self._pedir_entero("Tamaño de máscara (impar, ej: 3, 5, 7):", "Filtro Realce de Bordes")
                if tamano is None:
                    return
                self._mat_a = funciones.aplicar_filtro_realce_de_bordes(self._mat_a, tamano)
                self.status.set(f"Filtro Realce de Bordes ({tamano}×{tamano}) aplicado.")

            self._actualizar_paneles()

        except ValueError as e:
            messagebox.showerror("Valor inválido", str(e))
        except Exception as e:
            messagebox.showerror("Error TP1", str(e))
        finally:
            self.menu_tp1.set("TP1")


if __name__ == "__main__":
    App().mainloop()
