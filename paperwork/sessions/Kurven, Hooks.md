Topic: Hook Detection
Bot: Gemini
Date: 26 06 30

> Welche Möglichkeiten gibt es bei einem Linienzug "enge" Kurven zu detektieren (mind. Anfang). Der Linienzug besteht aus n Punkten mit jeweils Zeitstempel, x und y Koordinate. 

Um enge Kurven (inklusive ihres Anfangspunkts) in einem Linienzug aus zeitlich sortierten Punkten $P_i(x_i, y_i, t_i)$ zu detektieren, nutzt man mathematische und geometrische Kriterien. Die effektivsten Methoden basieren auf der Berechnung des Krümmungsradius, der Richtungsänderung (Winkel) oder der orthogonalen Distanz.
Hier sind die besten Ansätze, geordnet nach Komplexität und Zuverlässigkeit:
## 1. Geometrische Krümmung über Menger-Schwamm-Radius
Diese Methode berechnet direkt den Radius des Kreises, der durch drei aufeinanderfolgende Punkte $P_{i-1}$, $P_i$ und $P_{i+1}$ geht.

* Formel: Die Krümmung $\kappa$ (Kehrwert des Radius $R$) berechnet sich aus der Fläche $A$ des Dreiecks und den Seitenlängen $a, b, c$:
$$\kappa = \frac{1}{R} = \frac{4 \cdot A}{a \cdot b \cdot c}$$
wobei die Fläche $A$ über die Kreuzprodukt-Formel ermittelt wird:
$$A = \frac{1}{2} \vert{}x_{i-1}(y_i - y_{i+1}) + x_i(y_{i+1} - y_{i-1}) + x_{i+1}(y_{i-1} - y_i)\vert{}$$
* Detektion: Unterschreitet der Radius $R$ einen definierten Schwellenwert ($R < R_{max}$), liegt eine enge Kurve vor.
* Kurvenanfang: Der Punkt $P_{i-1}$ markiert den Beginn der Richtungsänderung.

## 2. Analyse der Richtungsänderung (Winkeländerung)
Hierbei wird der Winkel zwischen den aufeinanderfolgenden Richtungsvektoren $\vec{u} = P_i - P_{i-1}$ und $\vec{v} = P_{i+1} - P_i$ berechnet.

* Formel: Über das Skalarprodukt wird der eingeschlossene Winkel $\theta$ bestimmt:
$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\Vert{}\vec{u}\Vert{} \cdot \Vert{}\vec{v}\Vert{}}$$
$$\theta = \arccos\left(\frac{(x_i - x_{i-1})(x_{i+1} - x_i) + (y_i - y_{i-1})(y_{i+1} - y_i)}{\sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2} \cdot \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}}\right)$$
* Detektion: Ein großer Winkel $\theta$ (Abweichung von der Geraden) übersteigt einen Schwellenwert $\theta_{min}$.
* Kurvenanfang: Punkt $P_i$ ist der Scheitelpunkt, der Kurvenanfang liegt bei $P_{i-1}$.

## 3. Kinematische Methode (Zentripetalbeschleunigung)
Da Zeitstempel $t_i$ vorhanden sind, kann die physikalische Fliehkraft genutzt werden. Kurven zeichnen sich durch hohe Querbeschleunigungen aus.

* Schritt 1: Berechnung der Geschwindigkeitsvektoren $\vec{v}_i = \frac{P_i - P_{i-1}}{t_i - t_{i-1}}$.
* Schritt 2: Berechnung der Beschleunigungsvektoren $\vec{a}_i = \frac{\vec{v}_{i+1} - \vec{v}_i}{t_{i+1} - t_i}$.
* Schritt 3: Die Normalbeschleunigung (senkrecht zur Bewegungsrichtung) isolieren.
* Detektion: Übersteigt die Zentripetalbeschleunigung einen Grenzwert, detektiert dies dynamisch enge Kurven (abhängig von der Geschwindigkeit).
* Kurvenanfang: Der Punkt $P$, an dem die Beschleunigung sprunghaft ansteigt.

## 4. Douglas-Peucker-Algorithmus (Abweichungsanalyse)
Dieser klassische Algorithmus dient der Linienvereinfachung, kann aber umgekehrt zur Kurvenerkennung genutzt werden.

* Prinzip: Es wird eine gerade Linie zwischen Start- und Endpunkt gezogen und der Punkt mit der maximalen orthogonalen Distanz gesucht.
* Detektion: Liegt dieser Punkt weit abseits der Geraden (Distanz > Schwellenwert), teilt er die Kurve.
* Kurvenanfang: Durch iterative Segmentierung lässt sich der exakte Punkt bestimmen, an dem die Punkte beginnen, signifikant von der vorherigen Luftlinie abzuweichen.

------------------------------
## Empfehlung für die Umsetzung
Wenn die Punkte verrauscht sind (z. B. GPS-Daten), sollten Sie die Daten zuerst mit einem gleitenden Mittelwert oder einem Kalman-Filter glätten. Für die reine Geometrie ist die Winkeländerung (Methode 2) am einfachsten zu implementieren. Wenn die Dynamik (Geschwindigkeit) eine Rolle spielt, ist die Zentripetalbeschleunigung (Methode 3) am präzisesten.
Möchten Sie eine konkrete Code-Implementierung (z. B. in Python) für eine dieser Methoden sehen oder soll die Filterung von Sensorrauschen genauer betrachtet werden?

> Zeig mir eine Python Funktion für Method 2: Der Linienzug liegt als 2D NumPy Array for mit den Spalten 0: Timestamp, 1: x und 2: y.

Hier ist die Python-Funktion, die auf Basis von Methode 2 (Winkeländerung) enge Kurven und deren Anfangspunkte detektiert.
Die Funktion nutzt NumPy-Vektorisierung, um Berechnungen ohne langsame Schleifen extrem schnell durchzuführen. [1, 2]

```python
import numpy as np

def detect_sharp_curves(trajectory, angle_threshold_deg=30.0):
    """
    Detektiert enge Kurven und deren Anfangspunkte in einem Linienzug.

    Parameters:
    -----------
    trajectory : np.ndarray
        2D Array mit Form (N, 3). Spalten: [0: Timestamp, 1: x, 2: y]
    angle_threshold_deg : float
        Schwellenwert für die Richtungsänderung in Grad.
        Werte > Schwellenwert gelten als enge Kurve.

    Returns:
    --------
    curve_starts : np.ndarray
        Boolean-Maske der Länge N. 'True' markiert den Anfangspunkt einer engen Kurve.
    angles_deg : np.ndarray
        Das berechnete Array aller Winkeländerungen (Länge N-2).
    """
    # 1. Extrahiere die X- und Y-Koordinaten
    xy = trajectory[:, 1:3]

    # 2. Berechne die Richtungsvektoren zwischen aufeinanderfolgenden Punkten
    # u_i = P_i - P_{i-1}  für i von 1 bis N-1
    vectors = np.diff(xy, axis=0)

    # 3. Definiere eingehende (u) und ausgehende (v) Vektoren für jeden Kurvenpunkt
    u = vectors[:-1]  # Vektoren von P_{i-1} zu P_i
    v = vectors[1:]   # Vektoren von P_i zu P_{i+1}

    # 4. Berechne das Skalarprodukt (Dot Product) zeilenweise
    dot_products = np.einsum('ij,ij->i', u, v)

    # 5. Berechne die Magnituden (Längen) der Vektoren
    norms_u = np.linalg.norm(u, axis=1)
    norms_v = np.linalg.norm(v, axis=1)

    # Division durch Null verhindern (falls aufeinanderfolgende Punkte identisch sind)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = dot_products / (norms_u * norms_v)
        # Numerische Ungenauigkeiten abfangen (Werte müssen im Bereich [-1, 1] liegen)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Winkel im Bogenmaß berechnen
        angles_rad = np.arccos(cos_theta)

    # In Grad umrechnen
    angles_deg = np.degrees(angles_rad)

    # 6. Schwellenwert abgleichen
    # Ein hoher Winkel bedeutet eine starke Richtungsänderung (enge Kurve)
    is_curve_vertex = angles_deg > angle_threshold_deg

    # 7. Kurvenanfang bestimmen
    # Wenn P_i der Scheitelpunkt (Vertex) ist, fängt die Kurve bei P_{i-1} an.
    # Wir erstellen eine Maske für alle N Punkte der Trajektorie.
    curve_starts = np.zeros(len(trajectory), dtype=bool)

    # Verschiebung um 1 nach links: Wenn an Index i der Winkel groß ist,
    # markieren wir Index i (da vectors[i-1] zu P_i führt und vectors[i] von P_i weg).
    # Somit ist P_{i} der Scheitelpunkt und P_{i-1} der Startpunkt.
    # Da 'is_curve_vertex' bei Index 0 dem Punkt P_1 entspricht, ist der Startpunkt P_0.
    curve_starts[:-2] = is_curve_vertex

    return curve_starts, angles_deg
```

```python
# --- Anwendungsbeispiel ---if __name__ == "__main__":
    # Beispiel-Daten: Eine gerade Strecke, dann eine scharfe 90-Grad-Kurve, dann geradeaus
    data = np.array([
        [10.0,  0.0,  0.0],  # P0: Start gerade (Kurvenanfang!)
        [11.0,  1.0,  0.0],  # P1: Scheitelpunkt der Kurve
        [12.0,  1.0,  1.0],  # P2: Nach der Kurve
        [13.0,  1.0,  2.0],  # P3
    ])

    starts, angles = detect_sharp_curves(data, angle_threshold_deg=45.0)

    print(f"Berechnete Winkeländerungen: {angles}")
    print(f"Maske der Kurvenanfangspunkte: {starts}")
    print(f"Index des Kurvenanfangs: {np.where(starts)[0]}")
  ```

## Funktionsweise im Detail:

   1. np.diff: Erzeugt sofort alle Richtungsvektoren zwischen den Punkten.
   2. [np.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html): Berechnet extrem effizient das Skalarprodukt aller Vektorpaare gleichzeitig, ohne explizite Schleifen.
   3. np.clip: Verhindert, dass durch minimale Rundungsfehler Werte wie 1.0000001 entstehen, was bei np.arccos zu Fehlern (NaN) führen würde.
   4. Verschiebung (curve_starts[:-2]): Wenn die Vektoren um den Punkt $P_i$ knicken, fängt die Richtungsänderung mathematisch bereits im Segment davor an. Der Punkt $P_{i-1}$ wird daher als Anfang markiert.

Möchten Sie wissen, wie man zusammenhängende Kurvensegmente (also wenn eine enge Kurve über mehrere Punkte geht) gruppiert, oder soll das Ergebnis visuell dargestellt werden?

[1] [https://www.computerweekly.com](https://www.computerweekly.com/de/tipp/Datenanalyse-mit-Python-NumPy-pandas-und-Matplotlib)
[2] [https://www.flowhunt.io](https://www.flowhunt.io/de/glossar/numpy/)

> Ja, stelle das Ergebnis visuell dar (mit mathplotlib, wenn möglich.)

Hier ist ein vollständiges Skript zur Visualisierung mit Matplotlib.
Um die Funktion direkt zu testen, generiert das Skript eine synthetische Trajektorie mit zwei abrupten Richtungsänderungen (Haarnadelkurven). Der Linienzug wird gezeichnet, und die detektierten Kurvenanfänge werden farblich hervorgehoben.

import numpy as npimport matplotlib.pyplot as plt
def detect_sharp_curves(trajectory, angle_threshold_deg=30.0):
    """Detektiert enge Kurven und gibt die Startpunkte zurück."""
    xy = trajectory[:, 1:3]
    vectors = np.diff(xy, axis=0)

    u = vectors[:-1]
    v = vectors[1:]

    dot_products = np.einsum('ij,ij->i', u, v)
    norms_u = np.linalg.norm(u, axis=1)
    norms_v = np.linalg.norm(v, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = dot_products / (norms_u * norms_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles_rad = np.arccos(cos_theta)

    angles_deg = np.degrees(angles_rad)
    is_curve_vertex = angles_deg > angle_threshold_deg

    curve_starts = np.zeros(len(trajectory), dtype=bool)
    curve_starts[:-2] = is_curve_vertex

    return curve_starts, angles_deg
# =============================================================================# 1. Beispieldaten generieren (Trajektorie mit zwei engen Kurven)# =============================================================================
np.random.seed(42)n_points = 60t = np.linspace(0, 12, n_points)x = np.zeros(n_points)y = np.zeros(n_points)
# Abschnitt 1: Geradeaus nach rechts
x[0:20] = t[0:20]
y[0:20] = 0.0
# Abschnitt 2: Scharfe 45-Grad-Kurve nach oben links (Kurve 1)
x[20:40] = x[19] + (t[20:40] - t[19]) * np.cos(np.radians(135))
y[20:40] = y[19] + (t[20:40] - t[19]) * np.sin(np.radians(135))
# Abschnitt 3: Scharfer 90-Grad-Knick nach oben rechts (Kurve 2)
x[40:] = x[39] + (t[40:] - t[39]) * np.cos(np.radians(45))
y[40:] = y[39] + (t[40:] - t[39]) * np.sin(np.radians(45))
# Leichtes Rauschen hinzufügen, um reale Bedingungen zu simulieren
x += np.random.normal(0, 0.05, n_points)
y += np.random.normal(0, 0.05, n_points)
# Trajektorie zusammenbauen [Timestamp, X, Y]trajectory_data = np.column_stack((t, x, y))
# =============================================================================# 2. Kurven detektieren# =============================================================================threshold = 25.0  # Schwellenwert in Gradcurve_starts, angles = detect_sharp_curves(trajectory_data, angle_threshold_deg=threshold)
# =============================================================================# 3. Visualisierung mit Matplotlib# =============================================================================fig, ax = plt.subplots(figsize=(10, 6))
# Den gesamten Linienzug als graue Verbindungslinie plotten
ax.plot(x, y, color='gray', linestyle='--', alpha=0.7, label='Linienzug')
# Alle regulären Punkte als kleine blaue Punkte darstellen
ax.scatter(x, y, color='blue', s=20, alpha=0.5, label='Normaler Punkt')
# Detektierte Kurvenanfänge hervorheben (rot und größer)x_starts = x[curve_starts]y_starts = y[curve_starts]
ax.scatter(x_starts, y_starts, color='red', s=100, edgecolor='black',
           zorder=5, label='Detektierter Kurvenanfang')
# Indizes an die Kurvenanfänge schreiben zur besseren Nachvollziehbarkeitstart_indices = np.where(curve_starts)[0]for idx in start_indices:
    ax.annotate(f"Start P{idx}", (x[idx], y[idx]), textcoords="offset points",
                xytext=(0,10), ha='center', fontweight='bold', color='red')
# Diagramm-Styling
ax.set_title(f'Detektion enger Kurven (Winkel-Schwellenwert > {threshold}°)', fontsize=14)
ax.set_xlabel('X Koordinate')
ax.set_ylabel('Y Koordinate')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='best')
ax.set_aspect('equal') # Wichtig, damit Winkel visuell nicht verzerrt werden

plt.show()

## Was zeigt die Grafik?

* Blaue Punkte & graue Linie: Die gemessene Fahrtstrecke inklusive minimalem Sensorrauschen.
* Rote Punkte (Start PXX): Die exakten Punkte, an denen der Linienzug die enge Kurve beginnt. Das bedeutet, dass der Richtungsvektor zum nächsten Punkt bereits den kritischen Knick vollzieht.
* set_aspect('equal'): Diese Zeile sorgt dafür, dass die X- und Y-Achse den gleichen Maßstab haben. Ohne diesen Befehl würden Kurven je nach Fenstergröße gestaucht oder gestreckt wirken.

Möchten Sie zusätzlich eine Echtzeit-Filterung einbauen, falls Ihre realen Daten stärker verrauscht sind, oder reicht diese geometrische Lösung aus?

```python
import numpy as np
def detect_sharp_curves(trajectory, angle_threshold_deg=30.0):
    xy = trajectory[:, 1:3]
    vectors = np.diff(xy, axis=0)
    u = vectors[:-1]
    v = vectors[1:]
    dot_products = np.einsum('ij,ij->i', u, v)
    norms_u = np.linalg.norm(u, axis=1)
    norms_v = np.linalg.norm(v, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = dot_products / (norms_u * norms_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles_rad = np.arccos(cos_theta)

    angles_deg = np.degrees(angles_rad)
    is_curve_vertex = angles_deg > angle_threshold_deg

    curve_starts = np.zeros(len(trajectory), dtype=bool)
    curve_starts[:-2] = is_curve_vertex
    return curve_starts, angles_deg

# Test with a generated sharp curve path to verify correctnesst = np.linspace(0, 10, 50)x = np.zeros_like(t)y = np.zeros_like(t)
# Straight line, sharp turn, straight line
x[:20] = t[:20]
y[:20] = 0

x[20:35] = t[19] + (t[20:35] - t[19]) * np.cos(np.pi/4)
y[20:35] = (t[20:35] - t[19]) * np.sin(np.pi/4)
# Next sharp turn
x[35:] = x[34]
y[35:] = y[34] + (t[35:] - t[34])
trajectory = np.column_stack((t, x, y))starts, angles = detect_sharp_curves(trajectory, angle_threshold_deg=20.0)
print("Detected starts indices:", np.where(starts)[0])
```
