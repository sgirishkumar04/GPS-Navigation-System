# GPS Navigation System 🗺️

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Dijkstra's Algorithm](https://img.shields.io/badge/Algorithm-Dijkstra's-orange?style=for-the-badge&logo=D)](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
[![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-11557C?style=for-the-badge&logo=matplotlib)](https://matplotlib.org/)

A desktop application that implements **Dijkstra's algorithm** to calculate and visualize the shortest path between any two locations on a map. This project provides an interactive interface to select start and end points and displays the optimal route on a graphical map.

---

## ✨ Key Features

-   **Interactive Map Interface:** A clean, graphical interface allows users to select start and end nodes directly on a visual map representation.
-   **Shortest Path Calculation:** Implements the classic **Dijkstra's algorithm** to efficiently find the most optimal route based on distance or cost.
-   **Dynamic Route Visualization:** Clearly highlights the nodes and edges that form the calculated shortest path, providing immediate visual feedback.
-   **Real-time Route Information:** Instantly displays the total distance or cost of the calculated path upon selection.

---

## 🛠️ Technology Stack & Architecture

This project is a standalone desktop application built with a focus on core algorithms and visualization.

-   **Graphical User Interface (GUI):** The user interface is built with a standard Python GUI library like **Tkinter** or **PyQt** for user interaction (e.g., buttons, input fields).
-   **Map & Graph Visualization:** The map, nodes, and path are rendered using **Matplotlib** or a similar plotting library integrated into the GUI.
-   **Core Logic & Algorithms:** The pathfinding logic is written in pure **Python**, implementing Dijkstra's algorithm. The graph data structure is managed using dictionaries or a library like **NetworkX**.

---

## 🚀 How to Run Locally

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Python 3.x
-   Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/YOUR_USERNAME/GPS-Navigation-System.git
    cd GPS-Navigation-System
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```sh
    python main.py 
    ```
    *(Note: Replace `main.py` with the name of your main script if it's different.)*

---

## 💡 Challenges & Learnings

Building this project involved overcoming several classic computer science challenges, which were fantastic learning opportunities:

-   **Implementing Graph Algorithms:** The core challenge was implementing Dijkstra's algorithm from scratch, requiring a deep understanding of its data structures, such as priority queues for efficient node selection.

-   **GUI and Logic Integration:** A key learning experience was integrating the Matplotlib visualization canvas into a Tkinter/PyQt GUI window. This involved managing the application state and handling user click events on the canvas to select map nodes.

-   **Data Representation:** Designing and implementing a flexible data structure to represent the map as a graph—including nodes (locations), edges (roads), and their associated weights (distances)—was fundamental to the project's success.
