# Proyecto Capstone Backend

Este proyecto es un backend desarrollado en Python. Utiliza un entorno virtual (`venv`) y gestiona las dependencias con [uv](https://github.com/astral-sh/uv).

## Requisitos previos

- Python 3.8 o superior
- [uv](https://github.com/astral-sh/uv) instalado globalmente

## Instalación

1. **Clona el repositorio:**
    ```bash
    git clone https://github.com/SebastianG03/proyecto-capstone-backend.git
    cd proyecto-capstone-backend
    ```

2. **Crea el entorno virtual:**
    ```bash
    python -m venv venv
    ```

3. **Activa el entorno virtual:**
    - En Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - En macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. **Instala las dependencias:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Ejecución

1. **Activa el entorno virtual** (si no está activo).
2. **Ejecuta la aplicación:**
    ```bash
    python main.py
    ```

## Notas

- Todas las dependencias deben ser gestionadas con `uv`.
- Recuerda actualizar `requirements.txt` tras instalar nuevas librerías:
  ```bash
  uv pip freeze > requirements.txt
  ```
