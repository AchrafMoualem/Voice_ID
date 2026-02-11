"""
Legacy entrypoint for running the Flask application.

The main application code now lives in the `app` package and is created
via `create_app` (see `app/__init__.py`). This file is kept as a thin
shim so existing `python app.py` workflows continue to work.
"""

from app import create_app


app = create_app()


if __name__ == "__main__":
    print("🚀 Démarrage du serveur Flask...")
    app.run(debug=app.config.get("DEBUG", True), host="0.0.0.0", port=5000)
