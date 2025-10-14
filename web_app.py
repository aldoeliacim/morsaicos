"""
Backwards-compatible entry point for the redesigned photomosaic web app.

Legacy tooling expects a module-level `app` object, so we import the new app
factory from `app.main` and expose it here.
"""

from app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
