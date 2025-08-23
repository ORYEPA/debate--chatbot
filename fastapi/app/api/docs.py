from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.openapi.utils import get_openapi
from swagger_ui_bundle import swagger_ui_3_path
from app.config import DOCS_VERSION

def configure_docs(app):
    app.mount("/_static", StaticFiles(directory=swagger_ui_3_path), name="swagger_static")

    @app.get("/docs", include_in_schema=False)
    def custom_swagger_ui():
        return get_swagger_ui_html(
            openapi_url=f"{app.openapi_url}?v={DOCS_VERSION}",
            title="Debate Chatbot - Swagger UI",
            swagger_js_url=f"/_static/swagger-ui-bundle.js?v={DOCS_VERSION}",
            swagger_css_url=f"/_static/swagger-ui.css?v={DOCS_VERSION}",
            swagger_favicon_url=f"/_static/favicon-32x32.png?v={DOCS_VERSION}",
        )

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version="1.0.0",
            description="Debate Chatbot API",
            routes=app.routes,
        )
        openapi_schema["openapi"] = "3.0.3"
        openapi_schema["servers"] = [{"url": "/"}]
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
