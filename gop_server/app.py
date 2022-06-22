from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app(db_url: str = None):
    from gop_server.db import init_db

    if db_url is not None:
        init_db(db_url)
    else:
        init_db()

    from gop_server.api import auth, zh_api, core_api, tts, feedback
    app = FastAPI()
    app.include_router(core_api.router)
    app.include_router(zh_api.router)
    app.include_router(auth.router)
    app.include_router(tts.router)
    app.include_router(feedback.router)

    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
