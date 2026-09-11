from fastapi import FastAPI

from app.code_unit_api import router as code_unit_router
from app.search_api import router as search_router

app = FastAPI(title="RepoMind", version="0.1.0")
app.include_router(code_unit_router)
app.include_router(search_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
