"""Thin FastAPI wrapper exposing the same handler() over HTTP for local-dev and
manual checks. The Lambda runtime calls handler() directly; this is only an HTTP
shell over identical logic."""
from fastapi import FastAPI, Request

from pyspacer_function.handler import handler

app = FastAPI(title="pyspacer-inference (local-dev)")


@app.post("/classify")
async def classify_endpoint(request: Request):
    body = await request.json()
    return handler(body)
