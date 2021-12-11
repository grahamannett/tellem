from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/data")
def upload(data):
    return
    # return {"item_id": , "q": q}