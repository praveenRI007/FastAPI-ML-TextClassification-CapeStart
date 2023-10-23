from fastapi import FastAPI, Depends , Request , Form
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from article_predict import predict_a
import uvicorn

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]


templates = Jinja2Templates(directory="templates")

#app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_main(request : Request, response_class=HTMLResponse):
    return templates.TemplateResponse("main.html", {"request": request})


@app.post("/predict")
async def predict(request : Request,text: str = Form(...)):
    msg = predict_a(text)
    return templates.TemplateResponse("main.html", {"request": request, "msg": str(msg[0])})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)