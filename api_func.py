from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
from make_pred import make_prediction
#from train_model import make_model_save


class User_input(BaseModel):
    x1 : int
    x2 : str
    x3 : int
    x4 : str
    x5 : int




app = FastAPI()

#@app.get("/")
#async def root():
#    return {"message": "Hello World"}



@app.post("/predict")
def get_pred(input:User_input):
    p1 = [input.x1, input.x2, input.x3, input.x4, input.x5]
    x = pd.DataFrame([p1],columns=['id','date','store_nbr','family','onpromotion'])
    res = make_prediction(x)
    return res


#@app.get("/train_model")
#def train_model():
#    make_model_save()
#    return {"Response": "Training completed."}


#if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=5049)