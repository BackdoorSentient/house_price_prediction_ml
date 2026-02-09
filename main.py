from src.data_loader import load_data
from src.preprocesing import preprocess
from src.train import train_model
from src.evaluate import evaluate

def main():
    df=load_data("data/raw/data.csv")

    X,y=preprocess(df)

    model,X_test,y_test=train_model(X,y)

    y_pred=model.predict(X_test)
    metrics=evaluate(y_test,y_pred)

    print("model evaluation metrics:")
    for key,value in metrics.items():
        print(f"{key}:{value:.2f}")

if __name__=="__main__":
    main()