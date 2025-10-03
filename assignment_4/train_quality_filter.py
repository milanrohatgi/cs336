import fasttext

def main():
    model = fasttext.train_supervised(
        input="train.txt",
        lr=1.0,  
        epoch=25,         
        wordNgrams=2,     
        bucket=200000,     
        dim=100,             
        thread=4            
    )
    model.save_model("quality_classifier.bin")

if __name__ == "__main__":
    main()
