import json
import torch
import numpy as np


def reformat_data(filepath):
    all_data = json.load(open(filepath))

    new_data = []

    for i, d in enumerate(all_data):
        all_data[i]['prev_frame'] = json.loads(all_data[i]['prev_frame'])
        all_data[i]['frame'] = json.loads(all_data[i]['frame'])

        if i > 0 and all_data[i - 1]['video_id'] == all_data[i]['video_id']:
            # concat the previous two frames
            new_data.append({
                'video_id': d['video_id'],
                'prev_frame': all_data[i - 1]['prev_frame'] + all_data[i]['prev_frame'],
                'frame': d['frame'],
            })

    return new_data

def load_data(filepath):
    all_data = json.load(open(filepath))

    X = [d['prev_frame'] for d in all_data]
    Y = [d['frame'] for d in all_data]

    return np.array(X) / 8192, np.array(Y) / 8192

def train_test_loop(
        model, model_name,
        batch_size,
        n_epochs,
        Xtrain,
        Ytrain,
        Xtest,
        Ytest,
        small_model=None,
):
    import wandb

    wandb.init(
        project="10-707-proj",
        name=model_name,
    )

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    perm = np.random.shuffle(list(range(len(Xtrain))))
    trX = Xtrain[perm, :][0]
    trY = Ytrain[perm, :][0]
    trX = np.array([np.array(trX[i * batch_size:(i + 1) * batch_size]) for i in range(len(trX) // batch_size)])
    trY = np.array([np.array(trY[i * batch_size:(i + 1) * batch_size]) for i in range(len(trY) // batch_size)])

    trX = torch.Tensor(trX)
    trY = torch.Tensor(trY)
    Xtest = torch.Tensor(Xtest)
    Ytest = torch.Tensor(Ytest)

    best_train_loss = float("inf")
    best_test_loss = float("inf")

    # simulate training
    
    for epoch in range(n_epochs):

        if small_model is None:
            for i in range(len(trX)):
                optimizer.zero_grad()

                preds = model(trX[i])
                learner_loss = loss_fn(preds, trY[i])

                learner_loss.backward()

                optimizer.step()

                wandb.log({
                    "learner_loss": learner_loss,
                })

            preds = model.forward(trX)

            train_loss = loss_fn.forward(preds, trY)

            preds = model.forward(Xtest)
            test_loss = loss_fn.forward(preds, Ytest)

            print(f"Epoch {epoch} Train Loss: {train_loss} Test Loss: {test_loss}")

            # log metrics to wandb
            wandb.log({
                "train_loss": train_loss,
                "test_loss": test_loss,
            })
        else:
            with torch.no_grad():
                batch_losses = [
                    loss_fn(model(trX[i]), trY[i]) - loss_fn(small_model(trX[i]), trY[i])
                    for i in range(len(trX))
                ]
                
                batch_losses = torch.Tensor(batch_losses)

                batch_probs = torch.nn.Softmax()(batch_losses)

                batch_indices = torch.multinomial(batch_probs, len(trX), replacement=True)

            for i in batch_indices:
                optimizer.zero_grad()
                
                preds = model(trX[i])
                learner_loss = loss_fn(preds, trY[i])

                learner_loss.backward()

                optimizer.step()

                wandb.log({
                    "learner_loss": learner_loss,
                })

    #     best_train_loss = min(best_train_loss, train_loss)
    #     best_test_loss = min(best_test_loss, test_loss)

    # print(f"Best Train Loss: {best_train_loss}")
    # print(f"Best Test Loss: {best_test_loss}")
        
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
        

if __name__ == "__main__":
    # new_data = reformat_data("/grogu/user/mhzhou/youtube-curiosity/dataset/train.json")
    # json.dump(new_data, open("/grogu/user/mhzhou/youtube-curiosity/dataset/train-twoframes.json", "w"))

    # new_data = reformat_data("/grogu/user/mhzhou/youtube-curiosity/dataset/train-small.json")
    # json.dump(new_data, open("/grogu/user/mhzhou/youtube-curiosity/dataset/train-small-twoframes.json", "w"))

    # new_data = reformat_data("/grogu/user/mhzhou/youtube-curiosity/dataset/test.json")
    # json.dump(new_data, open("/grogu/user/mhzhou/youtube-curiosity/dataset/test-twoframes.json", "w"))

    Xtrain, Ytrain = load_data("/grogu/user/mhzhou/youtube-curiosity/dataset/train-twoframes.json")
    Xtrain_small, Ytrain_small = load_data("/grogu/user/mhzhou/youtube-curiosity/dataset/train-small-twoframes.json")

    Xtest, Ytest = load_data("/grogu/user/mhzhou/youtube-curiosity/dataset/test-twoframes.json")

    print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)

    MLP = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
    )

    train_test_loop(
        model=MLP,
        model_name="MLP_one_epoch",
        batch_size=5,
        n_epochs=1,
        Xtrain=Xtrain,
        Ytrain=Ytrain,
        Xtest=Xtest,
        Ytest=Ytest,
    )

    MLP_small = torch.nn.Sequential(
        torch.nn.Linear(512, 720),
        torch.nn.ReLU(),
        torch.nn.Linear(720, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
    )

    train_test_loop(
        model=MLP_small,
        model_name="MLP_small_one_epoch",
        batch_size=5,
        n_epochs=1,
        Xtrain=Xtrain_small,
        Ytrain=Ytrain_small,
        Xtest=Xtest,
        Ytest=Ytest,
    )

    # torch.save(MLP_small.state_dict(), "/grogu/user/mhzhou/youtube-curiosity/models/MLP_small.pt")

    # MLP_small.load_state_dict(torch.load("/grogu/user/mhzhou/youtube-curiosity/models/MLP_small.pt"))

    # train_test_loop(
    #     model=MLP,
    #     model_name="MLP_data_selection",
    #     small_model=MLP_small,
    #     batch_size=5,
    #     n_epochs=1,
    #     Xtrain=Xtrain,
    #     Ytrain=Ytrain,
    #     Xtest=Xtest,
    #     Ytest=Ytest,
    # )