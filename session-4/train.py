import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from app.model import SentimentAnalysis
from utils import YelpReviewPolarityDatasetLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataloader):
    model.train()

    # Train the model
    train_loss = 0
    train_acc = 0
    for text, offsets, label in dataloader:
        # TODO complete the training code. The inputs of the model are text and offsets
        # Compute model output
        output = model(text, offsets)
        # Calculate cross-entropy loss
        loss = criterion(output, label)

        train_loss += loss.item() * len(output)
        train_acc += (output.argmax(1) == label).sum().item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(dataloader.dataset), train_acc / len(dataloader.dataset)


def test(dataloader: DataLoader):
    model.eval()

    loss = 0
    acc = 0
    with torch.no_grad():
        for text, offsets, label in dataloader:
            output = model(text, offsets)
            loss_ = criterion(output, label)

            loss += loss_.item() * len(output)
            acc += (output.argmax(1) == label).sum().item()

    return loss / len(dataloader.dataset), acc / len(dataloader.dataset)


if __name__ == "__main__":

    # Hyperparameters
    NGRAMS = 1  # 2 or 3 will be better but slower.
    BATCH_SIZE = 16
    EMBED_DIM = 32
    N_EPOCHS = 2  # 5 would be ideal, but slower.

    # Load the dataset
    yelp_loader = YelpReviewPolarityDatasetLoader(NGRAMS, BATCH_SIZE, device=device)

    # Retrieve train, validation and test datasets
    train_val_dataset = yelp_loader.get_train_val_dataset()
    test_dataset = yelp_loader.get_test_dataset()

    # Retrieve vocabulary size and number of classes
    VOCAB_SIZE = yelp_loader.get_vocab_size()
    NUM_CLASS = yelp_loader.get_num_classes()

    # Load the model
    # TODO load the model
    model = SentimentAnalysis(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, num_class=NUM_CLASS)
        
    # We will use CrossEntropyLoss even though we are doing binary classification 
    # because the code is ready to also work for many classes
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Setup optimizer and LR scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    # Split train and val datasets
    # TODO split `train_val_dataset` in `train_dataset` and `valid_dataset`. The size of train dataset should be 95%

    # Assuming dataset has 100 samples
    total_length = len(train_val_dataset)
    train_length = int(0.95 * total_length)
    valid_length = total_length - train_length

    train_dataset, valid_dataset = random_split(dataset=train_val_dataset, lengths=[train_length, valid_length])
    
    # DataLoader needs an special function to generate the batches. 
    # Since we will have inputs of varying size, we will concatenate 
    # all the inputs in a single vector and create a vector with the "offsets" between inputs.
    # You can check the `generate_batch` function for more info.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yelp_loader.generate_batch)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yelp_loader.generate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yelp_loader.generate_batch)


    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train(train_loader)
        valid_loss, valid_acc = test(val_loader)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print(f"Epoch: {epoch + 1},  | time in {mins} minutes, {secs} seconds")
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    print("Training finished")

    test_loss, test_acc = test(test_loader)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

    # Now save the artifacts of the training
    savedir = "app/state_dict.pt"
    print(f"Saving checkpoint to {savedir}...")
    # We can save everything we will need later in the checkpoint.
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": yelp_loader.vocab,
        "vocab_size": VOCAB_SIZE,
        "ngrams": NGRAMS,
        "embed_dim": EMBED_DIM,
        "num_class": NUM_CLASS,
    }
    torch.save(checkpoint, savedir)
