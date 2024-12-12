from torch import nn
import torch
from tqdm import tqdm

def training(model, train_dataloader, test_dataloader, num_epochs, optimizer, criterion, device='cpu'):
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        for batch in tqdm(train_dataloader):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)

        # Print progress every epoch
        # if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_loss, test_loss


# from torch import nn
# import torch
# Define the autoencoder model
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim + latent_dim)//2),
            nn.ReLU(),
            nn.Linear((input_dim + latent_dim)//2, (input_dim + latent_dim)//2),
            nn.ReLU(),
            nn.Linear((input_dim + latent_dim)//2, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (input_dim + latent_dim)//2),
            nn.ReLU(),
            nn.Linear((input_dim + latent_dim)//2, (input_dim + latent_dim)//2),
            nn.ReLU(),
            nn.Linear((input_dim + latent_dim)//2, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded