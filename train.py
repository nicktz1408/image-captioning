import torch
from utils import bleu_score_sum, create_lookup_table

def generate_caption(model, image, vocab_to_idx, max_len, device='cuda'):
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0) # shape: (1, C, H, W)
        caption = [ vocab_to_idx['<start>'] ]
        
        for _ in range(max_len - 1):
            input_seq = torch.tensor(caption).unsqueeze(0).to(device)  # (1, seq_len)

            outputs = model(image, input_seq)  # (1 * seq_len, vocab_size)
            next_token = outputs[-1].argmax().item()

            caption.append(next_token)

            if next_token == vocab_to_idx['<end>']:
                break
        
        while len(caption) < max_len:
            caption.append(vocab_to_idx['<pad>'])

    return caption

def eval(model, test_loader, device):
    model.eval()

    total_bleu_score = 0.0
    total_samples = 0

    _, vocab_to_idx = create_lookup_table()

    with torch.no_grad():
        for images, target_captions in test_loader:
            images, target_captions = images.to(device), target_captions.to(device)

            output_captions = []

            for img_idx in range(images.size(0)):
                caption = generate_caption(model, images[img_idx], vocab_to_idx, 2, device)
                
                output_captions.append(caption)

            output_captions = torch.tensor(output_captions).to(device)

            total_bleu_score += bleu_score_sum(output_captions, target_captions)
            total_samples += images.size(0)
    
    avg_bleu_score = total_bleu_score / total_samples
    print(f"Test BLEU Score: {avg_bleu_score:.4f}")

def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=5):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for images, captions in train_loader:
            images, captions = images.to(device), captions.to(device)

            inputs = captions[:, :-1] # inputs to the RNN, the last one is the final output
            targets = captions[:, 1:] # targets for the outputs of the RNN (ie start from the 2nd word)

            optimizer.zero_grad()

            outputs = model(images, inputs) # (B * seq_len-1, vocab_size)

            loss = criterion(outputs, targets.contiguous().view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

        eval(model, test_loader, device)