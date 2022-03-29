import torch
from tqdm import tqdm


def train(train_data, valid_data, epoch, model, device, loss_func, optimizer):
    min_loss = 100
    for e in range(epoch):
        for step, (x, y) in enumerate(tqdm(train_data)):
            x, y = x.to(device), y.to(device)
            out = model(x)[0]
            loss = loss_func(out, y)
            loss_val = loss.data.cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0
            for i, (v_x, v_y) in enumerate(valid_data):
                v_x, v_y = v_x.to(device), v_y.to(device)
                cur_valid_loss = loss_func(model(v_x)[0], v_y)
                total_valid_loss += cur_valid_loss
                avg_valid_loss = total_valid_loss / (i + 1)
                if cur_valid_loss < min_loss:
                    min_loss = cur_valid_loss
                    # 保存验证集上accuracy最高的模型
                    torch.save(model.state_dict(), 'cnn.pth')
        print("\nepoch: ", e, "| train_loss: %.4f" % loss_val, "| valid_loss: %.4f" % avg_valid_loss)
