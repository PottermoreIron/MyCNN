def train(data, epoch, model, loss_func, optimizer):
    for e in range(epoch):
        for step, (x, y) in enumerate(data):
            out = model(x)[0]
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print("epoch: ", e, "| train_loss: %.4f" % loss.data.numpy())
