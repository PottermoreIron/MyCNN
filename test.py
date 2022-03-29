import torch
from painter import Painter


def test_acc(x, y, model):
    pred_y = torch.max(model(x)[0], 1)[1].data.numpy()
    success = len([label for label in pred_y if label in y])
    acc = success / len(y)
    print(acc)


def test_image(model):
    painter = Painter(model)
    painter.mainloop()
